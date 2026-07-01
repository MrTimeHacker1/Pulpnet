"""Semantic response cache (Redis with graceful in-process fallback).

- Exact-key get/set (Redis if reachable, else an in-process dict with TTL).
- Semantic cache: embed the query, compare against recently-cached query
  embeddings (in-process index), and return the cached answer on cosine
  similarity >= threshold.
- Exposes real hit/miss counters so a cost-reduction number can be reported from
  actual usage (not hardcoded).

Async interface (redis.asyncio). If Redis is unreachable at first use, it
transparently degrades to the local fallback — the API keeps working.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from config import get_settings
from ingestion.chunking.semantic_chunker import BGEEmbedder, Embedder

logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    semantic_hits: int = 0
    exact_hits: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "semantic_hits": self.semantic_hits,
            "exact_hits": self.exact_hits,
            "total": self.total,
            "hit_rate": round(self.hit_rate, 4),
        }


class SemanticCache:
    def __init__(
        self,
        embedder: Embedder | None = None,
        redis_url: str | None = None,
        threshold: float | None = None,
        ttl_seconds: int | None = None,
        max_semantic_entries: int = 512,
    ):
        s = get_settings()
        self.embedder = embedder or BGEEmbedder()
        self.redis_url = redis_url if redis_url is not None else s.redis_url
        self.threshold = threshold if threshold is not None else s.semantic_cache_threshold
        self.ttl = ttl_seconds if ttl_seconds is not None else s.cache_ttl_seconds
        self.stats = CacheStats()

        self._redis = None
        self._redis_ready = False
        # in-process fallback store: key -> (answer_json, expiry_ts)
        self._local: dict[str, tuple[str, float]] = {}
        # semantic index (always in-process): deque of (vec, key)
        self._sem_index: deque[tuple[np.ndarray, str]] = deque(maxlen=max_semantic_entries)

    # ---- Redis lifecycle ----
    async def _ensure_redis(self) -> None:
        if self._redis_ready or not self.redis_url:
            return
        try:
            import redis.asyncio as aioredis

            self._redis = aioredis.from_url(self.redis_url, decode_responses=True)
            await self._redis.ping()
            logger.info("Redis cache connected: %s", self.redis_url)
        except Exception as e:
            logger.warning("Redis unavailable (%s); using in-process cache.", e)
            self._redis = None
        self._redis_ready = True

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()

    # ---- key/value primitives ----
    @staticmethod
    def _key(query: str) -> str:
        return f"ragcache:{query.strip().lower()}"

    async def _raw_get(self, key: str) -> str | None:
        await self._ensure_redis()
        if self._redis is not None:
            return await self._redis.get(key)
        item = self._local.get(key)
        if item is None:
            return None
        value, expiry = item
        if expiry < time.time():
            self._local.pop(key, None)
            return None
        return value

    async def _raw_set(self, key: str, value: str) -> None:
        await self._ensure_redis()
        if self._redis is not None:
            await self._redis.set(key, value, ex=self.ttl)
        else:
            self._local[key] = (value, time.time() + self.ttl)

    # ---- public API ----
    async def get_exact(self, query: str) -> dict | None:
        raw = await self._raw_get(self._key(query))
        return json.loads(raw) if raw else None

    async def get(self, query: str) -> dict | None:
        """Exact then semantic lookup. Updates stats. Returns cached record or None."""
        # 1) exact
        rec = await self.get_exact(query)
        if rec is not None:
            self.stats.hits += 1
            self.stats.exact_hits += 1
            rec["_cache"] = "exact"
            return rec

        # 2) semantic
        if self._sem_index:
            qvec = np.asarray(self.embedder([query])[0], dtype=np.float64)
            qvec = qvec / (np.linalg.norm(qvec) + 1e-12)
            best_sim, best_key = -1.0, None
            for vec, key in self._sem_index:
                sim = float(np.dot(qvec, vec))
                if sim > best_sim:
                    best_sim, best_key = sim, key
            if best_key is not None and best_sim >= self.threshold:
                raw = await self._raw_get(best_key)
                if raw:
                    self.stats.hits += 1
                    self.stats.semantic_hits += 1
                    rec = json.loads(raw)
                    rec["_cache"] = "semantic"
                    rec["_similarity"] = round(best_sim, 4)
                    return rec

        self.stats.misses += 1
        return None

    async def set(self, query: str, record: dict) -> None:
        key = self._key(query)
        await self._raw_set(key, json.dumps(record))
        qvec = np.asarray(self.embedder([query])[0], dtype=np.float64)
        qvec = qvec / (np.linalg.norm(qvec) + 1e-12)
        self._sem_index.append((qvec, key))
