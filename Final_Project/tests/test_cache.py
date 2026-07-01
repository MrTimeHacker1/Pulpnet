"""Tests for the semantic cache (in-process fallback, mock embedder)."""

from __future__ import annotations

import asyncio

import numpy as np

from cache.redis_cache import SemanticCache


def _topic_embedder(texts: list[str]) -> np.ndarray:
    """Map any query mentioning a course code to that code's vector, so
    paraphrases about the same course are near-identical."""
    out = []
    for t in texts:
        tl = t.lower()
        if "ce212" in tl:
            v = np.array([1.0, 0.0, 0.0])
        elif "cs340" in tl:
            v = np.array([0.0, 1.0, 0.0])
        else:
            v = np.array([0.0, 0.0, 1.0])
        out.append(v)
    return np.asarray(out)


def _cache() -> SemanticCache:
    # redis_url=None forces the in-process fallback path.
    return SemanticCache(embedder=_topic_embedder, redis_url=None, threshold=0.95)


def test_exact_hit():
    c = _cache()

    async def run():
        await c.set("What is CE212?", {"answer": "Environment and Sustainability"})
        rec = await c.get("What is CE212?")
        assert rec is not None and rec["_cache"] == "exact"
        assert c.stats.exact_hits == 1

    asyncio.run(run())


def test_semantic_hit_on_paraphrase():
    c = _cache()

    async def run():
        await c.set("Tell me about CE212", {"answer": "Env & Sustainability"})
        # Different wording, same course → semantic hit (not exact).
        rec = await c.get("Give details on the CE212 course please")
        assert rec is not None
        assert rec["_cache"] == "semantic"
        assert rec["_similarity"] >= 0.95
        assert c.stats.semantic_hits == 1

    asyncio.run(run())


def test_miss_increments_stats():
    c = _cache()

    async def run():
        await c.set("About CE212", {"answer": "x"})
        rec = await c.get("What about CS340?")  # different topic vector
        assert rec is None
        assert c.stats.misses == 1
        assert 0.0 <= c.stats.hit_rate <= 1.0

    asyncio.run(run())
