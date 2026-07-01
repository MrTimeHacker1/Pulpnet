"""Hybrid retrieval: dense (Qdrant) + sparse (BM25), fused with RRF.

Dense search uses Qdrant `query_points` (NOT the deprecated `search`). BM25 runs
in-process over the same corpus (scrolled out of Qdrant payloads). Results are
fused with Reciprocal Rank Fusion. An optional metadata filter restricts results
by `doc_type` and/or `department`.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from config import get_settings
from ingestion.chunking.semantic_chunker import BGEEmbedder, Embedder

# NOTE: qdrant-client is deliberately NOT imported at module top. On this host,
# importing qdrant-client into a process that also runs torch inference causes
# intermittent native segfaults. The serving path loads a snapshot and never
# touches qdrant; the Qdrant fallback imports it lazily (used only by tooling
# that isn't running torch).

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
RRF_K = 60


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    payload: dict[str, Any]
    score: float  # fused RRF score


class HybridSearcher:
    """Loads the corpus + vectors from Qdrant ONCE, then searches fully in-process.

    Qdrant (local-file mode) is touched exactly once, at init, to scroll all points
    with their vectors. Query-time dense search is pure numpy cosine over the loaded
    matrix — so no Qdrant-native calls run interleaved with torch inference during
    serving (that interleaving segfaults on this host). BM25 is also in-process.
    """

    def __init__(self, client=None, embedder: Embedder | None = None):
        self.embedder = embedder or BGEEmbedder()
        self.settings = get_settings()
        self.ids: list[str] = []
        self.texts: list[str] = []
        self.payloads: list[dict[str, Any]] = []
        self.vectors: np.ndarray = np.zeros((0, self.settings.embed_dim), dtype=np.float32)
        self.bm25: BM25Okapi | None = None
        # Prefer the torch-safe snapshot (no qdrant import). Only fall back to
        # Qdrant when no snapshot exists (e.g. tests, or serving before ingest).
        from retrieval.snapshot import load_snapshot

        snap = load_snapshot(self.settings.retrieval_snapshot)
        if snap is not None:
            self._load_from_snapshot(snap)
        else:
            logger.warning("No retrieval snapshot; loading from Qdrant (imports qdrant-client).")
            self._load_corpus(client)

    def _load_from_snapshot(self, snap: dict) -> None:
        self.ids = list(snap["ids"])
        self.texts = list(snap["texts"])
        self.payloads = list(snap["payloads"])
        mat = np.asarray(snap["vectors"], dtype=np.float32)
        if mat.size:
            mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
            self.vectors = mat
            self.bm25 = BM25Okapi([_tokenize(t) for t in self.texts])
        logger.info("HybridSearcher loaded %d chunks from snapshot", len(self.ids))

    def _load_corpus(self, client=None) -> None:
        # Qdrant fallback (no snapshot present). Imports qdrant-client lazily;
        # only safe when torch inference is NOT also running in this process.
        from ingestion.embed_and_index import close_qdrant_client, get_qdrant_client

        client = client or get_qdrant_client()
        coll = self.settings.qdrant_collection
        if not client.collection_exists(coll):
            logger.warning("Collection %s missing; hybrid searcher is empty.", coll)
            close_qdrant_client()
            return
        vectors: list[list[float]] = []
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=coll, limit=512, offset=offset,
                with_payload=True, with_vectors=True,
            )
            for p in points:
                payload = p.payload or {}
                self.ids.append(str(p.id))
                self.texts.append(payload.get("text", ""))
                self.payloads.append(payload)
                vectors.append(p.vector)
            if offset is None:
                break
        close_qdrant_client()
        if vectors:
            mat = np.asarray(vectors, dtype=np.float32)
            # bge vectors are already normalized; normalize defensively for cosine=dot.
            mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
            self.vectors = mat
            self.bm25 = BM25Okapi([_tokenize(t) for t in self.texts])
        logger.info("HybridSearcher loaded %d chunks", len(self.ids))

    def _mask(self, doc_type: str | None, department: str | None) -> np.ndarray:
        mask = np.ones(len(self.ids), dtype=bool)
        if doc_type:
            mask &= np.array([p.get("doc_type") == doc_type for p in self.payloads])
        if department:
            mask &= np.array([p.get("department") == department for p in self.payloads])
        return mask

    def search(
        self,
        query: str,
        top_k: int | None = None,
        doc_type: str | None = None,
        department: str | None = None,
    ) -> list[RetrievedChunk]:
        top_k = top_k or self.settings.retrieve_top_k
        if not self.ids:
            return []

        mask = self._mask(doc_type, department)
        cand_idx = np.nonzero(mask)[0]
        if cand_idx.size == 0:
            return []

        # --- Dense (in-process cosine = dot over normalized vectors) ---
        qvec = np.asarray(self.embedder([query])[0], dtype=np.float32)
        qvec /= np.linalg.norm(qvec) + 1e-12
        sims = self.vectors[cand_idx] @ qvec
        dense_order = cand_idx[np.argsort(-sims)][: top_k * 2]
        dense_ranking = [self.ids[i] for i in dense_order]

        # --- Sparse (BM25 over full corpus, then filter) ---
        sparse_ranking: list[str] = []
        if self.bm25 is not None:
            bm = self.bm25.get_scores(_tokenize(query))
            bm_masked = np.where(mask, bm, -np.inf)
            for i in np.argsort(-bm_masked):
                if not np.isfinite(bm_masked[i]):
                    break
                sparse_ranking.append(self.ids[i])
                if len(sparse_ranking) >= top_k * 2:
                    break

        # --- Reciprocal Rank Fusion ---
        fused: dict[str, float] = {}
        for ranking in (dense_ranking, sparse_ranking):
            for rank, cid in enumerate(ranking):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)

        payload_by_id = {self.ids[i]: self.payloads[i] for i in range(len(self.ids))}
        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        out: list[RetrievedChunk] = []
        for cid, score in ordered:
            payload = payload_by_id.get(cid, {})
            out.append(
                RetrievedChunk(chunk_id=cid, text=payload.get("text", ""), payload=payload, score=score)
            )
        return out
