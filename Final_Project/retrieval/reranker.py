"""Cross-encoder reranking with BAAI/bge-reranker-base.

Takes the hybrid-retrieved candidates (top ~20) and reorders them by a
query-document relevance score, returning the top `rerank_top_k` (default 5).
The top score is consumed by the agent's sufficiency check.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from config import get_settings
from retrieval.hybrid_search import RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass
class RerankedChunk:
    chunk_id: str
    text: str
    payload: dict[str, Any]
    rerank_score: float


class Reranker:
    """Lazy-loaded bge-reranker-base cross-encoder."""

    def __init__(self, model_name: str | None = None, device: str | None = None):
        s = get_settings()
        self.model_name = model_name or s.reranker_model
        self.device = device or s.resolve_device()
        self._model = None

    def _ensure(self) -> None:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("Loading reranker %s on %s", self.model_name, self.device)
            self._model = CrossEncoder(self.model_name, device=self.device)

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int | None = None
    ) -> list[RerankedChunk]:
        top_k = top_k or get_settings().rerank_top_k
        if not candidates:
            return []
        self._ensure()
        pairs = [[query, c.text] for c in candidates]
        scores = self._model.predict(pairs, show_progress_bar=False)
        ranked = sorted(
            zip(candidates, scores), key=lambda cs: float(cs[1]), reverse=True
        )
        return [
            RerankedChunk(
                chunk_id=c.chunk_id, text=c.text, payload=c.payload, rerank_score=float(s)
            )
            for c, s in ranked[:top_k]
        ]
