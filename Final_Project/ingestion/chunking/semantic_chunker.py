"""Semantic chunking (embedding-distance breakpoints), not fixed-size.

Pipeline (Kamradt-style, made deterministic + testable):
  1. Split text into sentences (abbreviation-aware: e.g., "e.g.", "M.Tech.").
  2. For each sentence build a context window (sentence ± buffer_size neighbors)
     and embed each window (stabilises the signal vs. embedding lone sentences).
  3. Cosine DISTANCE between consecutive windows.
  4. Breakpoint = distance above the `breakpoint_percentile` (default 95th).
  5. Cut at breakpoints; merge chunks < min_chars; hard-split chunks > max_chars
     on sentence boundaries.

The embedder is INJECTABLE (`Callable[[list[str]], np.ndarray]`) so the splitter
can be unit-tested with a deterministic mock. `BGEEmbedder` provides the real
bge-base embedder (lazy-loaded, normalized).
"""

from __future__ import annotations

import logging
import re
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)

Embedder = Callable[[list[str]], np.ndarray]

# Abbreviations whose trailing period must NOT end a sentence.
_ABBREVIATIONS = [
    "e.g.", "i.e.", "etc.", "cf.", "vs.", "viz.", "Dr.", "Mr.", "Mrs.", "Ms.",
    "Prof.", "Sr.", "Jr.", "St.", "No.", "Fig.", "Eq.", "Ref.", "Sec.",
    "B.Tech.", "M.Tech.", "Ph.D.", "B.Sc.", "M.Sc.", "B.E.", "M.E.", "approx.",
]
# Placeholder for the period inside an abbreviation while splitting.
_DOT = "․"  # ONE DOT LEADER — visually a dot, won't be matched as a boundary


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, protecting common abbreviations."""
    if not text or not text.strip():
        return []
    protected = text
    for abbr in _ABBREVIATIONS:
        protected = protected.replace(abbr, abbr.replace(".", _DOT))
    # Also protect single-letter initials like "A. H Shapiro" and decimals "3.14".
    protected = re.sub(r"(?<=\b[A-Z])\.", _DOT, protected)
    protected = re.sub(r"(?<=\d)\.(?=\d)", _DOT, protected)

    # Split after sentence-ending punctuation followed by whitespace.
    parts = re.split(r"(?<=[.!?])\s+", protected)
    sentences = [p.replace(_DOT, ".").strip() for p in parts if p.strip()]
    return sentences


def _cosine_distances(embeddings: np.ndarray) -> np.ndarray:
    """Distance between consecutive rows. Assumes (approximately) unit-norm rows."""
    a = embeddings[:-1]
    b = embeddings[1:]
    # Normalize defensively in case the injected embedder isn't unit-norm.
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    sims = np.sum(a * b, axis=1)
    return 1.0 - sims


def _post_process(chunks: list[str], min_chars: int, max_chars: int) -> list[str]:
    """Merge too-small chunks forward; hard-split too-large chunks on sentences."""
    # Merge small chunks into the next (or previous) chunk.
    merged: list[str] = []
    for ch in chunks:
        if merged and len(ch) < min_chars:
            merged[-1] = f"{merged[-1]} {ch}".strip()
        else:
            merged.append(ch)
    # If the FIRST chunk ended up small, fold it into the second.
    if len(merged) > 1 and len(merged[0]) < min_chars:
        merged[1] = f"{merged[0]} {merged[1]}".strip()
        merged = merged[1:]

    # Hard-split oversized chunks on sentence boundaries.
    out: list[str] = []
    for ch in merged:
        if len(ch) <= max_chars:
            out.append(ch)
            continue
        sents = split_sentences(ch) or [ch]
        cur = ""
        for s in sents:
            if cur and len(cur) + len(s) + 1 > max_chars:
                out.append(cur.strip())
                cur = s
            else:
                cur = f"{cur} {s}".strip()
        if cur:
            out.append(cur.strip())
    return out


def semantic_chunk(
    text: str,
    embed_fn: Embedder,
    buffer_size: int = 1,
    breakpoint_percentile: float = 95.0,
    min_chars: int = 200,
    max_chars: int = 1200,
) -> list[str]:
    """Split `text` into semantically coherent chunks using embedding distances."""
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        stripped = text.strip()
        return [stripped] if stripped else []

    # Context windows.
    windows = []
    for i in range(len(sentences)):
        lo = max(0, i - buffer_size)
        hi = min(len(sentences), i + buffer_size + 1)
        windows.append(" ".join(sentences[lo:hi]))

    embeddings = np.asarray(embed_fn(windows), dtype=np.float64)
    distances = _cosine_distances(embeddings)
    if distances.size == 0:
        return [text.strip()]

    threshold = float(np.percentile(distances, breakpoint_percentile))
    # Breakpoints: gaps whose distance exceeds the percentile threshold. Using
    # strict `>` keeps near-uniform texts as a single chunk.
    breakpoints = [i for i, d in enumerate(distances) if d > threshold]

    # Cut sentences into groups after each breakpoint index.
    chunks: list[str] = []
    start = 0
    for bp in breakpoints:
        chunks.append(" ".join(sentences[start : bp + 1]).strip())
        start = bp + 1
    chunks.append(" ".join(sentences[start:]).strip())
    chunks = [c for c in chunks if c]

    return _post_process(chunks, min_chars, max_chars)


class BGEEmbedder:
    """Real embedder: BAAI/bge-base-en-v1.5 via sentence-transformers.

    Lazy-loads the model on first call so importing this module stays cheap and
    unit tests can avoid the heavy dependency entirely by injecting a mock.
    """

    def __init__(self, model_name: str | None = None, device: str | None = None):
        from config import get_settings

        s = get_settings()
        self.model_name = model_name or s.embed_model
        self.device = device or s.resolve_device()
        self._model = None

    def _ensure(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedder %s on %s", self.model_name, self.device)
            self._model = SentenceTransformer(self.model_name, device=self.device)

    def __call__(self, texts: list[str]) -> np.ndarray:
        self._ensure()
        return self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
