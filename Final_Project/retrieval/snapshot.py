"""Retrieval snapshot — decouples serving from Qdrant.

On this host, importing/using `qdrant-client` in the same process as torch
inference causes intermittent native segfaults. So ingestion writes a compact
snapshot (ids + texts + payloads + vectors) that the serving process loads
WITHOUT importing qdrant-client at all. Qdrant remains the canonical, idempotent
vector store; this snapshot is the fast, torch-safe read path for serving.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def save_snapshot(items, path: str | Path) -> int:
    """Persist [(Chunk, vector)] as a snapshot. Returns the number of rows."""
    ids, texts, payloads, vectors = [], [], [], []
    for chunk, vec in items:
        ids.append(chunk.chunk_id)
        texts.append(chunk.text)
        payloads.append(chunk.to_payload())
        vectors.append(vec)
    mat = np.asarray(vectors, dtype=np.float32) if vectors else np.zeros((0, 0), np.float32)
    data = {"ids": ids, "texts": texts, "payloads": payloads, "vectors": mat}
    path = Path(path)
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Wrote retrieval snapshot (%d rows) → %s", len(ids), path)
    return len(ids)


def load_snapshot(path: str | Path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)
