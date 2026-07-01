"""Embed Chunks with bge-base and upsert into Qdrant.

Qdrant runs in embedded local-file mode by default (no server) — set QDRANT_URL
to use a server instead. Collection creation is idempotent via
`collection_exists` + `create_collection` (NOT the deprecated
`recreate_collection`). Point ids are the deterministic `chunk_id`, so
re-ingestion overwrites rather than duplicates.
"""

from __future__ import annotations

import logging
from typing import Iterable

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from config import get_settings
from ingestion.chunking.semantic_chunker import BGEEmbedder, Embedder
from ingestion.schema import Chunk

logger = logging.getLogger(__name__)

_client: QdrantClient | None = None
_embedder: BGEEmbedder | None = None


def get_qdrant_client() -> QdrantClient:
    """Shared Qdrant client (local-file mode unless QDRANT_URL is set)."""
    global _client
    if _client is None:
        s = get_settings()
        if s.qdrant_url:
            _client = QdrantClient(url=s.qdrant_url)
            logger.info("Qdrant client → server %s", s.qdrant_url)
        else:
            _client = QdrantClient(path=s.qdrant_path)
            logger.info("Qdrant client → local path %s", s.qdrant_path)
    return _client


def close_qdrant_client() -> None:
    """Close and drop the shared Qdrant client.

    Critical on this host: an OPEN Qdrant local-mode client and torch inference in
    the same process segfault when both are active. Callers that only need a
    one-shot read (scroll/count) should close the client afterwards so no
    Qdrant-native state coexists with later torch inference during serving.
    """
    global _client
    if _client is not None:
        try:
            _client.close()
        except Exception:
            pass
        _client = None


def get_embedder() -> BGEEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = BGEEmbedder()
    return _embedder


def ensure_collection(client: QdrantClient | None = None) -> None:
    client = client or get_qdrant_client()
    s = get_settings()
    if not client.collection_exists(s.qdrant_collection):
        client.create_collection(
            collection_name=s.qdrant_collection,
            vectors_config=VectorParams(size=s.embed_dim, distance=Distance.COSINE),
        )
        logger.info("Created Qdrant collection %s (dim=%d)", s.qdrant_collection, s.embed_dim)


def index_chunks(
    chunks: Iterable[Chunk],
    client: QdrantClient | None = None,
    embed_fn: Embedder | None = None,
    batch_size: int = 64,
) -> int:
    """Embed and upsert chunks. Returns the number of points upserted."""
    client = client or get_qdrant_client()
    embed_fn = embed_fn or get_embedder()
    s = get_settings()
    ensure_collection(client)

    chunks = list(chunks)
    total = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        vectors = embed_fn([c.text for c in batch])
        points = [
            PointStruct(id=c.chunk_id, vector=list(map(float, vec)), payload=c.to_payload())
            for c, vec in zip(batch, vectors)
        ]
        client.upsert(collection_name=s.qdrant_collection, points=points)
        total += len(points)
        logger.info("Upserted %d/%d points", total, len(chunks))
    return total


def index_precomputed(
    items: list[tuple[Chunk, list[float]]],
    client: QdrantClient | None = None,
    batch_size: int = 256,
) -> int:
    """Upsert chunks whose vectors are ALREADY computed (no embedding here).

    This is the second phase of the decoupled pipeline: by the time we touch
    Qdrant, no torch inference is running, which avoids the native OpenMP/segfault
    interaction between torch and the Qdrant local-mode libs.
    """
    client = client or get_qdrant_client()
    s = get_settings()
    ensure_collection(client)
    total = 0
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        points = [
            PointStruct(id=c.chunk_id, vector=list(map(float, vec)), payload=c.to_payload())
            for c, vec in batch
        ]
        client.upsert(collection_name=s.qdrant_collection, points=points)
        total += len(points)
    return total


def count_points(client: QdrantClient | None = None) -> int:
    client = client or get_qdrant_client()
    s = get_settings()
    if not client.collection_exists(s.qdrant_collection):
        return 0
    return client.count(collection_name=s.qdrant_collection).count
