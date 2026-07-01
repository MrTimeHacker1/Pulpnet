"""Ingestion orchestrator (phased, segfault-safe).

Dispatches by folder:
  - courses/            → course_catalog_parser → Qdrant
  - academic_documents/ → academic_doc_parser (semantic chunk) → Qdrant
  - timetable/          → timetable_parser → Postgres (timetable_slots), NOT embedded

IMPORTANT — phased design: torch-inference (embedding) and Qdrant-local writes
segfault when interleaved in one process on this host. So we EMBED EVERYTHING
FIRST (Phase A, torch active), then upsert the precomputed vectors to Qdrant by
default in a SEPARATE subprocess (Phase B, torch idle/absent). Set
RAG_INPROC_INDEX=1 to do Phase B in-process instead.

Idempotent: Qdrant points key on deterministic chunk_id and a PDF's prior points
are deleted before re-indexing; timetable rows are delete-then-insert per source;
a `documents` row is upserted per PDF.

Run:  python -m ingestion.run_ingestion
"""

from __future__ import annotations

import logging
import os
import pickle
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

from sqlalchemy import delete, select

from config import configure_logging, get_settings
from db.models import Document, TimetableSlot
from db.session import init_db, session_scope
from ingestion.parsers import (
    academic_doc_parser,
    course_catalog_parser,
    timetable_parser,
)
from ingestion.schema import Chunk

logger = logging.getLogger(__name__)


# ----------------------------- Phase A: embed -----------------------------
def embed_all() -> tuple[list[tuple[Chunk, list[float]]], dict[str, tuple[str, str | None]]]:
    """Parse + embed every course/manual chunk. Returns (items, doc_meta).

    items     = [(chunk, vector), ...]
    doc_meta  = {source_pdf: (doc_type, department)}
    """
    from ingestion.chunking.semantic_chunker import BGEEmbedder

    s = get_settings()
    embedder = BGEEmbedder()
    embedder(["warmup"])

    chunks: list[Chunk] = []
    doc_meta: dict[str, tuple[str, str | None]] = {}

    for pdf in sorted(s.courses_dir.glob("*.pdf")):
        c, dept = course_catalog_parser.parse_to_chunks(pdf)
        chunks.extend(c)
        doc_meta[pdf.name] = ("course_catalog", dept)
        logger.info("Parsed %s: %d course chunks", pdf.name, len(c))

    for pdf in sorted(s.academic_docs_dir.glob("*.pdf")):
        c = academic_doc_parser.parse_to_chunks(pdf, embed_fn=embedder)
        chunks.extend(c)
        doc_meta[pdf.name] = ("academic_manual", None)
        logger.info("Parsed %s: %d manual chunks", pdf.name, len(c))

    logger.info("Embedding %d chunks…", len(chunks))
    vectors = embedder([c.text for c in chunks])
    items = [(c, list(map(float, v))) for c, v in zip(chunks, vectors)]
    return items, doc_meta


# ----------------------------- Phase B: index -----------------------------
def index_all(items, doc_meta) -> dict[str, int]:
    """Upsert precomputed vectors to Qdrant + write documents rows. No torch here."""
    from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue

    from ingestion.embed_and_index import (
        count_points,
        get_qdrant_client,
        index_precomputed,
    )

    client = get_qdrant_client()
    s = get_settings()

    # Group items by source for idempotent per-PDF replace.
    by_source: dict[str, list] = {}
    for c, v in items:
        by_source.setdefault(c.source_pdf, []).append((c, v))

    if client.collection_exists(s.qdrant_collection):
        for src in by_source:
            client.delete(
                collection_name=s.qdrant_collection,
                points_selector=FilterSelector(
                    filter=Filter(must=[FieldCondition(key="source_pdf", match=MatchValue(value=src))])
                ),
            )

    counts: dict[str, int] = {}
    for src, group in by_source.items():
        n = index_precomputed(group, client=client)
        counts[src] = n
        doc_type, dept = doc_meta[src]
        _upsert_document(src, doc_type, dept, n)
        logger.info("Indexed %s: %d points", src, n)

    counts["_qdrant_total"] = count_points(client)
    return counts


def _upsert_document(source_pdf, doc_type, department, n_chunks) -> None:
    with session_scope() as ses:
        doc = ses.execute(
            select(Document).where(Document.source_pdf == source_pdf)
        ).scalar_one_or_none()
        if doc is None:
            ses.add(Document(source_pdf=source_pdf, doc_type=doc_type,
                             department=department, n_chunks=n_chunks,
                             ingested_at=datetime.now(timezone.utc)))
        else:
            doc.doc_type, doc.department, doc.n_chunks = doc_type, department, n_chunks
            doc.ingested_at = datetime.now(timezone.utc)


# ----------------------------- timetable (no torch) -----------------------------
def ingest_timetable() -> int:
    s = get_settings()
    total = 0
    for pdf in sorted(s.timetable_dir.glob("*.pdf")):
        rows = timetable_parser.parse_to_rows(pdf)
        with session_scope() as ses:
            ses.execute(delete(TimetableSlot).where(TimetableSlot.source_pdf == pdf.name))
            ses.add_all([TimetableSlot(**r) for r in rows])
        _upsert_document(pdf.name, "timetable", None, len(rows))
        logger.info("Loaded %s: %d timetable rows", pdf.name, len(rows))
        total += len(rows)
    return total


# ----------------------------- orchestration -----------------------------
def run() -> dict[str, int]:
    configure_logging()
    init_db()

    # Phase A — embed (torch active).
    items, doc_meta = embed_all()

    # Write the torch-safe serving snapshot (no qdrant needed for this).
    from retrieval.snapshot import save_snapshot

    save_snapshot(items, get_settings().retrieval_snapshot)

    # Phase B — index. Default: isolated subprocess (torch-free) for stability.
    if os.environ.get("RAG_INPROC_INDEX") == "1":
        counts = index_all(items, doc_meta)
    else:
        counts = _index_in_subprocess(items, doc_meta)

    # Timetable → Postgres (no torch; safe in-process).
    tt = ingest_timetable()

    return {
        "course+manual_chunks": len(items),
        "qdrant_points": counts.get("_qdrant_total", 0),
        "timetable_rows": tt,
    }


def _index_in_subprocess(items, doc_meta) -> dict[str, int]:
    """Pickle precomputed vectors and run Phase B in a fresh, torch-free process."""
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        pickle.dump({"items": items, "doc_meta": doc_meta}, f)
        path = f.name
    logger.info("Launching torch-free indexing subprocess for %d items…", len(items))
    proc = subprocess.run(
        [sys.executable, "-m", "ingestion.index_worker", path],
        capture_output=True, text=True,
    )
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"Indexing subprocess failed (rc={proc.returncode})")
    # Last stdout line is the qdrant total.
    total = 0
    for line in proc.stdout.splitlines():
        if line.startswith("QDRANT_TOTAL="):
            total = int(line.split("=", 1)[1])
    os.unlink(path)
    return {"_qdrant_total": total}


def main() -> None:
    result = run()
    print("\n=== Ingestion complete ===")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
