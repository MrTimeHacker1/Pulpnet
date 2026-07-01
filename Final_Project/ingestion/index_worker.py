"""Phase-B indexing worker — runs in a SEPARATE, torch-free process.

Reads a pickle of precomputed {items: [(Chunk, vector)], doc_meta} produced by
run_ingestion's embed phase and upserts to Qdrant + writes `documents` rows.
Importing this module must NOT pull in torch/sentence-transformers (that's the
whole point of process isolation), so it only touches qdrant-client + the DB.

Usage:  python -m ingestion.index_worker <pickle_path>
"""

from __future__ import annotations

import pickle
import sys

from config import configure_logging
from db.session import init_db
from ingestion.run_ingestion import index_all


def main(path: str) -> None:
    configure_logging()
    init_db()
    with open(path, "rb") as f:
        data = pickle.load(f)
    counts = index_all(data["items"], data["doc_meta"])
    print(f"QDRANT_TOTAL={counts.get('_qdrant_total', 0)}")


if __name__ == "__main__":
    main(sys.argv[1])
