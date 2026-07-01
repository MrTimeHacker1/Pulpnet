"""Integration tests for hybrid retrieval against the LIVE Qdrant index.

Skipped automatically if the collection is empty (run ingestion first).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from config import get_settings

# Gate on the SNAPSHOT (not Qdrant) so collecting this file never imports
# qdrant-client into a pytest process that also runs torch (that combo segfaults).
_snap = Path(get_settings().retrieval_snapshot)
pytestmark = pytest.mark.skipif(
    not _snap.exists(), reason="No retrieval snapshot — run `python -m ingestion.run_ingestion`"
)


@pytest.fixture(scope="module")
def searcher():
    from retrieval.hybrid_search import HybridSearcher

    return HybridSearcher()


def test_corpus_loaded(searcher):
    assert len(searcher.ids) > 1000
    assert searcher.vectors.shape[0] == len(searcher.ids)


def test_course_query_returns_course_chunks(searcher):
    hits = searcher.search("machine learning courses", top_k=10, doc_type="course_catalog")
    assert hits
    assert all(h.payload.get("doc_type") == "course_catalog" for h in hits)


def test_department_filter(searcher):
    hits = searcher.search("structural engineering", top_k=10, department="CE")
    assert hits
    assert all(h.payload.get("department") == "CE" for h in hits)


def test_policy_query_returns_manual_chunks(searcher):
    hits = searcher.search("credit requirements for a minor", top_k=10, doc_type="academic_manual")
    assert hits
    assert all(h.payload.get("doc_type") == "academic_manual" for h in hits)
