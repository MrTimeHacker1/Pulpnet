"""Tests for the course catalog parser.

These run against the real CoS PDFs. Acceptance bar:
  - CE-CoS25 is the spec-validated golden file: 150 records, 100% on every field.
  - All files: 100% of records carry code + content, ~>=95% carry a title.
  - Credits are asserted "where present in source" — many grad-course pages
    (esp. CSE) genuinely omit the L-T-P-D-C string, so we only require that the
    parser extract a VALID L-T-P-D-C wherever one exists (CE proves the path).
"""

from __future__ import annotations

import pytest

from config import get_settings
from ingestion.parsers.course_catalog_parser import (
    CREDIT_FULL_RE,
    completeness_report,
    parse_to_chunks,
)

COURSES_DIR = get_settings().courses_dir
ALL_PDFS = sorted(COURSES_DIR.glob("*.pdf"))


def test_all_five_pdfs_present():
    assert len(ALL_PDFS) == 5, f"expected 5 CoS PDFs, found {[p.name for p in ALL_PDFS]}"


def test_ce_golden_full_completeness():
    chunks, dept = parse_to_chunks(COURSES_DIR / "CE-CoS25.pdf")
    assert dept == "CE"
    rep = completeness_report(chunks)
    assert rep["total"] == 150
    # CE is the validated golden file: 100% on every field.
    assert rep["with_code"] == 150
    assert rep["with_title"] == 150
    assert rep["with_credits"] == 150
    assert rep["with_content"] == 150


@pytest.mark.parametrize("pdf", ALL_PDFS, ids=[p.name for p in ALL_PDFS])
def test_structural_completeness(pdf):
    chunks, dept = parse_to_chunks(pdf)
    rep = completeness_report(chunks)
    assert rep["total"] > 20, f"{pdf.name}: implausibly few records ({rep['total']})"
    assert dept is not None
    # Code + content must be 100% (these are structural, always recoverable).
    assert rep["with_code"] == rep["total"]
    assert rep["with_content"] == rep["total"]
    # Titles: allow a small layout-edge-case tail.
    assert rep["with_title"] / rep["total"] >= 0.95


@pytest.mark.parametrize("pdf", ALL_PDFS, ids=[p.name for p in ALL_PDFS])
def test_extracted_credits_are_valid(pdf):
    """Every credit string the parser DID extract must be a valid L-T-P-(D-)C."""
    chunks, _ = parse_to_chunks(pdf)
    for c in chunks:
        if c.credits_raw and c.extra:
            assert CREDIT_FULL_RE.match(c.credits_raw), f"{c.course_code}: {c.credits_raw!r}"
            assert set(c.extra) == {"L", "T", "P", "D", "C"}


def test_chunk_ids_unique_and_stable():
    chunks, _ = parse_to_chunks(COURSES_DIR / "CE-CoS25.pdf")
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "chunk ids must be unique within a file"
    # stable across re-parse
    again, _ = parse_to_chunks(COURSES_DIR / "CE-CoS25.pdf")
    assert [c.chunk_id for c in again] == ids
