"""Tests for the academic manual parser (real PDFs, mock embedder)."""

from __future__ import annotations

import hashlib

import numpy as np

from config import get_settings
from ingestion.parsers.academic_doc_parser import (
    _strip_headers_footers,
    detect_sections,
    parse_to_chunks,
)

DOCS = get_settings().academic_docs_dir


def _det_embedder(texts: list[str]) -> np.ndarray:
    """Deterministic hash-based embedder (no model load)."""
    out = []
    for t in texts:
        h = hashlib.sha256(t.encode()).digest()
        out.append(np.frombuffer(h[:32], dtype=np.uint8).astype(np.float64))
    return np.asarray(out)


def test_strip_headers_footers_removes_repeats_and_page_numbers():
    pages = [
        ["IIT KANPUR MANUAL", "real content one", "12"],
        ["IIT KANPUR MANUAL", "real content two", "13"],
        ["IIT KANPUR MANUAL", "real content three", "14"],
    ]
    cleaned = _strip_headers_footers(pages)
    flat = [ln for pg in cleaned for ln in pg]
    assert "IIT KANPUR MANUAL" not in flat  # repeats on 100% of pages
    assert "12" not in flat and "13" not in flat  # bare page numbers
    assert any("real content" in ln for ln in flat)


def test_ug_manual_sections_detected():
    secs = detect_sections(DOCS / "UG_Manual.pdf")
    numbered = [s for s in secs if s.number != "Preamble"]
    assert len(numbered) >= 50
    titles = {(s.number, s.title) for s in numbered}
    assert ("1.1", "Introduction") in titles


def test_pg_manual_sections_detected():
    secs = detect_sections(DOCS / "PG-Manual.pdf")
    numbered = [s for s in secs if s.number != "Preamble"]
    assert len(numbered) >= 20


def test_parse_to_chunks_prepends_section_label():
    chunks = parse_to_chunks(
        DOCS / "UG_Manual.pdf", embed_fn=_det_embedder, min_chars=100
    )
    assert len(chunks) > 50
    for c in chunks[:20]:
        assert c.doc_type == "academic_manual"
        assert c.section is not None
        # label like "[1.1 Introduction] ..."
        assert c.text.startswith(f"[{c.section} ")
