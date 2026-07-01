"""Academic manual (UG/PG Manual) parser.

Manuals are free prose organised into numbered sections (4.5.2, 6.3.1, ...).
Steps:
  1. Extract text per page (pdfplumber).
  2. Strip repeating headers/footers: lines appearing on >=50% of pages and bare
     page numbers.
  3. Detect sections via SECTION_RE; tag each text block with (number, title).
     Text before the first section = "Preamble".
  4. Semantic-chunk each section body and prepend "[number title]" to each chunk's
     embedded text (retrieval context). Emit Chunks (doc_type="academic_manual").

Section detection is data-driven; verified against the real manuals (UG=117,
PG=55 headings) via `structural_report`.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pdfplumber

from ingestion.chunking.semantic_chunker import Embedder, semantic_chunk
from ingestion.schema import Chunk

logger = logging.getLogger(__name__)

SECTION_RE = re.compile(r"^\s*(\d+(?:\.\d+){0,3})\s+([A-Z][^\n]{2,80})$")
PAGE_NUM_RE = re.compile(r"^\s*\d{1,4}\s*$")
HEADER_FOOTER_MIN_FRACTION = 0.5


@dataclass
class Section:
    number: str
    title: str
    body: str
    page_start: int
    page_end: int


def _extract_pages(pdf_path: Path) -> list[list[str]]:
    """Return per-page lists of text lines."""
    pages: list[list[str]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages.append([ln.rstrip() for ln in text.split("\n")])
    return pages


def _strip_headers_footers(pages: list[list[str]]) -> list[list[str]]:
    """Drop lines that repeat on >=50% of pages, plus bare page numbers."""
    n_pages = len(pages)
    if n_pages == 0:
        return pages
    freq: Counter[str] = Counter()
    for lines in pages:
        # count each distinct stripped line once per page
        for ln in set(s.strip() for s in lines if s.strip()):
            freq[ln] += 1
    threshold = max(2, int(HEADER_FOOTER_MIN_FRACTION * n_pages))
    repeating = {ln for ln, c in freq.items() if c >= threshold}

    cleaned: list[list[str]] = []
    for lines in pages:
        kept = [
            ln
            for ln in lines
            if ln.strip()
            and ln.strip() not in repeating
            and not PAGE_NUM_RE.match(ln)
        ]
        cleaned.append(kept)
    return cleaned


def detect_sections(pdf_path: str | Path) -> list[Section]:
    """Parse a manual into ordered Sections (no embedding — cheap, testable)."""
    pdf_path = Path(pdf_path)
    pages = _strip_headers_footers(_extract_pages(pdf_path))

    sections: list[Section] = []
    cur_num, cur_title = "Preamble", "Preamble"
    cur_lines: list[str] = []
    cur_start = 0
    cur_last_page = 0

    def flush() -> None:
        body = " ".join(cur_lines).strip()
        if body or sections:  # keep non-empty; drop a leading empty preamble
            sections.append(
                Section(cur_num, cur_title, body, cur_start, cur_last_page)
            )

    for pi, lines in enumerate(pages):
        for ln in lines:
            m = SECTION_RE.match(ln)
            if m:
                if cur_lines or sections or cur_num != "Preamble":
                    flush()
                cur_num, cur_title = m.group(1), m.group(2).strip()
                cur_lines = []
                cur_start = pi
                cur_last_page = pi
            else:
                cur_lines.append(ln)
                cur_last_page = pi
    flush()

    return [s for s in sections if s.body]


def parse_to_chunks(
    pdf_path: str | Path,
    embed_fn: Embedder | None = None,
    **chunk_kwargs,
) -> list[Chunk]:
    """Parse a manual into semantically-chunked Chunks."""
    pdf_path = Path(pdf_path)
    if embed_fn is None:
        from ingestion.chunking.semantic_chunker import BGEEmbedder

        embed_fn = BGEEmbedder()

    sections = detect_sections(pdf_path)
    chunks: list[Chunk] = []
    for sec in sections:
        label = f"[{sec.number} {sec.title}]"
        for piece in semantic_chunk(sec.body, embed_fn, **chunk_kwargs):
            chunks.append(
                Chunk(
                    text=f"{label} {piece}".strip(),
                    doc_type="academic_manual",
                    source_pdf=pdf_path.name,
                    section=sec.number,
                    section_title=sec.title,
                    page_start=sec.page_start,
                    page_end=sec.page_end,
                )
            )
    return chunks


def structural_report(pdf_path: str | Path) -> dict:
    """Cheap structural summary for the dry-run acceptance check."""
    sections = detect_sections(pdf_path)
    real = [s for s in sections if s.number != "Preamble"]
    return {
        "n_sections": len(sections),
        "n_numbered": len(real),
        "first_headings": [(s.number, s.title) for s in real[:8]],
        "avg_body_chars": int(sum(len(s.body) for s in sections) / max(1, len(sections))),
    }
