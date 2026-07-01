"""Uniform ingestion record produced by every parser.

All PDF parsers emit a list of `Chunk`. A `Chunk` carries the text to embed plus
structured metadata. `chunk_id` is deterministic (sha256 of identifying fields)
so re-ingestion is idempotent — the same source content always maps to the same
Qdrant point id.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

DocType = Literal["course_catalog", "academic_manual", "timetable"]


@dataclass
class Chunk:
    """A single embeddable unit with provenance + structured metadata."""

    text: str
    doc_type: DocType
    source_pdf: str

    # Optional structured metadata (populated where relevant per doc_type).
    department: str | None = None
    course_code: str | None = None
    course_title: str | None = None
    credits_raw: str | None = None
    section: str | None = None
    section_title: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        """Deterministic id derived from sha256(doc_type|source_pdf|course_code|section|text).

        Formatted as a canonical UUID string so it is a valid Qdrant point id
        (Qdrant requires uint64 or UUID) while remaining stable across re-ingestion.
        """
        key = "|".join(
            [
                self.doc_type,
                self.source_pdf,
                self.course_code or "",
                self.section or "",
                self.text,
            ]
        )
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return str(uuid.UUID(hex=digest[:32]))

    def to_payload(self) -> dict[str, Any]:
        """Filterable payload for Qdrant (and a generic record for logging)."""
        return {
            "chunk_id": self.chunk_id,
            "doc_type": self.doc_type,
            "source_pdf": self.source_pdf,
            "department": self.department,
            "course_code": self.course_code,
            "course_title": self.course_title,
            "credits_raw": self.credits_raw,
            "section": self.section,
            "section_title": self.section_title,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "text": self.text,
            "extra": self.extra,
        }
