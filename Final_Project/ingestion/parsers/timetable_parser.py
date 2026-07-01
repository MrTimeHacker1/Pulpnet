"""Timetable parser → structured rows for Postgres (NOT embedded).

The schedule PDF (Course_Schedule_2026-27-1.pdf) is a clean, regular table:
`extract_tables()` returns exactly 1 table/page, 13 columns, header repeated on
every page. Real columns differ from a generic timetable schema — there is NO
explicit section and NO room; day/time live inside three schedule columns
(Lecture/Tutorial/Practical) as "<days> <HH:MM-HH:MM>" strings, multiple meetings
comma-separated, and `Th` (Thursday) must be parsed before single-letter `T`.

We emit ONE ROW PER SCHEDULED MEETING (exploding the schedule strings); a course
with no parseable meeting still yields one row (nulls) so exact code lookups hit.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import pdfplumber

logger = logging.getLogger(__name__)

CODE_RE = re.compile(r"^[A-Z]{2,4}\d{3}[A-Z]?$")
# One meeting: a run of day letters, then HH:MM-HH:MM.
MEETING_RE = re.compile(r"([A-Za-z]{1,8})\s*(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})")

# Order matters: multi-char tokens (Th) must be tried before single-char (T).
_DAY_TOKENS: list[tuple[str, str]] = [
    ("Th", "Thursday"),
    ("M", "Monday"),
    ("T", "Tuesday"),
    ("W", "Wednesday"),
    ("F", "Friday"),
    ("Su", "Sunday"),
    ("S", "Saturday"),
]

# Map a header cell (lowercased, newlines→space) → canonical field name.
_HEADER_KEYS: list[tuple[str, callable]] = [
    ("course_code", lambda h: "course" in h and "code" in h),
    ("course_name", lambda h: "course" in h and "name" in h),
    ("branch", lambda h: h == "branch"),
    ("group_name", lambda h: "group" in h),
    ("slot", lambda h: "slot" in h),
    ("credits_raw", lambda h: "unit" in h or "credit" in h),
    ("course_type", lambda h: "type" in h),
    ("instructor_email", lambda h: "email" in h),
    ("instructor", lambda h: h == "instructor"),
    ("lecture", lambda h: "lecture" in h),
    ("tutorial", lambda h: "tutorial" in h),
    ("practical", lambda h: "practical" in h),
]


def _norm(cell: Any) -> str:
    if cell is None:
        return ""
    return re.sub(r"\s+", " ", str(cell).replace("\n", " ")).strip()


def _expand_days(token: str) -> list[str]:
    """Expand a day-letter run like 'MWF' or 'TTh' into weekday names."""
    days: list[str] = []
    i = 0
    s = token
    while i < len(s):
        for tok, name in _DAY_TOKENS:
            if s.startswith(tok, i):
                days.append(name)
                i += len(tok)
                break
        else:
            i += 1  # skip an unrecognised char
    return days


def parse_schedule(cell: str) -> list[tuple[str, str, str]]:
    """Parse a schedule cell into (day, start_time, end_time) tuples."""
    cell = _norm(cell)
    if not cell:
        return []
    meetings: list[tuple[str, str, str]] = []
    for m in MEETING_RE.finditer(cell):
        day_tok, start, end = m.group(1), m.group(2), m.group(3)
        for day in _expand_days(day_tok):
            meetings.append((day, start, end))
    return meetings


def _build_colmap(header: list[Any]) -> dict[str, int]:
    colmap: dict[str, int] = {}
    norm = [_norm(h).lower() for h in header]
    for field, pred in _HEADER_KEYS:
        if field in colmap:
            continue
        for idx, h in enumerate(norm):
            if idx in colmap.values():
                continue
            if pred(h):
                colmap[field] = idx
                break
    return colmap


def _is_header_row(row: list[Any], colmap: dict[str, int]) -> bool:
    code = _norm(row[colmap["course_code"]]) if "course_code" in colmap else ""
    return code.lower().replace(" ", "") in {"coursecode", "code"}


def parse_to_rows(pdf_path: str | Path) -> list[dict[str, Any]]:
    """Parse the timetable PDF into per-meeting row dicts."""
    pdf_path = Path(pdf_path)
    rows: list[dict[str, Any]] = []
    colmap: dict[str, int] = {}

    with pdfplumber.open(pdf_path) as pdf:
        for pi, page in enumerate(pdf.pages):
            for table in page.extract_tables() or []:
                if not table:
                    continue
                # First row of each page's table is the (repeated) header.
                if not colmap:
                    colmap = _build_colmap(table[0])
                    if "course_code" not in colmap:
                        logger.warning("Timetable: could not map columns from header %r", table[0])
                        continue
                data_rows = table[1:]
                for row in data_rows:
                    if len(row) < max(colmap.values()) + 1:
                        continue
                    if _is_header_row(row, colmap):
                        continue
                    rows.extend(_emit_rows(row, colmap, pdf_path.name, pi))
    logger.info("Timetable: parsed %d meeting-rows from %s", len(rows), pdf_path.name)
    return rows


def _emit_rows(row: list[Any], colmap: dict[str, int], source: str, page: int) -> list[dict]:
    def g(field: str) -> str:
        return _norm(row[colmap[field]]) if field in colmap else ""

    raw_code = g("course_code")
    code = raw_code.upper().replace(" ", "")
    if not code:
        return []
    if not CODE_RE.match(code):
        logger.debug("Timetable: low-confidence course_code %r (page %d)", raw_code, page)

    base = {
        "course_code": code,
        "branch": g("branch") or None,
        "course_name": g("course_name") or None,
        "group_name": g("group_name") or None,
        "slot": g("slot") or None,
        "credits_raw": g("credits_raw") or None,
        "course_type": g("course_type") or None,
        "instructor": g("instructor") or None,
        "instructor_email": g("instructor_email") or None,
        "source_pdf": source,
    }

    out: list[dict] = []
    for kind in ("lecture", "tutorial", "practical"):
        cell = g(kind)
        for day, start, end in parse_schedule(cell):
            out.append(
                {**base, "schedule_kind": kind, "day": day, "start_time": start, "end_time": end}
            )
    if not out:  # course with no parseable meeting still gets a row (nulls)
        out.append(
            {**base, "schedule_kind": None, "day": None, "start_time": None, "end_time": None}
        )
    return out
