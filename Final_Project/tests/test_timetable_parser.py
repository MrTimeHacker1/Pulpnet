"""Tests for the timetable parser (real PDF)."""

from __future__ import annotations

from config import get_settings
from ingestion.parsers.timetable_parser import (
    _expand_days,
    parse_schedule,
    parse_to_rows,
)

PDF = get_settings().timetable_dir / "Course_Schedule_2026-27-1.pdf"


def test_expand_days_thursday_before_tuesday():
    assert _expand_days("TTh") == ["Tuesday", "Thursday"]
    assert _expand_days("MWF") == ["Monday", "Wednesday", "Friday"]
    assert _expand_days("ThF") == ["Thursday", "Friday"]


def test_parse_schedule_multiple_meetings():
    out = parse_schedule("MF 08:00-09:00 ,T 09:00-10:00")
    assert ("Monday", "08:00", "09:00") in out
    assert ("Friday", "08:00", "09:00") in out
    assert ("Tuesday", "09:00", "10:00") in out


def test_parse_to_rows_real_pdf():
    rows = parse_to_rows(PDF)
    assert len(rows) > 1000
    codes = {r["course_code"] for r in rows}
    assert len(codes) > 400
    # Normalized course codes
    assert all(r["course_code"] == r["course_code"].upper() for r in rows)
    assert "AE209" in codes
    # Schedule kinds constrained
    kinds = {r["schedule_kind"] for r in rows}
    assert kinds <= {"lecture", "tutorial", "practical", None}
    # Thursday actually appears (Th handling)
    assert any(r["day"] == "Thursday" for r in rows)
    # Every row has the expected keys
    expected = {
        "course_code", "branch", "course_name", "group_name", "slot",
        "credits_raw", "course_type", "instructor", "instructor_email",
        "source_pdf", "schedule_kind", "day", "start_time", "end_time",
    }
    assert set(rows[0]) == expected
