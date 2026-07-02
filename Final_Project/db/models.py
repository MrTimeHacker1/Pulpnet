"""SQLAlchemy 2.0 models (DeclarativeBase / Mapped / mapped_column).

Backend-agnostic: runs on SQLite (dev default) or Postgres (set DB_URL). JSON
columns use the generic `JSON` type which maps to SQLite JSON1 / Postgres JSONB.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_pdf: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    doc_type: Mapped[str] = mapped_column(String(32), index=True)
    department: Mapped[str | None] = mapped_column(String(32), nullable=True)
    n_chunks: Mapped[int] = mapped_column(Integer, default=0)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class TimetableSlot(Base):
    """One scheduled meeting (exploded from the timetable). No room column exists
    in the source PDF; `group_name` is the closest analogue to a section."""

    __tablename__ = "timetable_slots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    course_code: Mapped[str] = mapped_column(String(16), index=True)
    branch: Mapped[str | None] = mapped_column(String(16), nullable=True)
    course_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    group_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    slot: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    credits_raw: Mapped[str | None] = mapped_column(String(32), nullable=True)
    course_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    instructor: Mapped[str | None] = mapped_column(Text, nullable=True)
    instructor_email: Mapped[str | None] = mapped_column(Text, nullable=True)
    schedule_kind: Mapped[str | None] = mapped_column(String(16), nullable=True)
    day: Mapped[str | None] = mapped_column(String(12), nullable=True, index=True)
    start_time: Mapped[str | None] = mapped_column(String(8), nullable=True)
    end_time: Mapped[str | None] = mapped_column(String(8), nullable=True)
    source_pdf: Mapped[str] = mapped_column(String(255))


class QueryLog(Base):
    __tablename__ = "query_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    query: Mapped[str] = mapped_column(Text)
    route: Mapped[str | None] = mapped_column(String(32), nullable=True)
    retrieved_chunk_ids: Mapped[list[Any] | None] = mapped_column(JSON, nullable=True)
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    guardrail_flags: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    cache_hit: Mapped[bool] = mapped_column(Boolean, default=False)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    scores: Mapped[list["EvalScore"]] = relationship(
        back_populates="query_log", cascade="all, delete-orphan"
    )


class EvalScore(Base):
    __tablename__ = "eval_scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    query_log_id: Mapped[int] = mapped_column(ForeignKey("query_logs.id"), index=True)
    faithfulness: Mapped[float | None] = mapped_column(Float, nullable=True)
    answer_relevancy: Mapped[float | None] = mapped_column(Float, nullable=True)
    context_relevancy: Mapped[float | None] = mapped_column(Float, nullable=True)
    context_precision: Mapped[float | None] = mapped_column(Float, nullable=True)
    on_topic: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    query_log: Mapped["QueryLog"] = relationship(back_populates="scores")
