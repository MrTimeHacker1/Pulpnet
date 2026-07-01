"""Pydantic v2 request/response models for the API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    department: str | None = Field(default=None, max_length=16)
    top_k: int | None = Field(default=None, ge=1, le=50)


class Citation(BaseModel):
    n: int
    doc_type: str | None = None
    source_pdf: str | None = None
    course_code: str | None = None
    section: str | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    route: str | None = None
    cache_hit: bool = False
    latency_ms: int = 0
    guardrail_flags: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    qdrant: bool
    postgres: bool
    redis: bool
    qdrant_points: int = 0


class JobAck(BaseModel):
    status: str
    detail: str


class CacheStatsResponse(BaseModel):
    hits: int
    misses: int
    semantic_hits: int
    exact_hits: int
    total: int
    hit_rate: float
