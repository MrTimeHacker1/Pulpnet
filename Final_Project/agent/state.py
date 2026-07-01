"""Agent state (Pydantic v2) shared across LangGraph nodes."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AgentState(BaseModel):
    query: str
    department: str | None = None

    route: str | None = None
    rewritten_query: str | None = None

    # Reranked context chunks as plain dicts (text + payload + score).
    retrieved: list[dict[str, Any]] = Field(default_factory=list)
    top_score: float | None = None
    retry_count: int = 0

    needs_retry: bool = False

    answer: str | None = None
    citations: list[dict[str, Any]] = Field(default_factory=list)
    guardrail_flags: dict[str, Any] = Field(default_factory=dict)

    @property
    def effective_query(self) -> str:
        return self.rewritten_query or self.query
