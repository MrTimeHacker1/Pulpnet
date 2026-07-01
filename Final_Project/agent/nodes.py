"""LangGraph nodes: router, retrieve_rerank, sufficiency, generate.

Exactly four nodes and one real conditional loop (sufficiency → retrieve_rerank).
The router doubles as the timetable fast-path: a timetable question with a course
code is answered by a direct Postgres lookup, bypassing vector retrieval.

Heavy dependencies (LLM, searcher, reranker) are injected so the graph is testable
with mocks.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from sqlalchemy import select

from agent.prompts import (
    GENERATE_SYSTEM,
    REFUSAL_OUT_OF_SCOPE,
    REWRITE_SYSTEM,
    ROUTER_SYSTEM,
    build_generate_user,
)
from agent.state import AgentState
from config import get_settings
from db.models import TimetableSlot
from db.session import session_scope

logger = logging.getLogger(__name__)

COURSE_CODE_RE = re.compile(r"\b([A-Z]{2,4}\d{3}[A-Z]?)\b")
VALID_ROUTES = {"course", "policy", "timetable", "out_of_scope"}
ROUTE_TO_DOCTYPE = {"course": "course_catalog", "policy": "academic_manual"}


def extract_course_code(text: str) -> str | None:
    # Uppercase but KEEP spacing so the \b word boundaries still match a code
    # embedded in prose (e.g. "When does CS340 meet?").
    m = COURSE_CODE_RE.search(text.upper())
    return m.group(1) if m else None


class AgentNodes:
    """Bundles node implementations over injected dependencies."""

    def __init__(self, llm, searcher, reranker):
        self.llm = llm
        self.searcher = searcher
        self.reranker = reranker
        self.settings = get_settings()

    # ---------------- router ----------------
    def router(self, state: AgentState) -> dict[str, Any]:
        label = self.llm.generate(ROUTER_SYSTEM, state.query, max_new_tokens=8).strip().lower()
        route = next((r for r in VALID_ROUTES if r in label), "course")

        if route == "out_of_scope":
            return {
                "route": "out_of_scope",
                "answer": REFUSAL_OUT_OF_SCOPE,
                "guardrail_flags": {**state.guardrail_flags, "out_of_scope": True},
            }

        if route == "timetable":
            code = extract_course_code(state.query)
            if code:
                rows = self._timetable_lookup(code)
                return {
                    "route": "timetable_lookup",
                    "retrieved": rows,
                    "top_score": 1.0 if rows else 0.0,
                }
            # timetable question without a code → fall back to vector retrieval
        return {"route": route}

    def _timetable_lookup(self, code: str) -> list[dict[str, Any]]:
        with session_scope() as ses:
            rows = ses.execute(
                select(TimetableSlot).where(TimetableSlot.course_code == code)
            ).scalars().all()
            blocks: list[dict[str, Any]] = []
            for r in rows:
                when = (
                    f"{r.day} {r.start_time}-{r.end_time}"
                    if r.day else "schedule not listed"
                )
                text = (
                    f"{r.course_code} ({r.course_name}) — {r.schedule_kind or 'class'}: "
                    f"{when}; slot {r.slot}; instructor {r.instructor}"
                )
                blocks.append(
                    {
                        "text": text,
                        "payload": {
                            "doc_type": "timetable",
                            "course_code": r.course_code,
                            "source_pdf": r.source_pdf,
                        },
                        "score": 1.0,
                    }
                )
        return blocks

    # ---------------- retrieve_rerank ----------------
    def retrieve_rerank(self, state: AgentState) -> dict[str, Any]:
        doc_type = ROUTE_TO_DOCTYPE.get(state.route)
        hits = self.searcher.search(
            state.effective_query,
            top_k=self.settings.retrieve_top_k,
            doc_type=doc_type,
            department=state.department,
        )
        reranked = self.reranker.rerank(state.effective_query, hits, top_k=self.settings.rerank_top_k)
        retrieved = [
            {"text": r.text, "payload": r.payload, "score": r.rerank_score} for r in reranked
        ]
        top = reranked[0].rerank_score if reranked else 0.0
        return {"retrieved": retrieved, "top_score": float(top)}

    # ---------------- sufficiency ----------------
    def sufficiency(self, state: AgentState) -> dict[str, Any]:
        """Decide whether to loop. Rewrites the query when looping back."""
        score = state.top_score if state.top_score is not None else 0.0
        if score < self.settings.sufficiency_min_score and state.retry_count < self.settings.max_retries:
            rewritten = self.llm.generate(
                REWRITE_SYSTEM, state.effective_query, max_new_tokens=48
            ).strip()
            return {
                "rewritten_query": rewritten or state.effective_query,
                "retry_count": state.retry_count + 1,
                "needs_retry": True,
            }
        return {"needs_retry": False}

    # ---------------- generate ----------------
    def generate(self, state: AgentState) -> dict[str, Any]:
        if state.route == "out_of_scope":
            return {"answer": state.answer or REFUSAL_OUT_OF_SCOPE}
        if not state.retrieved:
            return {
                "answer": "I couldn't find relevant information in the official documents "
                "to answer that.",
                "citations": [],
            }
        blocks = [c["text"] for c in state.retrieved]
        user = build_generate_user(state.effective_query, blocks)
        answer = self.llm.generate(GENERATE_SYSTEM, user)
        citations = [
            {
                "n": i + 1,
                "doc_type": c["payload"].get("doc_type"),
                "source_pdf": c["payload"].get("source_pdf"),
                "course_code": c["payload"].get("course_code"),
                "section": c["payload"].get("section"),
            }
            for i, c in enumerate(state.retrieved)
        ]
        return {"answer": answer.strip(), "citations": citations}
