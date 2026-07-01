"""Agent graph tests with mocked LLM / searcher / reranker.

Verifies: out-of-scope refusal, course routing → retrieval → cited generation,
the real sufficiency LOOP (low score → rewrite → retry → proceed), and the
timetable Postgres fast-path.
"""

from __future__ import annotations

import pytest

from agent.graph import build_agent
from agent.prompts import GENERATE_SYSTEM, REWRITE_SYSTEM, ROUTER_SYSTEM
from agent.state import AgentState
from db.models import TimetableSlot
from db.session import init_db, session_scope
from retrieval.hybrid_search import RetrievedChunk
from retrieval.reranker import RerankedChunk


class MockRouterLLM:
    """Routes by keyword; rewrites by appending; generates a canned answer."""

    def __init__(self, route_word: str):
        self.route_word = route_word
        self.calls: list[str] = []

    def generate(self, system: str, user: str, max_new_tokens=None) -> str:
        if system == ROUTER_SYSTEM:
            return self.route_word
        if system == REWRITE_SYSTEM:
            self.calls.append("rewrite")
            return user + " rewritten"
        if system == GENERATE_SYSTEM:
            return "Grounded answer [1]."
        return "?"


class MockSearcher:
    def __init__(self):
        self.queries: list[str] = []

    def search(self, query, top_k=20, doc_type=None, department=None):
        self.queries.append(query)
        return [
            RetrievedChunk(
                chunk_id="c1",
                text="CE212 Environment and Sustainability ... 3-0-0-0-9",
                payload={"doc_type": "course_catalog", "course_code": "CE212", "source_pdf": "CE.pdf"},
                score=0.5,
            )
        ]


class MockReranker:
    """Returns a low score on the first call, high on the second (drives the loop)."""

    def __init__(self, scores):
        self.scores = list(scores)
        self.i = 0

    def rerank(self, query, candidates, top_k=5):
        score = self.scores[min(self.i, len(self.scores) - 1)]
        self.i += 1
        return [
            RerankedChunk(chunk_id=c.chunk_id, text=c.text, payload=c.payload, rerank_score=score)
            for c in candidates
        ]


def test_out_of_scope_refusal():
    app = build_agent(MockRouterLLM("out_of_scope"), MockSearcher(), MockReranker([1.0]))
    out = app.invoke(AgentState(query="best pizza?"))
    assert out["route"] == "out_of_scope"
    assert "only help with" in out["answer"].lower()


def test_course_route_with_citations():
    app = build_agent(MockRouterLLM("course"), MockSearcher(), MockReranker([0.9]))
    out = app.invoke(AgentState(query="What is CE212?"))
    assert out["route"] == "course"
    assert out["answer"] == "Grounded answer [1]."
    assert out["citations"] and out["citations"][0]["course_code"] == "CE212"


def test_sufficiency_loop_fires():
    llm = MockRouterLLM("course")
    searcher = MockSearcher()
    # first rerank score below threshold (default 0.2) → loop once, then high.
    reranker = MockReranker([0.05, 0.9])
    app = build_agent(llm, searcher, reranker)
    out = app.invoke(AgentState(query="vague query about credits"))
    assert out["retry_count"] == 1  # exactly one retry happened
    assert "rewrite" in llm.calls  # query was rewritten
    assert len(searcher.queries) == 2  # retrieval ran twice (loop)
    assert out["answer"] == "Grounded answer [1]."


def test_timetable_fast_path(tmp_path, monkeypatch):
    # point DB at a temp sqlite and seed a timetable row
    monkeypatch.setenv("DB_URL", f"sqlite:///{tmp_path/'tt.db'}")
    import db.session as dbs
    dbs._engine = None
    dbs._SessionFactory = None
    from config import get_settings
    get_settings.cache_clear()
    init_db()
    with session_scope() as s:
        s.add(TimetableSlot(course_code="CS340", course_name="Theory of Computation",
                            slot="SLOT-2", schedule_kind="lecture", day="Monday",
                            start_time="10:00", end_time="11:00", instructor="Dr X",
                            source_pdf="tt.pdf"))

    app = build_agent(MockRouterLLM("timetable"), MockSearcher(), MockReranker([1.0]))
    out = app.invoke(AgentState(query="When does CS340 meet?"))
    assert out["route"] == "timetable_lookup"
    assert out["retrieved"], "timetable lookup should populate retrieved"
    # The Postgres fast-path built context from the seeded row, bypassing vectors.
    assert "CS340" in out["retrieved"][0]["text"]
    assert "Monday" in out["retrieved"][0]["text"]
    assert out["answer"]  # generate ran
