"""LangGraph wiring: 4 nodes, one real conditional loop.

  START → router
  router ─┬─ out_of_scope ───────────────→ generate → END
          ├─ timetable_lookup ───────────→ generate → END
          └─ course/policy/timetable ───→ retrieve_rerank
  retrieve_rerank → sufficiency
  sufficiency ─┬─ (low score, retries left) → retrieve_rerank   [the loop]
               └─ otherwise ───────────────→ generate → END
"""

from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from agent.nodes import AgentNodes
from agent.state import AgentState
from config import get_settings

logger = logging.getLogger(__name__)


def _route_after_router(state: AgentState) -> str:
    if state.route in ("out_of_scope", "timetable_lookup"):
        return "generate"
    return "retrieve_rerank"


def _route_after_sufficiency(state: AgentState) -> str:
    # The sufficiency node sets `needs_retry` exactly when it rewrote the query
    # and wants another retrieval pass; otherwise we proceed to generation.
    return "retrieve_rerank" if state.needs_retry else "generate"


def build_agent(llm=None, searcher=None, reranker=None):
    """Build and compile the agent graph. Heavy deps are lazily constructed."""
    if llm is None:
        from llm.gemma import get_llm

        llm = get_llm()
    if searcher is None:
        from retrieval.hybrid_search import HybridSearcher

        searcher = HybridSearcher()
    if reranker is None:
        from retrieval.reranker import Reranker

        reranker = Reranker()

    nodes = AgentNodes(llm=llm, searcher=searcher, reranker=reranker)

    g = StateGraph(AgentState)
    g.add_node("router", nodes.router)
    g.add_node("retrieve_rerank", nodes.retrieve_rerank)
    g.add_node("sufficiency", nodes.sufficiency)
    g.add_node("generate", nodes.generate)

    g.add_edge(START, "router")
    g.add_conditional_edges(
        "router",
        _route_after_router,
        {"retrieve_rerank": "retrieve_rerank", "generate": "generate"},
    )
    g.add_edge("retrieve_rerank", "sufficiency")
    g.add_conditional_edges(
        "sufficiency",
        _route_after_sufficiency,
        {"retrieve_rerank": "retrieve_rerank", "generate": "generate"},
    )
    g.add_edge("generate", END)
    return g.compile()
