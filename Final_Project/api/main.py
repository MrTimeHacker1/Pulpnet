"""FastAPI application for the IITK Agentic RAG platform.

Uses the lifespan context manager (NOT @app.on_event) to load models/clients once
at startup. Endpoints:
  GET  /health     - liveness + Qdrant/Postgres/Redis connectivity
  POST /query      - guardrails → cache → agent → output guardrails → log
  POST /ingest     - background ingestion
  POST /eval/run   - background RAGAS evaluation
  GET  /cache/stats- real cache hit/miss numbers
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException
from sqlalchemy import text as sql_text

from api.schemas import (
    CacheStatsResponse,
    HealthResponse,
    JobAck,
    QueryRequest,
    QueryResponse,
)
from config import configure_logging, get_settings
from db.session import init_db, session_scope

logger = logging.getLogger(__name__)


class AppState:
    agent = None
    cache = None
    embedder = None
    searcher = None


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    s = get_settings()
    logger.info("Starting IITK RAG API…")
    init_db()

    # Shared embedder for searcher + cache (avoid loading bge twice).
    from ingestion.chunking.semantic_chunker import BGEEmbedder

    state.embedder = BGEEmbedder()

    from cache.redis_cache import SemanticCache

    state.cache = SemanticCache(embedder=state.embedder)

    # Build the agent (searcher scrolls Qdrant once; llm + reranker lazy/loaded).
    try:
        from agent.graph import build_agent
        from llm.gemma import get_llm
        from retrieval.hybrid_search import HybridSearcher
        from retrieval.reranker import Reranker

        searcher = HybridSearcher(embedder=state.embedder)
        state.searcher = searcher
        reranker = Reranker()
        llm = get_llm()
        state.agent = build_agent(llm=llm, searcher=searcher, reranker=reranker)
        logger.info("Agent ready (%d chunks indexed).", len(searcher.ids))
    except Exception as e:  # API still serves /health if models/index unavailable
        logger.exception("Agent initialization failed: %s", e)
        state.agent = None

    yield

    if state.cache is not None:
        await state.cache.close()
    logger.info("Shutdown complete.")


app = FastAPI(title="IITK Agentic RAG", version="1.0.0", lifespan=lifespan)


# ------------------------------- /health -------------------------------
@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    s = get_settings()
    # Postgres/SQLite
    pg_ok = False
    try:
        with session_scope() as ses:
            ses.execute(sql_text("SELECT 1"))
        pg_ok = True
    except Exception as e:
        logger.warning("DB health failed: %s", e)

    # Qdrant — report from the in-memory searcher; do NOT reopen the Qdrant
    # local client during serving (an open client + torch inference segfaults).
    qdrant_ok = state.searcher is not None and len(state.searcher.ids) > 0
    points = len(state.searcher.ids) if state.searcher is not None else 0

    # Redis (best-effort; in-process fallback means cache still works either way)
    redis_ok = False
    try:
        if state.cache is not None:
            await state.cache._ensure_redis()
            redis_ok = state.cache._redis is not None
    except Exception:
        redis_ok = False

    status = "ok" if (pg_ok and qdrant_ok and state.agent is not None) else "degraded"
    return HealthResponse(
        status=status, qdrant=qdrant_ok, postgres=pg_ok, redis=redis_ok, qdrant_points=points
    )


# ------------------------------- /query -------------------------------
@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    from agent.state import AgentState
    from guardrails.input_guardrails import check_input
    from guardrails.output_guardrails import check_output

    t0 = time.perf_counter()

    # 1) Input guardrails (hard gate).
    gin = check_input(req.query)
    if not gin.allowed:
        latency = int((time.perf_counter() - t0) * 1000)
        _log_query(req.query, "blocked", [], gin.reason, gin.flags, False, latency)
        return QueryResponse(
            answer=gin.reason, route="blocked", cache_hit=False,
            latency_ms=latency, guardrail_flags=gin.flags,
        )

    # 2) Cache.
    if state.cache is not None:
        cached = await state.cache.get(req.query)
        if cached is not None:
            latency = int((time.perf_counter() - t0) * 1000)
            return QueryResponse(
                answer=cached["answer"], citations=cached.get("citations", []),
                route=cached.get("route"), cache_hit=True, latency_ms=latency,
                guardrail_flags=cached.get("guardrail_flags", {}),
            )

    if state.agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized (run ingestion / check models).")

    # 3) Agent graph.
    try:
        result = state.agent.invoke(
            AgentState(query=req.query, department=req.department)
        )
    except Exception as e:
        logger.exception("Agent invocation failed")
        raise HTTPException(status_code=500, detail="Internal error while answering.") from e

    answer = result.get("answer") or ""
    citations = result.get("citations", [])
    route = result.get("route")
    retrieved = result.get("retrieved", [])
    flags = dict(result.get("guardrail_flags", {}))

    # 4) Output guardrails (faithfulness/grounding) — skip for refusals/out-of-scope.
    if route not in ("out_of_scope", "blocked") and retrieved:
        contexts = [c["text"] for c in retrieved]
        from llm.gemma import get_llm

        answer, gflags = check_output(answer, contexts, llm=get_llm())
        flags.update(gflags)

    latency = int((time.perf_counter() - t0) * 1000)

    # 5) Cache write.
    record = {"answer": answer, "citations": citations, "route": route, "guardrail_flags": flags}
    if state.cache is not None and route not in ("blocked",):
        await state.cache.set(req.query, record)

    # 6) Log.
    chunk_ids = [c.get("payload", {}).get("chunk_id") for c in retrieved]
    _log_query(req.query, route, chunk_ids, answer, flags, False, latency)

    return QueryResponse(
        answer=answer, citations=citations, route=route,
        cache_hit=False, latency_ms=latency, guardrail_flags=flags,
    )


def _log_query(query, route, chunk_ids, answer, flags, cache_hit, latency_ms) -> None:
    try:
        from db.models import QueryLog

        with session_scope() as ses:
            ses.add(QueryLog(
                query=query, route=route, retrieved_chunk_ids=chunk_ids, answer=answer,
                guardrail_flags=flags, cache_hit=cache_hit, latency_ms=latency_ms,
            ))
    except Exception as e:
        logger.warning("Failed to write query log: %s", e)


# ------------------------------- /ingest -------------------------------
def _run_ingestion_job() -> None:
    try:
        from ingestion.run_ingestion import run

        run()
    except Exception:
        logger.exception("Background ingestion failed")


@app.post("/ingest", response_model=JobAck)
async def ingest(background: BackgroundTasks) -> JobAck:
    background.add_task(_run_ingestion_job)
    return JobAck(status="accepted", detail="Ingestion started in the background.")


# ------------------------------- /eval/run -------------------------------
def _run_eval_job() -> None:
    try:
        from eval.ragas_eval import run_eval

        run_eval()
    except Exception:
        logger.exception("Background eval failed")


@app.post("/eval/run", response_model=JobAck)
async def eval_run(background: BackgroundTasks) -> JobAck:
    background.add_task(_run_eval_job)
    return JobAck(status="accepted", detail="RAGAS evaluation started in the background.")


# ------------------------------- /cache/stats -------------------------------
@app.get("/cache/stats", response_model=CacheStatsResponse)
async def cache_stats() -> CacheStatsResponse:
    if state.cache is None:
        raise HTTPException(status_code=503, detail="Cache not initialized.")
    return CacheStatsResponse(**state.cache.stats.as_dict())
