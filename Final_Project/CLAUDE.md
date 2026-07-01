# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is
An Agentic RAG backend answering IIT Kanpur student questions about courses, academic
regulations, and class timetables, grounded in official PDFs (in `data/`). Backend + FastAPI
only; no frontend.

## Commands
- Install: `pip install -r requirements.txt` (then `pip check`)
- Ingest (parse PDFs → embed → Qdrant + DB + snapshot): `python -m ingestion.run_ingestion`
- Serve API: `uvicorn api.main:app --host 0.0.0.0 --port 8000`
  - Fast/offline (skip the heavy Gemma load): prefix any command with `GENERATOR_MODEL=mock`
- Tests: `python -m pytest -q` · single test: `python -m pytest tests/test_agent.py::test_sufficiency_loop_fires`
- Manual check scripts (not part of pytest): `python -m tests.check_api`,
  `tests.check_course_parser`, `tests.check_retrieval`, `tests.check_gemma_live`
- Eval: `python -m eval.ragas_eval --limit 5`
- Docker (full stack: app + ingest + eval + qdrant/postgres/redis): `docker compose up --build`
  · RAGAS: `docker compose --profile eval run --rm eval`

## Request pipeline (one LangGraph, 4 nodes, one real loop)
`POST /query` (`api/main.py`) → input guardrails (`guardrails/input_guardrails.py`, HARD gate:
PII/injection/off-topic) → semantic cache (`cache/redis_cache.py`) → agent graph
(`agent/graph.py`) → output guardrail grounding check (`guardrails/output_guardrails.py`) →
cache write + `query_logs` row.

Agent nodes (`agent/nodes.py`): **router** (classify course|policy|timetable|out_of_scope;
timetable+course-code → direct Postgres lookup, bypassing vectors) → **retrieve_rerank** (hybrid
BM25+dense+RRF then bge cross-encoder) → **sufficiency** (if top rerank score <
`sufficiency_min_score` and retries left → rewrite query + loop back; the `needs_retry` state flag
drives the conditional edge) → **generate** (Gemma, grounded + cited).

## Data → storage mapping
- `data/courses/` → `course_catalog_parser` → Qdrant. Word-COORDINATE parser (not line text);
  per-file column boundaries in `PER_FILE_BOUNDS`; credits regex allows optional D field
  (`3-0-0-9` and `3-0-0-0-9`). Course records are NOT semantic-chunked.
- `data/academic_documents/` → `academic_doc_parser` → semantic chunk → Qdrant.
- `data/timetable/` → `timetable_parser` → **Postgres/SQLite** (`timetable_slots`), NOT embedded.
  One row per meeting; day strings expand with `Th` before `T`.

## CRITICAL invariants (do not break)
- **Never import `qdrant-client` (directly or transitively) into a process that also runs torch
  inference.** On the dev host this segfaults. Enforced by architecture:
  - Ingestion (`ingestion/run_ingestion.py`) embeds first (torch), writes
    `retrieval_snapshot.pkl` (`retrieval/snapshot.py`), then upserts to Qdrant in an isolated
    torch-free subprocess (`ingestion/index_worker.py`).
  - Serving `HybridSearcher` (`retrieval/hybrid_search.py`) loads the snapshot and does dense
    (numpy) + BM25 + RRF fully in-process — it must keep qdrant imports lazy/absent. `/health`
    reports index size from the in-memory searcher, not by reopening Qdrant.
- **Gemma is used text-only via `AutoTokenizer`** (`llm/gemma.py`), not `AutoProcessor` — the
  latter pulls an image processor needing `torchvision` (intentionally absent).
- `config.py` sets thread/OpenMP env vars **before** torch imports; keep that block first.
- `eval/ragas_eval.py` shims `langchain_community.chat_models.vertexai` into `sys.modules`
  before importing ragas (version-compat; do not remove).
- `Chunk.chunk_id` (`ingestion/schema.py`) is a deterministic UUID → idempotent Qdrant upserts.

## Config & backends
All config in `config.py` via env (`.env.example`). Defaults are embedded/local (no Docker):
Qdrant local-file mode, SQLite, optional Redis with in-process fallback. Switch to servers by
setting `QDRANT_URL`, `DB_URL` (Postgres), `REDIS_URL` (what docker-compose does). Get settings
via `get_settings()`; DB via `db/session.py` (`session_scope()`), models in `db/models.py`.

## API-version conventions in use
Pydantic v2 (`pydantic-settings` `SettingsConfigDict`), SQLAlchemy 2.0 (`Mapped`/`select()`),
FastAPI `lifespan` (not `@app.on_event`), qdrant `collection_exists`/`create_collection`/
`query_points`, `redis.asyncio`, LangGraph `START`/`END`, tz-aware `datetime.now(timezone.utc)`.
Heavy models (bge embedder, reranker, Gemma) are process singletons — load once, never per call.
