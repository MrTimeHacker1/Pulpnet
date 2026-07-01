# IITK Agentic RAG Platform

A production-grade Agentic RAG backend that answers IIT Kanpur student questions about
**courses**, **academic regulations**, and **class timetables**, grounded in the official PDFs.
Backend through FastAPI (no frontend).

## Architecture

```
User query
  → Input guardrails (PII / prompt-injection / off-topic HARD gate)
  → Semantic cache (embedding-similarity hit → return cached)
  → LangGraph agent (4 nodes, one real conditional loop):
        router            classify course | policy | timetable | out_of_scope
                          (timetable + course-code → DIRECT Postgres lookup, bypassing vectors)
        retrieve_rerank   hybrid BM25 + dense (in-memory) → bge cross-encoder rerank
        sufficiency       if top rerank score < threshold and retries left → rewrite + loop
        generate          Gemma 4 E4B-it, grounded + cited
  → Output guardrails (LLM-judged faithfulness/grounding → refuse if unsupported)
  → Cache write + Postgres query log
  → Response + citations
```

### Components
| Concern | Choice |
|---|---|
| PDF parsing | `pdfplumber` (word-coordinate course parser; table-based timetable parser) |
| Embeddings | `BAAI/bge-base-en-v1.5` (sentence-transformers, normalized) |
| Reranker | `BAAI/bge-reranker-base` cross-encoder |
| Generator | `google/gemma-4-E4B-it` (transformers, thinking OFF, temp 0.3) |
| Vector store | **Qdrant** (embedded local-file mode by default) |
| Sparse | BM25 (`rank-bm25`), in-process |
| Agent | **LangGraph** |
| Cache | **Redis** (`redis.asyncio`) with in-process fallback |
| Persistence | **SQLAlchemy 2.0** → SQLite (dev) / Postgres (prod) |
| Eval | **RAGAS** reference-free (Faithfulness, ResponseRelevancy) |
| API | **FastAPI** + `uvicorn` (lifespan), Pydantic v2 |

## Data layout (input, already present)
```
data/
├── courses/            AE, BSBE, CE, CSE, EE  "Courses of Study 2025" PDFs
├── academic_documents/ UG_Manual.pdf, PG-Manual.pdf
└── timetable/          Course_Schedule_2026-27-1.pdf
```
- `courses/` → structured course records → **Qdrant**
- `academic_documents/` → semantic-chunked policy prose → **Qdrant**
- `timetable/` → structured rows → **PostgreSQL/SQLite** (exact lookup, NOT embedded)

## Quickstart (Docker — recommended)

Fully containerized: app + ingestion + RAGAS eval + Qdrant + Postgres + Redis.

```bash
docker compose up --build
```
This brings up the backends, runs a one-shot **ingest** job (parse PDFs → embed →
Qdrant + Postgres + retrieval snapshot), then starts the **API on http://localhost:8000**.
The `app` waits for ingestion to finish; model weights are cached on a shared volume so
they download only once.

```bash
# Health + a query
curl -s localhost:8000/health | python -m json.tool
curl -s -X POST localhost:8000/query -H 'content-type: application/json' \
  -d '{"query":"How many credits is CE212 and what does it cover?"}' | python -m json.tool

# RAGAS evaluation (reference-free), on demand:
docker compose --profile eval run --rm eval
```

Useful toggles:
- **Fast, no heavy model:** `GENERATOR_MODEL=mock docker compose up --build` (instant answers).
- **GPU:** install the NVIDIA Container Toolkit and uncomment the `deploy:` blocks in
  `docker-compose.yml`; `MODEL_DEVICE=auto` then uses the GPU (torch's CUDA wheel is already bundled).
- **Re-ingest:** `docker compose run --rm ingest` (idempotent), or remove the `artifacts` volume.

## Setup (local, without Docker)

```bash
pip install -r requirements.txt
pip check
# Optional: copy and edit environment
cp .env.example .env
```

### Backends
This repo defaults to **zero-infra embedded mode** (no Docker needed):
- Qdrant runs in **local-file mode** at `./qdrant_storage`.
- The DB defaults to **SQLite** (`rag.db`). Switch to Postgres by setting `DB_URL`.
- Redis is **optional**; if unreachable the cache transparently falls back in-process.

For the documented production topology (Qdrant + Postgres + Redis as services):
```bash
docker compose up -d          # requires Docker (not used in dev mode)
# then in .env: QDRANT_URL=..., DB_URL=postgresql+psycopg://..., REDIS_URL=...
```

### Ingest
```bash
python -m ingestion.run_ingestion
```
This parses all PDFs, embeds course + manual chunks into Qdrant, and loads timetable rows
into the DB. Idempotent (re-running replaces a PDF's content, never duplicates).

### Serve
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
Endpoints: `GET /health`, `POST /query`, `POST /ingest`, `POST /eval/run`, `GET /cache/stats`.

Example:
```bash
curl -s localhost:8000/health
curl -s -X POST localhost:8000/query -H 'content-type: application/json' \
  -d '{"query":"How many credits is CE212 and what does it cover?"}'
```

### Evaluate
```bash
python -m eval.ragas_eval --limit 5     # reference-free RAGAS over the eval set
```

## Tests
```bash
pytest -q                  # full unit suite (parsers, chunker, guardrails, cache, agent)
python -m tests.check_course_parser     # per-PDF course extraction summary
python -m tests.check_retrieval         # probe-query retrieval + rerank
```

## Environment notes — tested live vs. mocked (honest)

This was built and run on a shared host with specific constraints; here is exactly what was
exercised live versus mocked:

- **Course/timetable/academic parsers** — tested live on the real PDFs. Course extraction:
  100% code+content, ~99% title across all 5 departments; credits extracted wherever the
  source PDF provides them (CE 150/150; some grad CSE courses genuinely omit credits in the
  source — verified with `pdftotext`). Timetable: 1630 meeting-rows, 684 codes, correct
  `Th`/`T` day disambiguation.
- **Embedding + Qdrant ingestion** — tested live; full corpus (≈1165 chunks) embedded with
  bge-base and indexed into Qdrant.
- **Hybrid retrieval + rerank** — tested live against the populated index.
- **Agent graph (routing, sufficiency loop, timetable fast-path)** — tested live with mock
  LLM/searcher/reranker (deterministic), plus end-to-end via the API.
- **Guardrails + semantic cache** — unit-tested live (PII/injection blocked, paraphrase cache
  hit) with the in-process cache fallback.
- **Full API pipeline** — tested live end-to-end via `tests/check_api.py` (with `GENERATOR_MODEL=mock`
  for speed): `/health` green, routing, retrieval+rerank, input guardrail blocking off-topic,
  output-guardrail grounding, semantic cache exact-hit on a repeat (real hit-rate reported), and
  `query_logs` rows written. Full unit suite: **48 passed**.
- **Gemma generator** — `google/gemma-4-E4B-it`, **verified LIVE**: the model loads and produces a
  grounded, cited answer (e.g. *"CE212 ... covers environmental degradation, resource scarcity, air
  and water pollution [1]."*). It is a multimodal checkpoint, so we use the **text path only** via
  `AutoTokenizer` + `AutoModelForCausalLM` (avoids the vision/`torchvision` dependency). On this
  CPU-only host a cold load + one generation takes ~20 min; unit tests use `MockLLM` and
  `GENERATOR_MODEL=mock` runs the whole pipeline instantly. Run `python -m tests.check_gemma_live`
  to reproduce the live generation.
- **RAGAS** — wired and runs end-to-end (dataset assembly → Faithfulness + ResponseRelevancy →
  `eval_scores` written). Non-NaN scores require the real Gemma judge (the mock can't emit RAGAS's
  structured output). Use `--limit` for a bounded live run.

### Serving architecture note (important)
On this host, importing `qdrant-client` into a process that also runs torch inference causes
intermittent native segfaults. This is handled architecturally, not worked around per-call:
- **Ingestion** embeds everything first (torch), writes a **retrieval snapshot** (`retrieval_snapshot.pkl`),
  and performs the Qdrant upsert in an **isolated, torch-free subprocess**.
- **Serving** (`HybridSearcher`) loads that snapshot into memory and does dense (numpy cosine) +
  BM25 + RRF fully in-process — it **never imports qdrant-client**, so torch stays stable.
Qdrant remains the canonical, idempotent vector store; the snapshot is the torch-safe read path.

### Known environment quirks handled
- **No Docker** → embedded/local backends (Qdrant file mode, SQLite, optional Redis).
- **Mismatched `torchvision`** (broke `transformers` import) → removed; not needed for text.
- **torch CUDA build vs. driver mismatch** → GPU unavailable here; device falls back to CPU
  automatically (`MODEL_DEVICE=auto`). On a box with a matching CUDA driver it uses the GPU.
- **torch ↔ Qdrant-local native segfault** when embedding and vector-writes interleave →
  ingestion runs in two phases (embed, then a torch-free indexing subprocess), and the API
  loads all vectors into memory once so retrieval never calls Qdrant-native alongside torch.
- **RAGAS ↔ langchain 1.x** import incompatibility → unused VertexAI import path is shimmed.
```
