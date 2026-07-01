"""Reference-free RAGAS evaluation (Faithfulness + ResponseRelevancy).

RAGAS 0.4.3 hard-imports an old `langchain_community.chat_models.vertexai`
integration at package load; the installed langchain stack (1.x) dropped it. We
shim that unused path (we judge with our own Gemma, never VertexAI) BEFORE
importing ragas.

The judge LLM is our local Gemma and embeddings are bge — wrapped into the
ragas `BaseRagasLLM` / `BaseRagasEmbeddings` interfaces. Only reference-free
metrics are used (they need just question + answer + contexts, no gold answers).

NOTE: on a CPU-only host the Gemma judge is slow; pass `limit` to evaluate a
subset, or set GENERATOR_MODEL=mock for a wiring smoke-test.
"""

from __future__ import annotations

import json
import logging
import sys
import types
from pathlib import Path
from typing import Any

# --- shim the unused VertexAI import path before importing ragas ---
if "langchain_community.chat_models.vertexai" not in sys.modules:
    _stub = types.ModuleType("langchain_community.chat_models.vertexai")
    _stub.ChatVertexAI = type("ChatVertexAI", (), {})
    sys.modules["langchain_community.chat_models.vertexai"] = _stub

from langchain_core.outputs import Generation, LLMResult  # noqa: E402
from ragas import EvaluationDataset, SingleTurnSample, evaluate  # noqa: E402
from ragas.embeddings.base import BaseRagasEmbeddings  # noqa: E402
from ragas.llms.base import BaseRagasLLM  # noqa: E402
from ragas.metrics import Faithfulness, ResponseRelevancy  # noqa: E402

from config import configure_logging, get_settings  # noqa: E402
from db.models import EvalScore, QueryLog  # noqa: E402
from db.session import init_db, session_scope  # noqa: E402

logger = logging.getLogger(__name__)
EVAL_QUERIES = Path(__file__).parent / "eval_queries.json"


class GemmaRagasLLM(BaseRagasLLM):
    """Adapts our Gemma wrapper to the ragas judge-LLM interface."""

    def __init__(self, llm):
        self._llm = llm

    def is_finished(self, response: LLMResult) -> bool:
        return True

    def generate_text(self, prompt, n=1, temperature=1e-8, stop=None, callbacks=None) -> LLMResult:
        text = self._llm.generate("You are a precise evaluation assistant.", prompt.to_string())
        return LLMResult(generations=[[Generation(text=text)] for _ in range(n)])

    async def agenerate_text(self, prompt, n=1, temperature=1e-8, stop=None, callbacks=None) -> LLMResult:
        return self.generate_text(prompt, n, temperature, stop, callbacks)


class BGERagasEmbeddings(BaseRagasEmbeddings):
    """Adapts bge embedder to the ragas embeddings interface."""

    def __init__(self, embedder):
        self._embedder = embedder

    def embed_query(self, text: str) -> list[float]:
        return [float(x) for x in self._embedder([text])[0]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(x) for x in v] for v in self._embedder(texts)]

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)


def load_queries() -> list[str]:
    return json.loads(EVAL_QUERIES.read_text())["queries"]


def build_dataset(agent, queries: list[str]) -> tuple[EvaluationDataset, list[dict[str, Any]]]:
    """Run the agent over each query → SingleTurnSamples + raw records for logging."""
    from agent.state import AgentState

    samples, records = [], []
    for q in queries:
        result = agent.invoke(AgentState(query=q))
        answer = result.get("answer") or ""
        contexts = [c["text"] for c in result.get("retrieved", [])] or [""]
        samples.append(
            SingleTurnSample(user_input=q, response=answer, retrieved_contexts=contexts)
        )
        records.append({"query": q, "answer": answer, "route": result.get("route"),
                        "contexts": contexts})
        logger.info("Eval sample built: %s", q[:50])
    return EvaluationDataset(samples=samples), records


def run_eval(limit: int | None = None, agent=None) -> dict[str, float]:
    """Evaluate the system on the eval set, write eval_scores, print a summary."""
    configure_logging()
    init_db()
    s = get_settings()

    if agent is None:
        from agent.graph import build_agent

        agent = build_agent()

    from ingestion.chunking.semantic_chunker import BGEEmbedder
    from llm.gemma import get_llm

    queries = load_queries()
    if limit:
        queries = queries[:limit]

    dataset, records = build_dataset(agent, queries)

    judge = GemmaRagasLLM(get_llm())
    embeddings = BGERagasEmbeddings(BGEEmbedder())
    metrics = [Faithfulness(llm=judge), ResponseRelevancy(llm=judge, embeddings=embeddings)]

    logger.info("Running RAGAS over %d samples…", len(queries))
    result = evaluate(dataset=dataset, metrics=metrics, llm=judge, embeddings=embeddings)
    df = result.to_pandas()

    # Persist per-query scores.
    faith_col = next((c for c in df.columns if "faith" in c.lower()), None)
    rel_col = next((c for c in df.columns if "relevanc" in c.lower()), None)
    with session_scope() as ses:
        for i, rec in enumerate(records):
            ql = QueryLog(query=rec["query"], route=rec["route"], answer=rec["answer"],
                          retrieved_chunk_ids=None, guardrail_flags={"eval": True})
            ses.add(ql)
            ses.flush()
            ses.add(EvalScore(
                query_log_id=ql.id,
                faithfulness=_safe(df, faith_col, i),
                answer_relevancy=_safe(df, rel_col, i),
            ))

    summary = {
        "faithfulness": float(df[faith_col].mean()) if faith_col else None,
        "answer_relevancy": float(df[rel_col].mean()) if rel_col else None,
        "n": len(queries),
    }
    print("\n=== RAGAS summary (reference-free) ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return summary


def _safe(df, col, i) -> float | None:
    if not col:
        return None
    try:
        val = float(df[col].iloc[i])
        return val if val == val else None  # NaN guard
    except Exception:
        return None


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N queries")
    run_eval(limit=ap.parse_args().limit)
