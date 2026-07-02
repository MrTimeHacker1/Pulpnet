"""Reference-free RAGAS evaluation (Faithfulness + ResponseRelevancy +
LLMContextPrecisionWithoutReference + AspectCritic/on_topic).

RAGAS 0.4.3 hard-imports an old `langchain_community.chat_models.vertexai`
integration at package load; the installed langchain stack (1.x) dropped it. We
shim that unused path (we judge with the configured judge LLM, never VertexAI)
BEFORE importing ragas.

The judge LLM is whatever llm.gemma.get_llm() returns (Gemma, Qwen via HF
Inference Providers, or MockLLM — see GENERATOR_MODEL). Embeddings are bge —
both are wrapped into the ragas BaseRagasLLM / BaseRagasEmbeddings interfaces.
Only reference-free metrics are used (they need just question + answer +
contexts, no gold answers).

NOTE: on a CPU-only host the judge can be slow; pass `limit` to evaluate a
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
from ragas.metrics import (  # noqa: E402
    AspectCritic,
    Faithfulness,
    LLMContextPrecisionWithoutReference,
    ResponseRelevancy,
)

from config import configure_logging, get_settings  # noqa: E402
from db.models import EvalScore, QueryLog  # noqa: E402
from db.session import init_db, session_scope  # noqa: E402

logger = logging.getLogger(__name__)
EVAL_QUERIES = Path(__file__).parent / "eval_queries.json"
SAMPLES_CACHE = Path(__file__).parent / "eval_samples_cache.json"


class RagasLLMAdapter(BaseRagasLLM):
    """Adapts the configured judge LLM (see llm.gemma.get_llm()) to the ragas judge-LLM interface."""

    def __init__(self, llm):
        self._llm = llm

    def is_finished(self, response: LLMResult) -> bool:
        return True

    @staticmethod
    def _clean_json(text: str) -> str:
        """Remove trailing empty dicts/objects from JSON arrays.

        Llama 3.3 sometimes appends a stray {} at the end of arrays (e.g. the
        NLIStatementOutput statements list), which fails RAGAS pydantic validation.
        """
        import json
        try:
            obj = json.loads(text)
        except Exception:
            return text  # not JSON — return as-is
        changed = False
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, list):
                    cleaned = [item for item in v if item != {}]
                    if len(cleaned) != len(v):
                        obj[k] = cleaned
                        changed = True
        return json.dumps(obj) if changed else text

    def generate_text(self, prompt, n=1, temperature=1e-8, stop=None, callbacks=None) -> LLMResult:
        text = self._llm.generate("You are a precise evaluation assistant.", prompt.to_string())
        text = self._clean_json(text)
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


def save_samples(records: list[dict]) -> None:
    """Persist records (query/answer/route/contexts) to SAMPLES_CACHE for reuse."""
    SAMPLES_CACHE.write_text(json.dumps(records, indent=2))
    logger.info("Saved %d samples to %s", len(records), SAMPLES_CACHE)


def load_samples() -> tuple[EvaluationDataset, list[dict]]:
    """Reload a previously saved sample cache, skipping the agent phase."""
    records = json.loads(SAMPLES_CACHE.read_text())
    samples = [
        SingleTurnSample(
            user_input=r["query"], response=r["answer"], retrieved_contexts=r["contexts"]
        )
        for r in records
    ]
    logger.info("Loaded %d cached samples from %s", len(records), SAMPLES_CACHE)
    return EvaluationDataset(samples=samples), records


def run_eval(limit: int | None = None, agent=None, use_cache: bool = False) -> dict[str, float]:
    """Evaluate the system on the eval set, write eval_scores, print a summary.

    Pass use_cache=True (or --load-samples CLI flag) to skip the agent phase and
    reuse samples from a previous run saved in eval_samples_cache.json.
    """
    configure_logging()
    init_db()

    from ingestion.chunking.semantic_chunker import BGEEmbedder
    from llm.gemma import get_llm

    if use_cache and SAMPLES_CACHE.exists():
        dataset, records = load_samples()
        if limit:
            dataset = EvaluationDataset(dataset.samples[:limit])
            records = records[:limit]
    else:
        if agent is None:
            from agent.graph import build_agent
            agent = build_agent()

        queries = load_queries()
        if limit:
            queries = queries[:limit]

        dataset, records = build_dataset(agent, queries)
        save_samples(records)

    judge = RagasLLMAdapter(get_llm())
    embeddings = BGERagasEmbeddings(BGEEmbedder())
    metrics = [
        Faithfulness(llm=judge),
        ResponseRelevancy(llm=judge, embeddings=embeddings),
        LLMContextPrecisionWithoutReference(llm=judge),
        AspectCritic(
            name="on_topic",
            definition=(
                "Does the response stay within IIT Kanpur academic topics "
                "(courses, academic policies, or class timetables) without "
                "addressing unrelated or off-scope requests?"
            ),
            llm=judge,
        ),
    ]

    logger.info("Running RAGAS over %d samples…", len(records))
    result = evaluate(dataset=dataset, metrics=metrics, llm=judge, embeddings=embeddings)
    df = result.to_pandas()

    # Sniff column names — ragas may vary them slightly across minor versions.
    faith_col = next((c for c in df.columns if "faith" in c.lower()), None)
    rel_col = next((c for c in df.columns if "relevanc" in c.lower()), None)
    ctx_col = next((c for c in df.columns if "precision" in c.lower()), None)
    topic_col = next((c for c in df.columns if "on_topic" in c.lower()), None)

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
                context_precision=_safe(df, ctx_col, i),
                on_topic=_safe(df, topic_col, i),
            ))

    summary = {
        "faithfulness": float(df[faith_col].mean()) if faith_col else None,
        "answer_relevancy": float(df[rel_col].mean()) if rel_col else None,
        "context_precision": float(df[ctx_col].mean()) if ctx_col else None,
        "on_topic": float(df[topic_col].mean()) if topic_col else None,
        "n": len(records),
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
    ap.add_argument("--load-samples", action="store_true",
                    help="skip agent phase and reuse eval_samples_cache.json from a prior run")
    args = ap.parse_args()
    run_eval(limit=args.limit, use_cache=args.load_samples)
