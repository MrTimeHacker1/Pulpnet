"""End-to-end API smoke test using FastAPI TestClient.

Run with GENERATOR_MODEL=mock for a fast check (no Gemma download), or unset it to
exercise the real model. Hits /health and /query, prints the cited answer with
route/cache_hit/latency/guardrail_flags, and confirms a query_logs row is written.

  GENERATOR_MODEL=mock python -m tests.check_api
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app


def main() -> None:
    with TestClient(app) as client:
        # --- /health ---
        h = client.get("/health").json()
        print("HEALTH:", h)

        # --- /query: a course question ---
        for q in [
            "How many credits is CE212 and what does it cover?",
            "When does AE209 meet during the week?",
            "What is the best pizza topping?",  # out-of-scope / guardrail
        ]:
            r = client.post("/query", json={"query": q}).json()
            print("\nQUERY:", q)
            print("  route:", r.get("route"), "| cache_hit:", r.get("cache_hit"),
                  "| latency_ms:", r.get("latency_ms"))
            print("  guardrail_flags:", r.get("guardrail_flags"))
            print("  citations:", r.get("citations"))
            print("  answer:", (r.get("answer") or "")[:300])

        # --- cache hit on a repeat ---
        client.post("/query", json={"query": "How many credits is CE212 and what does it cover?"})
        stats = client.get("/cache/stats").json()
        print("\nCACHE STATS:", stats)

        # --- confirm a query_logs row exists ---
        from sqlalchemy import func, select

        from db.models import QueryLog
        from db.session import session_scope

        with session_scope() as s:
            n = s.execute(select(func.count()).select_from(QueryLog)).scalar()
        print("query_logs rows:", n)


if __name__ == "__main__":
    main()
