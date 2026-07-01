"""Manual acceptance check for retrieval + rerank (spec §12.7).

Runs the two probe queries and prints hybrid hits + reranked order so we can
eyeball that relevant chunks return and rerank reorders sensibly.
"""

from __future__ import annotations

from retrieval.hybrid_search import HybridSearcher
from retrieval.reranker import Reranker

PROBES = [
    ("machine learning courses in CE", {"doc_type": "course_catalog"}),
    ("credit requirements for CSE minor", {}),
]


def main() -> None:
    searcher = HybridSearcher()
    reranker = Reranker()
    for query, flt in PROBES:
        print(f"\n{'='*70}\nQUERY: {query!r}  filter={flt}")
        hits = searcher.search(query, top_k=20, **flt)
        print(f"  hybrid returned {len(hits)} hits; top-5 by RRF:")
        for h in hits[:5]:
            tag = h.payload.get("course_code") or h.payload.get("section") or h.payload.get("doc_type")
            print(f"    [{tag}] rrf={h.score:.4f}  {h.text[:80]!r}")
        reranked = reranker.rerank(query, hits, top_k=5)
        print("  reranked top-5:")
        for r in reranked:
            tag = r.payload.get("course_code") or r.payload.get("section") or r.payload.get("doc_type")
            print(f"    [{tag}] ce={r.rerank_score:.3f}  {r.text[:80]!r}")


if __name__ == "__main__":
    main()
