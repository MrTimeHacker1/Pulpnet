"""Live Gemma check — loads the REAL google/gemma-4-E4B-it and generates once.

Heavy (multi-GB download + slow on CPU). NOT part of the default pytest suite.
Run explicitly:  python -m tests.check_gemma_live
"""

from __future__ import annotations

import time


def main() -> None:
    from llm.gemma import GemmaLLM

    t0 = time.time()
    llm = GemmaLLM()
    system = (
        "You are an assistant for IIT Kanpur students. Answer ONLY from the CONTEXT "
        "and cite with [n]."
    )
    user = (
        "CONTEXT:\n[1] CE212 Environment and Sustainability — credits 3-0-0-0-9. Covers "
        "environmental degradation, resource scarcity, air and water pollution.\n\n"
        "QUESTION: How many credits is CE212 and what does it cover? Answer with [n] citations."
    )
    out = llm.generate(system, user, max_new_tokens=128)
    print(f"\n=== Gemma loaded + generated in {time.time()-t0:.0f}s ===")
    print("ANSWER:\n", out)


if __name__ == "__main__":
    main()
