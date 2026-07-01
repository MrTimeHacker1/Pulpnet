"""Output guardrail — faithfulness / grounding check (= source verification).

One merged step: an LLM judge decides whether the answer's claims are supported by
the retrieved context. If grounding is low (or there is no context), the answer is
replaced with a safe refusal. Returns (final_answer, flags).
"""

from __future__ import annotations

import logging
import re

from llm.gemma import LLM, get_llm

logger = logging.getLogger(__name__)

REFUSAL = (
    "I couldn't find enough grounded information in the official IIT Kanpur "
    "documents to answer that confidently. Please rephrase or ask about a specific "
    "course, regulation, or schedule."
)

JUDGE_SYSTEM = (
    "You are a strict faithfulness judge for a retrieval-augmented system. "
    "Given a CONTEXT and an ANSWER, decide whether every factual claim in the ANSWER "
    "is supported by the CONTEXT. Reply on a single line as:\n"
    "VERDICT: GROUNDED or UNSUPPORTED\nSCORE: <0.0-1.0>\n"
    "Do not explain."
)

_SCORE_RE = re.compile(r"score\s*[:=]\s*([01](?:\.\d+)?)", re.IGNORECASE)
_UNSUPPORTED_RE = re.compile(r"\bunsupported\b|\bnot\s+grounded\b|\bungrounded\b", re.IGNORECASE)
_GROUNDED_RE = re.compile(r"\bgrounded\b|\bsupported\b", re.IGNORECASE)


def _parse_verdict(text: str) -> tuple[bool, float | None]:
    score = None
    m = _SCORE_RE.search(text)
    if m:
        score = float(m.group(1))
    if _UNSUPPORTED_RE.search(text):
        return False, score
    if _GROUNDED_RE.search(text):
        return True, score
    # Fall back to the score if no explicit verdict word.
    if score is not None:
        return score >= 0.5, score
    return False, score


def check_output(
    answer: str,
    contexts: list[str],
    llm: LLM | None = None,
    min_score: float = 0.5,
) -> tuple[str, dict]:
    """Verify grounding; refuse if unsupported. Returns (answer_or_refusal, flags)."""
    flags: dict = {"grounded": None, "grounding_score": None, "refused": False}

    if not answer or not answer.strip():
        flags.update(grounded=False, refused=True)
        return REFUSAL, flags
    if not contexts:
        flags.update(grounded=False, refused=True, reason="no_context")
        return REFUSAL, flags

    llm = llm or get_llm()
    context_block = "\n\n---\n\n".join(contexts)
    user = f"CONTEXT:\n{context_block}\n\nANSWER:\n{answer}\n\nJudge now."
    try:
        verdict_text = llm.generate(JUDGE_SYSTEM, user, max_new_tokens=32)
    except Exception as e:  # judge failure → fail closed (refuse)
        logger.warning("Output guardrail judge failed: %s", e)
        flags.update(grounded=False, refused=True, reason="judge_error")
        return REFUSAL, flags

    grounded, score = _parse_verdict(verdict_text)
    flags["grounded"] = grounded
    flags["grounding_score"] = score
    effective = score if score is not None else (1.0 if grounded else 0.0)
    if not grounded or effective < min_score:
        flags["refused"] = True
        return REFUSAL, flags
    return answer, flags
