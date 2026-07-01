"""Input guardrails — the HARD gate before any retrieval/generation.

Three checks (regex/heuristic, no model needed):
  1. PII detection: emails, phone numbers, IITK roll numbers.
  2. Prompt-injection heuristics ("ignore previous instructions", etc.).
  3. Off-topic gate: query must plausibly relate to IITK academics.

Returns (allowed, reason, flags). The router does NOT re-check scope — this is
the single scope gate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# --- PII patterns ---
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s-]?)?(?:\d[\s-]?){10}(?!\d)")
# IITK roll numbers are typically 6-9 digits, sometimes prefixed by year.
ROLL_RE = re.compile(r"\b(?:roll\s*(?:no\.?|number)?\s*[:#-]?\s*)?\d{6,9}\b", re.IGNORECASE)

# --- Prompt-injection heuristics ---
INJECTION_PATTERNS = [
    r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions",
    r"disregard\s+(?:the\s+)?(?:previous|prior|above|system)",
    r"forget\s+(?:everything|all|your\s+instructions)",
    r"you\s+are\s+now\s+(?:a|an|in)\b",
    r"system\s*prompt",
    r"reveal\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions)",
    r"act\s+as\s+(?:a\s+)?(?:dan|jailbreak)",
    r"developer\s+mode",
    r"</?\s*(?:system|assistant|user)\s*>",
]
INJECTION_RE = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)

# --- Off-topic gate ---
# Query must contain at least one IITK-academics signal OR a course-code token.
ON_TOPIC_KEYWORDS = {
    "course", "courses", "credit", "credits", "prerequisite", "prereq", "syllabus",
    "department", "professor", "instructor", "lecture", "tutorial", "lab", "slot",
    "timetable", "schedule", "class", "semester", "exam", "grade", "grading", "cpi",
    "spi", "minor", "major", "degree", "program", "programme", "btech", "mtech",
    "phd", "msc", "iit", "kanpur", "iitk", "manual", "policy", "academic", "ug", "pg",
    "registration", "drop", "audit", "branch", "dual", "thesis", "project", "elective",
    "attendance", "convocation", "scholarship", "hostel", "engineering", "department",
}
COURSE_CODE_RE = re.compile(r"\b[A-Z]{2,4}\d{3}[A-Z]?\b")
WORD_RE = re.compile(r"[a-zA-Z]+")


@dataclass
class GuardResult:
    allowed: bool
    reason: str = ""
    flags: dict[str, bool] = field(default_factory=dict)


def detect_pii(text: str) -> dict[str, bool]:
    return {
        "email": bool(EMAIL_RE.search(text)),
        "phone": bool(PHONE_RE.search(text)),
        "roll_number": bool(ROLL_RE.search(text)),
    }


def detect_injection(text: str) -> bool:
    return bool(INJECTION_RE.search(text))


def is_on_topic(text: str) -> bool:
    if COURSE_CODE_RE.search(text):
        return True
    words = {w.lower() for w in WORD_RE.findall(text)}
    return bool(words & ON_TOPIC_KEYWORDS)


def check_input(query: str) -> GuardResult:
    """Run all input guardrails. Returns a hard allow/deny decision."""
    flags: dict[str, bool] = {}
    if not query or not query.strip():
        return GuardResult(False, "Empty query.", {"empty": True})

    pii = detect_pii(query)
    flags.update({f"pii_{k}": v for k, v in pii.items()})
    if any(pii.values()):
        kinds = ", ".join(k for k, v in pii.items() if v)
        return GuardResult(
            False, f"Query appears to contain personal information ({kinds}). "
            "Please remove it and ask about courses, policies, or schedules.", flags
        )

    injection = detect_injection(query)
    flags["prompt_injection"] = injection
    if injection:
        return GuardResult(
            False, "Query looks like a prompt-injection attempt and was blocked.", flags
        )

    on_topic = is_on_topic(query)
    flags["off_topic"] = not on_topic
    if not on_topic:
        return GuardResult(
            False, "I can only help with IIT Kanpur courses, academic regulations, "
            "and class timetables. Please ask something in that scope.", flags
        )

    return GuardResult(True, "", flags)
