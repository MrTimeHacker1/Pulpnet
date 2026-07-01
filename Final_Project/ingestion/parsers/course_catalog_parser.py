"""Course catalog ("Courses of Study 2025") parser.

The CoS PDFs are 3-column tables: [Course ID | Credits L-T-P-D-C | Content].
The content column wraps across many physical lines while the code/title/credits
sit at the vertical midpoint of each content block on the left. Naive line-by-line
text extraction scrambles this, so we parse by WORD COORDINATES.

Algorithm (validated against CE-CoS25.pdf → 150/150 complete records):
  1. Find the catalog start page: first page whose text matches
     `DEPARTMENT OF <DEPT>` (case-insensitive — AE uses title-case "Department of AE").
  2. From there, for each page group words into visual lines by `top`.
  3. Assign each word to a column by `x0`: left (<left_max), credits
     (left_max..credits_max, numeric-ish only), content (>=credits_max).
  4. A left token matching the course-code regex is a record ANCHOR at (page, top).
  5. Titles: non-code left tokens attach to the nearest anchor on the SAME page
     within ±TITLE_VWINDOW pt (prevents the next course's title bleeding in).
  6. Credits: join credit-column tokens near each anchor; match a regex handling
     both `3-0-0-0-9` and `3-0-0-0 [9]` and decimals.
  7. Content: each content token belongs to the closest anchor at-or-above it in
     global (page, top) reading order; track page_end for page-spanning blocks.

Column boundaries (left_max=210, credits_max=295) are tuned to CE-CoS25 and
overridable per file via PER_FILE_BOUNDS.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path

import pdfplumber

from ingestion.schema import Chunk

logger = logging.getLogger(__name__)

# --- Tunables ---
DEFAULT_LEFT_MAX = 210.0
DEFAULT_CREDITS_MAX = 295.0
LINE_TOL = 3.0          # words within this many pt of `top` share a visual line
TITLE_VWINDOW = 40.0    # max distance a title token may sit BELOW its anchor
TITLE_ABOVE_TOL = 8.0   # tolerance a title token may sit ABOVE its anchor (baseline jitter)
CREDIT_VWINDOW = 40.0   # vertical window to gather credit tokens around an anchor

# Per-file overrides (left_max, credits_max), derived by dumping extract_words()
# x0 histograms on each file's course-table pages. CE/AE/CSE use the default
# (code/title <210, credits ~230-265, content >300). BSBE & EE shift the whole
# table left: credits land at x0~190, so we tighten the bands.
PER_FILE_BOUNDS: dict[str, tuple[float, float]] = {
    "BSBE-CoS25.pdf": (180.0, 230.0),
    "EE-CoS25.pdf": (180.0, 230.0),
}

CODE_RE = re.compile(r"^[A-Z]{2,4}\d{3}[A-Z]?$")
DEPT_RE = re.compile(r"DEPARTMENT OF\s+([A-Z]{2,4})", re.IGNORECASE)
CREDIT_TOKEN_RE = re.compile(r"^[\d.\-\[\]]+$")
# Handles BOTH the 5-number L-T-P-D-C form (3-0-0-0-9) and the 4-number L-T-P-C
# form (3-0-0-9, 3-0-3-12) — the D (project) field is optional. Also handles the
# bracketed credit (3-0-0-0 [9] / 3-0-0-0[9]) and decimals (1.5-0-0-0-5).
NUM = r"\d+(?:\.\d+)?"
CREDIT_FULL_RE = re.compile(
    rf"^({NUM})-({NUM})-({NUM})(?:-({NUM}))?(?:[-\s]+|\s*)\[?(\d+)\]?$"
)


def _find_start_page(pdf: pdfplumber.PDF) -> tuple[int | None, str | None]:
    """Return (page_index, dept_code) of the first 'DEPARTMENT OF X' page."""
    for i, page in enumerate(pdf.pages):
        text = page.extract_text() or ""
        m = DEPT_RE.search(text)
        if m:
            return i, m.group(1).upper()
    return None, None


def _group_lines(words: list[dict], tol: float = LINE_TOL) -> list[list[dict]]:
    """Group words into visual lines by their `top` coordinate."""
    lines: list[list[dict]] = []
    for w in sorted(words, key=lambda w: (w["top"], w["x0"])):
        if lines and abs(w["top"] - lines[-1][0]["top"]) <= tol:
            lines[-1].append(w)
        else:
            lines.append([w])
    for ln in lines:
        ln.sort(key=lambda w: w["x0"])
    return lines


def _join_line_tokens(tokens: list[dict], line_tol: float = LINE_TOL) -> str:
    """Reconstruct readable text from tokens: group into lines, join with spaces."""
    if not tokens:
        return ""
    ordered = sorted(tokens, key=lambda t: (t["page"], t["top"], t["x0"]))
    out_lines: list[str] = []
    cur: list[dict] = []
    for t in ordered:
        if cur and (t["page"] != cur[-1]["page"] or abs(t["top"] - cur[-1]["top"]) > line_tol):
            out_lines.append(" ".join(x["text"] for x in cur))
            cur = []
        cur.append(t)
    if cur:
        out_lines.append(" ".join(x["text"] for x in cur))
    return " ".join(out_lines).strip()


def _parse_credits(tokens: list[dict]) -> tuple[str | None, dict[str, str] | None]:
    """Join credit tokens (sorted) and match the L-T-P-D-C regex.

    Returns (credits_raw, {L,T,P,D,C}) or (raw_or_None, None) if unmatched.
    """
    if not tokens:
        return None, None
    ordered = sorted(tokens, key=lambda t: (t["top"], t["x0"]))
    pieces = [t["text"] for t in ordered]
    # Try the space-joined form first, then the no-space concatenation
    # (a few courses split credits into two tokens, e.g. "3-0-0-0" + "[9]").
    for candidate in (" ".join(pieces), "".join(pieces)):
        m = CREDIT_FULL_RE.match(candidate.strip())
        if m:
            L, T, P, D, C = m.groups()
            # 4-number L-T-P-C form leaves D unmatched → normalize to "0".
            return candidate.strip(), {"L": L, "T": T, "P": P, "D": D or "0", "C": C}
    return " ".join(pieces).strip() or None, None


def parse_to_chunks(
    pdf_path: str | Path,
    left_max: float | None = None,
    credits_max: float | None = None,
) -> tuple[list[Chunk], str | None]:
    """Parse one CoS PDF into one Chunk per course. Returns (chunks, dept_code)."""
    pdf_path = Path(pdf_path)
    name = pdf_path.name
    lb, cb = PER_FILE_BOUNDS.get(name, (DEFAULT_LEFT_MAX, DEFAULT_CREDITS_MAX))
    if left_max is not None:
        lb = left_max
    if credits_max is not None:
        cb = credits_max

    with pdfplumber.open(pdf_path) as pdf:
        start, dept = _find_start_page(pdf)
        if start is None:
            logger.warning("%s: no 'DEPARTMENT OF' start page found; no records.", name)
            return [], None

        # Collect classified words across catalog pages (tagging each with its page).
        left_tokens: list[dict] = []
        credit_tokens: list[dict] = []
        content_tokens: list[dict] = []
        anchors: list[dict] = []  # {code, page, top}

        for pi in range(start, len(pdf.pages)):
            words = pdf.pages[pi].extract_words(use_text_flow=False)
            for w in words:
                w["page"] = pi
                x0 = w["x0"]
                if x0 < lb:
                    left_tokens.append(w)
                    if CODE_RE.match(w["text"]):
                        anchors.append({"code": w["text"], "page": pi, "top": w["top"]})
                elif x0 < cb:
                    if CREDIT_TOKEN_RE.match(w["text"]):
                        credit_tokens.append(w)
                else:
                    content_tokens.append(w)

        if not anchors:
            logger.warning("%s: start page %d found but no course-code anchors.", name, start)
            return [], dept

        anchors.sort(key=lambda a: (a["page"], a["top"]))

        # --- Titles: non-code left token → nearest same-page anchor within window ---
        title_by_anchor: dict[int, list[dict]] = defaultdict(list)
        anchors_by_page: dict[int, list[dict]] = defaultdict(list)
        for idx, a in enumerate(anchors):
            anchors_by_page[a["page"]].append({**a, "idx": idx})
        for w in left_tokens:
            if CODE_RE.match(w["text"]):
                continue
            cands = anchors_by_page.get(w["page"], [])
            if not cands:
                continue
            best = min(cands, key=lambda a: abs(a["top"] - w["top"]))
            # Title tokens sit on the code's line or wrap BELOW it; they never sit
            # well above the code. The small upward tolerance (TITLE_ABOVE_TOL)
            # absorbs font-baseline jitter while rejecting the column header row
            # that floats above the first anchor (the EE "Courses Title ID" bleed).
            dtop = w["top"] - best["top"]
            if -TITLE_ABOVE_TOL <= dtop <= TITLE_VWINDOW:
                title_by_anchor[best["idx"]].append(w)

        # --- Credits: credit tokens → nearest same-page anchor within window ---
        credit_by_anchor: dict[int, list[dict]] = defaultdict(list)
        for w in credit_tokens:
            cands = anchors_by_page.get(w["page"], [])
            if not cands:
                continue
            best = min(cands, key=lambda a: abs(a["top"] - w["top"]))
            if abs(best["top"] - w["top"]) <= CREDIT_VWINDOW:
                credit_by_anchor[best["idx"]].append(w)

        # --- Content: each content token → closest anchor at-or-above in reading order ---
        content_by_anchor: dict[int, list[dict]] = defaultdict(list)
        anchor_keys = [(a["page"], a["top"]) for a in anchors]
        for w in content_tokens:
            key = (w["page"], w["top"])
            # rightmost anchor whose (page, top) <= token key
            lo, hi = 0, len(anchor_keys)
            while lo < hi:
                mid = (lo + hi) // 2
                if anchor_keys[mid] <= key:
                    lo = mid + 1
                else:
                    hi = mid
            if lo > 0:
                content_by_anchor[lo - 1].append(w)

        # --- Build one Chunk per anchor ---
        chunks: list[Chunk] = []
        for idx, a in enumerate(anchors):
            code = a["code"]
            title = _join_line_tokens(title_by_anchor.get(idx, []))
            content = _join_line_tokens(content_by_anchor.get(idx, []))
            credits_raw, ltpdc = _parse_credits(credit_by_anchor.get(idx, []))

            content_tokens_a = content_by_anchor.get(idx, [])
            page_start = a["page"]
            page_end = max((t["page"] for t in content_tokens_a), default=a["page"])

            text = f"{code} {title}\n\n{content}".strip()
            chunks.append(
                Chunk(
                    text=text,
                    doc_type="course_catalog",
                    source_pdf=name,
                    department=dept,
                    course_code=code,
                    course_title=title or None,
                    credits_raw=credits_raw,
                    page_start=page_start,
                    page_end=page_end,
                    extra=ltpdc or {},
                )
            )

    return chunks, dept


def completeness_report(chunks: list[Chunk]) -> dict[str, int]:
    """Count records complete on each field (for the acceptance check)."""
    return {
        "total": len(chunks),
        "with_code": sum(1 for c in chunks if c.course_code),
        "with_title": sum(1 for c in chunks if c.course_title),
        "with_credits": sum(1 for c in chunks if c.credits_raw and c.extra),
        "with_content": sum(1 for c in chunks if c.text.split("\n\n", 1)[-1].strip()),
    }
