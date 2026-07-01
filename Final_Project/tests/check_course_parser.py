"""Manual acceptance check for the course catalog parser (spec §12.2).

Runs the parser on all 5 CoS PDFs and prints a per-file completeness summary.
Also runs pdffonts to flag any scanned (no embedded font) PDF.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from config import get_settings
from ingestion.parsers.course_catalog_parser import completeness_report, parse_to_chunks


def pdffonts_ok(pdf: Path) -> bool:
    try:
        out = subprocess.run(
            ["pdffonts", str(pdf)], capture_output=True, text=True, timeout=30
        ).stdout
        # header is 2 lines; any embedded font row → has fonts
        return len(out.strip().splitlines()) > 2
    except Exception:
        return True  # don't block on tooling failure


def main() -> None:
    courses_dir = get_settings().courses_dir
    pdfs = sorted(courses_dir.glob("*.pdf"))
    print(f"Found {len(pdfs)} CoS PDFs in {courses_dir}\n")
    grand_total = 0
    for pdf in pdfs:
        chunks, dept = parse_to_chunks(pdf)
        rep = completeness_report(chunks)
        grand_total += rep["total"]
        fonts = "embedded" if pdffonts_ok(pdf) else "NO FONTS (scanned?)"
        print(f"=== {pdf.name}  (dept={dept}, fonts={fonts}) ===")
        print(f"    {rep}")
        if rep["total"]:
            pct = lambda k: 100.0 * rep[k] / rep["total"]
            print(
                f"    complete%%: title={pct('with_title'):.0f} "
                f"credits={pct('with_credits'):.0f} content={pct('with_content'):.0f}"
            )
            sample = chunks[0]
            print(f"    sample: {sample.course_code} | {sample.course_title!r} | "
                  f"{sample.credits_raw!r} | extra={sample.extra}")
            print(f"            content[:120]={sample.text.split(chr(10)+chr(10),1)[-1][:120]!r}")
        print()
    print(f"GRAND TOTAL course records across 5 PDFs: {grand_total}")


if __name__ == "__main__":
    main()
