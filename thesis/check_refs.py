#!/usr/bin/env python3
"""Validate the manuscript's internal cross-references and house rules.

Run from the repo root:  python thesis/check_refs.py

Checks, in order:
  1. every "Section x[.y[.z]]" mention resolves to a heading that exists
  2. every "Table Tn" and "Figure Fn" mention has a matching definition
  3. the mandated top-level skeleton is present and in order
  4. no em dashes in body text (published titles in the reference list excepted)
Exit status is non-zero if anything fails, so it can gate a build.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MD = ROOT / "MANUSCRIPT.md"

# The author chose a self-contained layout: each system carries its own results, so
# "Results" appears as a named subsection inside Sections 3 and 4 rather than as a
# top-level section. This list guards that intended structure.
MANDATED = ["1. Introduction",
            "2. Materials and methods common to both systems",
            "3. System A: metric multi-camera warehouse monitoring",
            "4. System B: conveyor parcel classification and sorter triggering",
            "5. Discussion", "6. Conclusions"]


def main() -> int:
    text = MD.read_text()
    body, _, refs_section = text.partition("\n# References")
    failures: list[str] = []

    # ---------------------------------------------------------------- headings
    headings = re.findall(r"^#{1,4} (\d+(?:\.\d+)*)\.?\s+(.*)$", body, flags=re.M)
    numbers = {n for n, _ in headings}
    tops = [f"{n}. {t}" for n, t in headings if "." not in n]

    # 1. section references
    for m in re.finditer(r"Sections? (\d+(?:\.\d+)*)", body):
        target = m.group(1)
        if target not in numbers:
            ctx = body[max(0, m.start() - 60):m.end() + 30].replace("\n", " ")
            failures.append(f"dangling section reference {target!r}: ...{ctx}...")

    # 2. tables and figures
    tables = set(re.findall(r"^\*\*Table (T\d+):", body, flags=re.M))
    figures = set(re.findall(r"^\[Figure (F\d+):", body, flags=re.M))
    for kind, defined in (("Table", tables), ("Figure", figures)):
        letter = kind[0]
        for m in re.finditer(rf"{kind} ({letter}\d+)", body):
            if m.group(1) not in defined:
                ctx = body[max(0, m.start() - 50):m.end() + 25].replace("\n", " ")
                failures.append(f"{kind} {m.group(1)} referenced but never defined: ...{ctx}...")

    # 3. mandated skeleton
    if tops != MANDATED:
        failures.append(f"top-level skeleton is {tops}, expected {MANDATED}")

    # 4. em dashes in body text
    for i, line in enumerate(body.split("\n"), 1):
        if "—" in line or re.search(r"(?<!\|)--(?!-)", line):
            # horizontal rules, table rules, and the literal CLI flag are not em dashes
            if line.strip() == "---" or line.startswith("|") or "--fix_intrinsic" in line:
                continue
            failures.append(f"em dash at line {i}: {line.strip()[:80]}")

    print(f"headings: {len(headings)}   tables: {len(tables)}   figures: {len(figures)}")
    if failures:
        print(f"\nFAIL ({len(failures)}):")
        for f in failures:
            print("  -", f)
        return 1
    print("all checks pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
