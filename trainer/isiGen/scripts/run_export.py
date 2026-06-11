"""Phase 8 — CLIP-filter the synthetics, then export the dataset.

  python scripts/run_export.py --project pallets_v1 [--skip-filter]
"""

from __future__ import annotations

import argparse

import _bootstrap
from src.core.runners import run_export, run_filter


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", required=True)
    ap.add_argument("--skip-filter", action="store_true")
    args = ap.parse_args()
    pdir = _bootstrap.project_dir(args.project)
    out: dict = {}
    if not args.skip_filter:
        out.update(run_filter(pdir))
    out.update(run_export(pdir))
    print(out)


if __name__ == "__main__":
    main()
