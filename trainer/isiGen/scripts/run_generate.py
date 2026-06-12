"""Phases 5+7 — init the SDXL ControlNet pipeline and mint synthetics from the
pending scaffolds. Resumable (index updates after every image).

  python scripts/run_generate.py --project pallets_v1 [--limit 5]
"""

from __future__ import annotations

import argparse

import _bootstrap
from src.core.runners import run_generation


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", required=True)
    ap.add_argument("--limit", type=int, default=None,
                    help="mint at most N images this run")
    args = ap.parse_args()
    print(run_generation(_bootstrap.project_dir(args.project), limit=args.limit))


if __name__ == "__main__":
    main()
