"""Phase 6 — synthetic scaffolds: paired (control map, ground-truth mask).

  python scripts/run_scaffolds.py --project pallets_v1 [--count 200]
"""

from __future__ import annotations

import argparse

import _bootstrap
from src.core.runners import run_scaffolds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", required=True)
    ap.add_argument("--count", type=int, default=None,
                    help="override phases.scaffolds.count")
    args = ap.parse_args()
    print(run_scaffolds(_bootstrap.project_dir(args.project), count=args.count))


if __name__ == "__main__":
    main()
