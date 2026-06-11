"""Phase 1 — ingest real images into a project.

  python scripts/run_curate.py --project pallets_v1 --source /path/imgs --class-name palette
  python scripts/run_curate.py --project pallets_v1 --source /path/byclass --auto-class
"""

from __future__ import annotations

import argparse

import _bootstrap
from src.core.runners import run_curate


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", required=True)
    ap.add_argument("--source", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--class-name")
    g.add_argument("--auto-class", action="store_true")
    args = ap.parse_args()
    out = run_curate(_bootstrap.project_dir(args.project), source=args.source,
                     class_name=args.class_name, auto_class=args.auto_class)
    print(out)


if __name__ == "__main__":
    main()
