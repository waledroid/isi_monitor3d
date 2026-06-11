"""Phase 3 — anti-bleed captions (trigger word + exhaustive background).

  python scripts/run_captions.py --project pallets_v1 [--force]
"""

from __future__ import annotations

import argparse

import _bootstrap
from src.core.runners import run_captions


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", required=True)
    ap.add_argument("--force", action="store_true",
                    help="regenerate non-edited captions")
    args = ap.parse_args()
    print(run_captions(_bootstrap.project_dir(args.project), force=args.force))


if __name__ == "__main__":
    main()
