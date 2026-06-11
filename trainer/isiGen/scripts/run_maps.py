"""Phase 2 — dual-layer maps: control maps (depth/canny) + SAM2 ground-truth masks.

  python scripts/run_maps.py --project pallets_v1 --stage all
  python scripts/run_maps.py --project pallets_v1 --stage depth,canny
  python scripts/run_maps.py --project pallets_v1 --stage mask --force
"""

from __future__ import annotations

import argparse

import _bootstrap
from src.core.runners import run_control_maps, run_masks

_STAGE_TO_EXTRACTOR = {"depth": "depth_anything_v2", "canny": "canny"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", required=True)
    ap.add_argument("--stage", default="all",
                    help="comma list of depth|canny|mask, or 'all'")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    pdir = _bootstrap.project_dir(args.project)
    stages = [s.strip() for s in args.stage.split(",")] if args.stage != "all" \
        else ["depth", "canny", "mask"]
    out: dict = {}
    extractors = [_STAGE_TO_EXTRACTOR[s] for s in stages if s in _STAGE_TO_EXTRACTOR]
    if extractors:
        out.update(run_control_maps(pdir, stages=extractors, force=args.force))
    if "mask" in stages:
        out.update(run_masks(pdir, force=args.force))
    print(out)


if __name__ == "__main__":
    main()
