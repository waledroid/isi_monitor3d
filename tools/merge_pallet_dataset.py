#!/usr/bin/env python3
"""Merge LabelMe pallet datasets into one single-class dataset (`empty_pallet`).

Combines ``video/pallet_v1`` (labels ``Neu``, ``pallets-``) and
``video/pallet_v2`` (label ``Pallet``) into ``video/pallet.v3``, keeping the
**LabelMe** format that X-AnyLabeling reads: each image keeps its sibling
``.json`` sidecar, and **every shape's ``label`` is rewritten to
``empty_pallet``** so the merged set has exactly one class.

Layout: LabelMe is a flat folder of ``<name>.jpg`` + ``<name>.json`` pairs
(open it in X-AnyLabeling via "Open Dir"). Output filenames are prefixed with
the source tag (``v1_`` / ``v2_``) so the two datasets can't collide, and each
json's ``imagePath`` is updated to match its renamed image.

Sources are read-only; only ``video/pallet.v3`` is written. Windows
``:Zone.Identifier`` files are ignored.

Usage:
    python tools/merge_pallet_dataset.py            # build video/pallet.v3
    python tools/merge_pallet_dataset.py --force     # overwrite if it exists
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

TARGET_LABEL = "empty_pallet"

REPO_ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = REPO_ROOT / "video"

# (source_tag, images_dir) — each dir holds <name>.jpg + <name>.json LabelMe pairs.
DEFAULT_SOURCES: list[tuple[str, Path]] = [
    ("v1", VIDEO_DIR / "pallet_v1" / "train" / "images"),
    ("v2", VIDEO_DIR / "pallet_v2" / "train" / "images"),
    ("v2", VIDEO_DIR / "pallet_v2" / "valid" / "images"),
]


def _is_cruft(path: Path) -> bool:
    return path.name.endswith("Zone.Identifier")


def remap_json(src_json: Path, new_image_name: str) -> dict:
    """Load a LabelMe json, set every shape label to TARGET_LABEL, and point
    imagePath at the renamed image. Everything else (points, shape_type,
    imageData, geometry) is preserved verbatim."""
    data = json.loads(src_json.read_text())
    for shape in data.get("shapes", []):
        shape["label"] = TARGET_LABEL
    data["imagePath"] = new_image_name
    return data


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=VIDEO_DIR / "pallet.v3")
    ap.add_argument("--force", action="store_true", help="overwrite existing output dir")
    args = ap.parse_args()

    out_root: Path = args.output
    if out_root.exists():
        if not args.force:
            print(f"ERROR: {out_root} already exists. Re-run with --force to overwrite.")
            return 1
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    pairs = 0
    images_only = 0
    boxes = 0
    print("Merging (LabelMe format):")
    for tag, images_dir in DEFAULT_SOURCES:
        if not images_dir.is_dir():
            print(f"  ! skip {images_dir} (missing)")
            continue
        n_dir = 0
        for img in sorted(images_dir.glob("*.jpg")):
            if _is_cruft(img):
                continue
            new_name = f"{tag}_{img.name}"
            shutil.copy2(img, out_root / new_name)
            sidecar = img.with_suffix(".json")
            if sidecar.exists() and not _is_cruft(sidecar):
                data = remap_json(sidecar, new_name)
                boxes += len(data.get("shapes", []))
                (out_root / f"{tag}_{img.stem}.json").write_text(
                    json.dumps(data, ensure_ascii=False, indent=2)
                )
                pairs += 1
            else:
                images_only += 1
            n_dir += 1
        print(f"  {images_dir.relative_to(REPO_ROOT)}: {n_dir} images")

    print(f"\nWrote {out_root.relative_to(REPO_ROOT)} (flat LabelMe folder)")
    print(f"  image+annotation pairs: {pairs}")
    print(f"  images without json:    {images_only}")
    print(f"  total shapes:           {boxes}  (all label='{TARGET_LABEL}')")
    print("  open in X-AnyLabeling via 'Open Dir' -> video/pallet.v3")
    return 0


if __name__ == "__main__":
    sys.exit(main())
