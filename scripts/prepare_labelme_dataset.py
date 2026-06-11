#!/usr/bin/env python3
"""Build a clean LabelMe dataset from a folder of images + LabelMe json.

Recursively gathers ``<name>.json`` LabelMe sidecars (+ their sibling image)
from one or more source dirs, optionally rewrites every shape's ``label`` to a
single target class, fixes each json's ``imagePath``, drops Windows
``:Zone.Identifier`` cruft and orphan/duplicate stems, and writes a *proper*
LabelMe dataset (each output folder is a flat ``image + .json`` set you can open
in X-AnyLabeling via "Open Dir").

Two output modes:
  * split (default, ``--val-frac > 0``)::

        <out>/train/  <name>.jpg + <name>.json
        <out>/valid/  <name>.jpg + <name>.json

  * flat (``--val-frac 0``) — a single normalized LabelMe folder::

        <out>/  <name>.jpg + <name>.json

Pass ``--class-name ""`` to keep original labels (only clean + split).

Usage:
    # rename class + 80/20 split
    python scripts/prepare_labelme_dataset.py \
        --src video/pallet_v2 --out video/pallet_v2_labelme \
        --class-name palette_vide --val-frac 0.2

    # just normalize into one clean LabelMe folder (no split, keep labels)
    python scripts/prepare_labelme_dataset.py \
        --src raw_dump --out clean_labelme --class-name "" --val-frac 0
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def _is_cruft(p: Path) -> bool:
    return p.name.endswith("Zone.Identifier")


def _find_image(json_path: Path) -> Path | None:
    for ext in IMAGE_EXTS:
        for cand in (json_path.with_suffix(ext), json_path.with_suffix(ext.upper())):
            if cand.exists() and not _is_cruft(cand):
                return cand
    return None


def _write_pair(img: Path, jp: Path, dest: Path, class_name: str) -> int:
    """Copy image + (relabeled) json into dest. Returns shape count."""
    shutil.copy2(img, dest / img.name)
    data = json.loads(jp.read_text())
    n = 0
    for shape in data.get("shapes", []):
        if class_name:
            shape["label"] = class_name
        n += 1
    data["imagePath"] = img.name
    (dest / f"{jp.stem}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2))
    return n


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--src", type=Path, nargs="+", required=True,
                    help="source dir(s) searched recursively for LabelMe json")
    ap.add_argument("--out", type=Path, required=True, help="output dataset root")
    ap.add_argument("--class-name", default="palette_vide",
                    help="rename every shape to this class; empty string keeps originals")
    ap.add_argument("--val-frac", type=float, default=0.2,
                    help="fraction for the valid split; 0 -> single flat folder (no split)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="overwrite existing output")
    args = ap.parse_args()

    out: Path = args.out
    if out.exists():
        if not args.force:
            print(f"ERROR: {out} already exists. Re-run with --force to overwrite.")
            return 1
        shutil.rmtree(out)

    # Collect image+json pairs from all sources (dedupe by stem).
    pairs: list[tuple[Path, Path]] = []
    seen: set[str] = set()
    for src in args.src:
        if not src.is_dir():
            print(f"ERROR: source not found: {src}")
            return 1
        for jp in sorted(src.rglob("*.json")):
            if _is_cruft(jp):
                continue
            img = _find_image(jp)
            if img is None:
                print(f"  ! no image for {jp.name} — skipped")
                continue
            if jp.stem in seen:
                print(f"  ! duplicate stem {jp.stem!r} — skipped")
                continue
            seen.add(jp.stem)
            pairs.append((img, jp))

    if not pairs:
        print("ERROR: no image+json pairs found.")
        return 1

    boxes = 0
    if args.val_frac and args.val_frac > 0:
        random.Random(args.seed).shuffle(pairs)
        n_val = round(len(pairs) * args.val_frac)
        splits = {"valid": pairs[:n_val], "train": pairs[n_val:]}
        for split, items in splits.items():
            (out / split).mkdir(parents=True, exist_ok=True)
            for img, jp in items:
                boxes += _write_pair(img, jp, out / split, args.class_name)
        print(f"Wrote {out} (LabelMe, split)"
              + (f", class='{args.class_name}'" if args.class_name else ", labels kept"))
        print(f"  train: {len(splits['train'])} images")
        print(f"  valid: {len(splits['valid'])} images")
    else:
        out.mkdir(parents=True, exist_ok=True)
        for img, jp in pairs:
            boxes += _write_pair(img, jp, out, args.class_name)
        print(f"Wrote {out} (LabelMe, flat)"
              + (f", class='{args.class_name}'" if args.class_name else ", labels kept"))
        print(f"  images: {len(pairs)}")

    print(f"  total: {len(pairs)} images, {boxes} shapes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
