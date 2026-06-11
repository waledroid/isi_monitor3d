#!/usr/bin/env python3
"""Convert a LabelMe dataset to a YOLOv11 detection dataset (single class).

Reads LabelMe ``<name>.json`` files (each with a sibling image), takes the
axis-aligned bounding box of every shape, and writes YOLO detection labels
(``<class_id> cx cy w h``, normalized). Output layout (Ultralytics-ready)::

    <out>/
      images/{train,val}/*.jpg
      labels/{train,val}/*.txt
      data.yaml            # nc + names

Splitting:
  * default — pool a flat LabelMe folder and split fresh (``--val-frac``);
  * ``--preserve-splits`` (or auto-detected when ``--src`` already has
    ``train/`` + ``valid/`` subfolders) — map the existing splits through
    unchanged (LabelMe ``valid`` → YOLO ``val``).

All shapes are mapped to a single class (id 0). Windows ``:Zone.Identifier``
cruft is ignored.

Usage:
    python scripts/labelme_to_yolo.py \
        --src video/pallet_v2_labelme \
        --out trainer/isidet/data/pallet_v2_yolo \
        --class-name palette_vide --preserve-splits
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("labelme2yolo")

REPO_ROOT = Path(__file__).resolve().parents[1]   # .../isi_monitor3d
IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def _find_image(json_path: Path) -> Path | None:
    for ext in IMAGE_EXTS:
        for cand in (json_path.with_suffix(ext), json_path.with_suffix(ext.upper())):
            if cand.exists():
                return cand
    return None


def _bbox_from_points(points: list, w: int, h: int) -> tuple[float, float, float, float] | None:
    """Axis-aligned YOLO bbox (cx, cy, bw, bh) normalized, or None if degenerate."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0 or w <= 0 or h <= 0:
        return None
    cx = (x1 + x2) / 2.0 / w
    cy = (y1 + y2) / 2.0 / h
    nbw = bw / w
    nbh = bh / h
    clamp = lambda v: max(0.0, min(1.0, v))  # noqa: E731
    return clamp(cx), clamp(cy), clamp(nbw), clamp(nbh)


def _polygon_from_points(points: list, w: int, h: int) -> list[float] | None:
    """Flat normalized [x1, y1, x2, y2, ...] polygon, clamped to [0, 1]. Returns
    None if the polygon is degenerate (<3 points or zero-size frame)."""
    if w <= 0 or h <= 0 or len(points) < 3:
        return None
    out: list[float] = []
    for x, y in points:
        out.append(max(0.0, min(1.0, float(x) / w)))
        out.append(max(0.0, min(1.0, float(y) / h)))
    return out


def convert_one(json_path: Path, task: str = "detect") -> list[str] | None:
    """Return YOLO label lines for one LabelMe json, or None if no image.

    ``task='detect'`` → ``class cx cy w h`` (AABB of each shape, current default).
    ``task='seg'`` → ``class x1 y1 x2 y2 ...`` (the polygon points; only
    ``shape_type='polygon'`` shapes are emitted)."""
    if _find_image(json_path) is None:
        return None
    data = json.loads(json_path.read_text())
    w = int(data.get("imageWidth") or 0)
    h = int(data.get("imageHeight") or 0)
    lines: list[str] = []
    for shape in data.get("shapes", []):
        pts = shape.get("points") or []
        if task == "seg":
            if shape.get("shape_type") != "polygon":
                continue
            poly = _polygon_from_points(pts, w, h)
            if poly is None:
                continue
            lines.append("0 " + " ".join(f"{v:.6f}" for v in poly))
        else:  # detect (default, back-compat)
            if len(pts) < 2:
                continue
            box = _bbox_from_points(pts, w, h)
            if box is None:
                continue
            cx, cy, bw, bh = box
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=REPO_ROOT / "video" / "pallet.v3",
                    help="LabelMe folder (flat <name>.jpg + <name>.json, or split into train/valid)")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "trainer" / "isidet" / "data" / "pallet_yolo",
                    help="output YOLO dataset root")
    ap.add_argument("--class-name", default="empty_pallet")
    ap.add_argument("--task", choices=("detect", "seg"), default="detect",
                    help="emit YOLO detect bboxes (class cx cy w h) — default, "
                         "back-compat — or YOLO-seg polygons (class x1 y1 x2 y2 …)")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--preserve-splits", action="store_true",
                    help="src already has train/ + valid/ (and optional test/) "
                         "LabelMe subfolders — map them to YOLO splits as-is "
                         "instead of pooling + re-splitting")
    ap.add_argument("--force", action="store_true", help="overwrite existing output dir")
    args = ap.parse_args()

    src: Path = args.src
    out: Path = args.out
    if not src.is_dir():
        logger.error(f"❌ source folder not found: {src}")
        return 1
    if out.exists():
        if not args.force:
            logger.error(f"❌ {out} already exists. Re-run with --force to overwrite.")
            return 1
        shutil.rmtree(out)

    def _gather(folder: Path) -> tuple[list, int]:
        """Return ([(img, yolo_lines), ...], total_shapes) for a flat LabelMe folder."""
        out_samples, n_shapes = [], 0
        for jp in sorted(folder.glob("*.json")):
            if jp.name.endswith("Zone.Identifier"):
                continue
            img = _find_image(jp)
            if img is None:
                logger.warning(f"⚠️ no image for {jp.name} — skipped")
                continue
            lines = convert_one(jp, task=args.task)
            if lines is None:
                continue
            n_shapes += len(lines)
            out_samples.append((img, lines))
        return out_samples, n_shapes

    # Auto-detect an already-split source (train/ + valid/ subfolders).
    has_subsplits = (src / "train").is_dir() and (src / "valid").is_dir()
    preserve = args.preserve_splits or has_subsplits

    shape_count = 0
    if preserve:
        # YOLO uses 'val'; LabelMe split here uses 'valid'.
        mapping = {"train": "train", "valid": "val"}
        if (src / "test").is_dir():
            mapping["test"] = "test"
        splits = {}
        for lm_split, yolo_split in mapping.items():
            samples, n = _gather(src / lm_split)
            splits[yolo_split] = samples
            shape_count += n
        logger.info(f"📁 Preserving existing splits: { {k: len(v) for k, v in splits.items()} }")
    else:
        samples, shape_count = _gather(src)
        if not samples:
            logger.error("❌ no image+annotation pairs found.")
            return 1
        random.Random(args.seed).shuffle(samples)
        n_val = round(len(samples) * args.val_frac)
        splits = {"val": samples[:n_val], "train": samples[n_val:]}

    if not any(splits.values()):
        logger.error("❌ no image+annotation pairs found.")
        return 1

    for split, items in splits.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img, lines in items:
            shutil.copy2(img, out / "images" / split / img.name)
            (out / "labels" / split / f"{img.stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else "")
            )

    has_test = "test" in splits and splits["test"]
    data_yaml = (
        f"path: {out.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        + ("test: images/test\n" if has_test else "")
        + "nc: 1\n"
        f"names: ['{args.class_name}']\n"
    )
    (out / "data.yaml").write_text(data_yaml)

    logger.info(f"✅ Wrote {out}  (task={args.task})")
    units = "polygons" if args.task == "seg" else "boxes"
    logger.info(f"   train={len(splits.get('train', []))}  val={len(splits.get('val', []))}  total_{units}={shape_count}")
    logger.info(f"   single class id 0 = '{args.class_name}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
