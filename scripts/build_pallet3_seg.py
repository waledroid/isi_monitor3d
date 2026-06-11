#!/usr/bin/env python3
"""Build the 3-class YOLO-seg dataset (palette, carton, polybag) for training.

Sources:
  • palette: a flat LabelMe folder (image + .json polygons, single 'palette'
    class)              → class 0, images renamed ``pallets-NNNNN``
  • colis:   a YOLO-seg dataset (carton=0, polybag=1, images/ + labels/ splits)
                        → remapped carton=1, polybag=2, images renamed ``colis-NNNNN``

Everything is pooled and re-split train/val, so val sees all three classes. The
``pallets-`` / ``colis-`` prefixes keep each source findable in the merged set.

Output (Ultralytics YOLO-seg layout):
    <out>/
      images/{train,val}/  pallets-00001.jpg, colis-00001.jpg, …
      labels/{train,val}/  pallets-00001.txt  (``<cls> x1 y1 … xn yn`` normalised)
      data.yaml            nc: 3, names: [palette, carton, polybag]

    python scripts/build_pallet3_seg.py \
      --labelme trainer/isidet/data/pallet_universal_labelme \
      --colis   trainer/isidet/data/colis_universal_dataset \
      --out     trainer/isidet/data/pallet3_yolo_seg --force
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import shutil
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError:                       # pragma: no cover
    Image = None
import cv2

_AUG_RE = re.compile(r"_aug\d+$", re.IGNORECASE)   # Roboflow offline-aug suffix

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("build_pallet3_seg")

CLASSES = ["palette", "carton", "polybag"]
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def image_wh(path: Path) -> tuple[int, int]:
    if Image is not None:
        try:
            with Image.open(path) as im:
                return im.size
        except Exception:
            pass
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"cannot read image: {path}")
    h, w = img.shape[:2]
    return w, h


def _sibling_image(json_path: Path) -> Path | None:
    for ext in IMG_EXTS:
        p = json_path.with_suffix(ext)
        if p.exists():
            return p
    return None


def labelme_seg_lines(json_path: Path, image: Path, cls_id: int) -> list[str]:
    """LabelMe shapes → YOLO-seg lines (normalised polygons), all mapped to
    ``cls_id``. Rectangles (2-pt) are expanded to 4 corners."""
    data = json.loads(json_path.read_text())
    w = data.get("imageWidth") or 0
    h = data.get("imageHeight") or 0
    if not (w and h):
        w, h = image_wh(image)
    lines: list[str] = []
    for s in data.get("shapes", []):
        pts = s.get("points") or []
        if s.get("shape_type") == "rectangle" and len(pts) == 2:
            (x1, y1), (x2, y2) = pts
            pts = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        if len(pts) < 3:
            continue
        norm: list[float] = []
        for x, y in pts:
            norm.append(min(max(x / w, 0.0), 1.0))
            norm.append(min(max(y / h, 0.0), 1.0))
        lines.append(f"{cls_id} " + " ".join(f"{v:.6f}" for v in norm))
    return lines


def yolo_seg_remap_lines(label_path: Path, class_offset: int) -> list[str]:
    """Read YOLO-seg lines, shift the class id by ``class_offset`` (bbox lines are
    expanded to a 4-corner polygon so the merged set is uniformly polygons)."""
    out: list[str] = []
    for line in label_path.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        try:
            cls = int(float(p[0])) + class_offset
            coords = [float(t) for t in p[1:]]
        except ValueError:
            continue
        if len(coords) == 4:                          # bbox cx,cy,w,h → 4 corners
            cx, cy, bw, bh = coords
            x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
            coords = [x1, y1, x2, y1, x2, y2, x1, y2]
        elif len(coords) < 6 or len(coords) % 2:
            continue
        out.append(f"{cls} " + " ".join(f"{v:.6f}" for v in coords))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labelme", type=Path, required=True, help="palette LabelMe flat folder")
    ap.add_argument("--colis", type=Path, required=True, help="colis YOLO-seg dataset")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--val-frac", type=float, default=0.15,
                    help="palette val fraction (colis split preserved when --preserve-colis-split)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--colis-originals-only", action="store_true",
                    help="skip colis _augN offline-augmented copies — keep originals + augment online")
    ap.add_argument("--preserve-colis-split", action="store_true",
                    help="keep colis's own train/val split (leak-free) instead of re-pooling its variants")
    args = ap.parse_args()

    if args.out.exists():
        if not args.force:
            print(f"error: {args.out} exists — pass --force", file=sys.stderr)
            return 2
        shutil.rmtree(args.out)

    # Each entry: (new_stem, src_image, label_lines, fixed_split | None)
    # fixed_split=None means "assign by the global palette val-frac split".
    pal_samples: list[tuple[str, Path, list[str], str | None]] = []
    col_samples: list[tuple[str, Path, list[str], str | None]] = []

    # ---- palette LabelMe → class 0, renamed pallets-NNNNN (split by val-frac) ----
    n_pal = 0
    for jf in sorted(args.labelme.glob("*.json")):
        if jf.name.startswith("dataset"):
            continue
        img = _sibling_image(jf)
        if img is None:
            continue
        lines = labelme_seg_lines(jf, img, cls_id=0)
        if not lines:
            continue
        n_pal += 1
        pal_samples.append((f"pallets-{n_pal:05d}", img, lines, None))
    log.info("palette: %d images → class 0 (pallets-*)", n_pal)

    # ---- colis YOLO-seg → carton=1, polybag=2, renamed colis-NNNNN ----
    n_col = skipped_aug = 0
    for split in ("train", "val", "test"):
        img_dir = args.colis / "images" / split
        lbl_dir = args.colis / "labels" / split
        if not lbl_dir.is_dir():
            continue
        for lf in sorted(lbl_dir.glob("*.txt")):
            if lf.name.endswith("Zone.Identifier") or lf.name == "classes.txt":
                continue
            if args.colis_originals_only and _AUG_RE.search(lf.stem):
                skipped_aug += 1
                continue
            img = next((img_dir / f"{lf.stem}{e}" for e in IMG_EXTS
                        if (img_dir / f"{lf.stem}{e}").exists()), None)
            if img is None:
                continue
            lines = yolo_seg_remap_lines(lf, class_offset=1)
            if not lines:
                continue
            n_col += 1
            # Preserve colis's leak-free split (test folds into train).
            fixed = ("val" if split == "val" else "train") if args.preserve_colis_split else None
            col_samples.append((f"colis-{n_col:05d}", img, lines, fixed))
    log.info("colis: %d images → classes 1/2 (colis-*)%s", n_col,
             f"  (skipped {skipped_aug} _aug copies)" if skipped_aug else "")

    if not (pal_samples or col_samples):
        print("no samples", file=sys.stderr)
        return 1

    # Build the split: fixed_split assignments are honoured; the rest (palette,
    # and colis when not preserving its split) are split by val-frac.
    groups: dict[str, list] = {"train": [], "val": []}
    floating = [s for s in (pal_samples + col_samples) if s[3] is None]
    for s in (pal_samples + col_samples):
        if s[3] is not None:
            groups[s[3]].append(s)
    random.Random(args.seed).shuffle(floating)
    n_val = round(len(floating) * args.val_frac)
    groups["val"] += floating[:n_val]
    groups["train"] += floating[n_val:]

    cls_counts = {0: 0, 1: 0, 2: 0}
    for split, grp in groups.items():
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for stem, img, lines, _fixed in grp:
            shutil.copy2(img, args.out / "images" / split / f"{stem}{img.suffix.lower()}")
            (args.out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n")
            for ln in lines:
                cls_counts[int(ln.split()[0])] += 1

    (args.out / "data.yaml").write_text(
        f"path: {args.out.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(CLASSES)}\n"
        f"names: {CLASSES}\n"
    )
    summary = {
        "out": str(args.out),
        "classes": CLASSES,
        "images": {"train": len(groups["train"]), "val": len(groups["val"]),
                   "total": len(groups["train"]) + len(groups["val"])},
        "from": {"palette(pallets-*)": n_pal, "colis(colis-*)": n_col},
        "colis_originals_only": args.colis_originals_only,
        "preserve_colis_split": args.preserve_colis_split,
        "instances_per_class": {CLASSES[i]: cls_counts[i] for i in range(3)},
        "val_frac": args.val_frac, "seed": args.seed,
    }
    (args.out / "dataset.json").write_text(json.dumps(summary, indent=2))
    log.info("done — %s", json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
