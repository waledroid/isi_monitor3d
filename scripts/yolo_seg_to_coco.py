#!/usr/bin/env python3
"""Convert a YOLO-seg dataset → COCO format for RF-DETR-Seg.

RF-DETR's trainer wants ``train/ valid/ test/`` subdirs, each holding the images
+ an ``_annotations.coco.json``. This converts an Ultralytics YOLO-seg dataset
(``images/{train,val}`` + ``labels/{train,val}`` + ``data.yaml``) into that layout
so RF-DETR trains on the SAME split as YOLO (fair comparison, same leak-free
split). ``test/`` duplicates ``valid/`` (RF-DETR requires the folder; swap in a
real held-out set later if you want an independent test metric).

Categories follow the Roboflow/RF-DETR convention used by the colis export:
``id 0 = _background_``, then the dataset classes at ``id 1..N`` (so YOLO class k
→ COCO category k+1). Segmentations are denormalised to pixel polygons.

    python scripts/yolo_seg_to_coco.py \
      --src trainer/isidet/data/pallet3_yolo_seg \
      --out trainer/isidet/data/pallet3_coco --force
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml

try:
    from PIL import Image
except ImportError:                       # pragma: no cover
    Image = None
import cv2

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


def poly_area(xy: list[float]) -> float:
    """Shoelace area of a flat [x1,y1,...] polygon (pixels)."""
    n = len(xy) // 2
    a = 0.0
    for i in range(n):
        x1, y1 = xy[2 * i], xy[2 * i + 1]
        x2, y2 = xy[2 * ((i + 1) % n)], xy[2 * ((i + 1) % n) + 1]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0


def build_split(src_img_dir: Path, src_lbl_dir: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    images, annotations = [], []
    img_id = ann_id = 0
    for lf in sorted(src_lbl_dir.glob("*.txt")):
        if lf.name.endswith("Zone.Identifier") or lf.name == "classes.txt":
            continue
        img = next((src_img_dir / f"{lf.stem}{e}" for e in IMG_EXTS
                    if (src_img_dir / f"{lf.stem}{e}").exists()), None)
        if img is None:
            continue
        w, h = image_wh(img)
        img_id += 1
        shutil.copy2(img, out_dir / img.name)
        images.append({"id": img_id, "file_name": img.name, "width": w, "height": h})
        for line in lf.read_text().splitlines():
            p = line.split()
            if len(p) < 7:
                continue
            try:
                cls = int(float(p[0]))
                coords = [float(t) for t in p[1:]]
            except ValueError:
                continue
            if len(coords) % 2:
                continue
            seg = []
            xs, ys = [], []
            for i in range(0, len(coords), 2):
                px = min(max(coords[i] * w, 0.0), w)
                py = min(max(coords[i + 1] * h, 0.0), h)
                seg += [round(px, 2), round(py, 2)]
                xs.append(px)
                ys.append(py)
            if len(seg) < 6:
                continue
            x0, y0 = min(xs), min(ys)
            bw, bh = max(xs) - x0, max(ys) - y0
            ann_id += 1
            annotations.append({
                "id": ann_id, "image_id": img_id, "category_id": cls + 1,
                "segmentation": [seg], "area": poly_area(seg),
                "bbox": [round(x0, 2), round(y0, 2), round(bw, 2), round(bh, 2)],
                "iscrowd": 0, "ignore": 0,
            })
    return {"images": images, "annotations": annotations}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="YOLO-seg dataset root")
    ap.add_argument("--out", type=Path, required=True, help="COCO output root")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    names = yaml.safe_load((args.src / "data.yaml").read_text())["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    categories = [{"id": 0, "name": "_background_", "supercategory": None}]
    categories += [{"id": i + 1, "name": n, "supercategory": None} for i, n in enumerate(names)]

    if args.out.exists():
        if not args.force:
            print(f"error: {args.out} exists — pass --force", file=sys.stderr)
            return 2
        shutil.rmtree(args.out)

    # YOLO split → RF-DETR split (val → valid; test duplicates valid).
    plan = [("train", "train"), ("val", "valid"), ("val", "test")]
    summary = {}
    for yolo_split, coco_split in plan:
        img_dir = args.src / "images" / yolo_split
        lbl_dir = args.src / "labels" / yolo_split
        if not lbl_dir.is_dir():
            continue
        out_split = args.out / coco_split
        doc = build_split(img_dir, lbl_dir, out_split)
        doc["categories"] = categories
        (out_split / "_annotations.coco.json").write_text(json.dumps(doc))
        summary[coco_split] = {"images": len(doc["images"]),
                               "annotations": len(doc["annotations"])}
        print(f"{coco_split}: {summary[coco_split]['images']} imgs, "
              f"{summary[coco_split]['annotations']} anns")

    print(f"categories: {[c['name'] for c in categories]}")
    print(f"done → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
