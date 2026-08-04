"""Derive an object-centric cropped YOLO-seg dataset from pallet3_yolo_seg.

Crops square windows around clusters of GROUND-TRUTH objects (never a
model's detections), remaps the segmentation polygons into each crop, and
writes a new dataset sized for the isimonitor3d nano zone-inference domain.
The source dataset is never modified.

Rules (see docs/superpowers/specs/2026-08-04-crop-dataset-design.md):
- crop size --size (default 384); content is never upscaled (gray-114 pad);
- an object with < --keep-frac of its area inside a crop is NOT labeled —
  its in-crop pixels are painted gray 114 (like the inference polygon fill);
- split-preserving (train->train, val->val); backgrounds fold in 90/10.

Usage:
  conda activate monitor3d
  python tools/make_crop_dataset.py \\
      --src trainer/isidet/data/pallet3_yolo_seg \\
      --out trainer/isidet/data/pallet3_yolo_seg_crop384 \\
      [--size 384] [--backgrounds trainer/isidet/data/grouped_backgrounds] \\
      [--margin 0.10 0.25] [--keep-frac 0.30] [--seed 0] [--preview N]
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

logger = logging.getLogger("make_crop_dataset")

GRAY = 114


# ---- label I/O ----

def parse_label_file(path: Path, img_w: int, img_h: int):
    """YOLO-seg label file -> [(cls, poly_px (N,2) float64)]; bad lines skipped."""
    objs = []
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        try:
            cls = int(parts[0])
            vals = np.array([float(v) for v in parts[1:]], dtype=np.float64)
            if len(vals) < 6 or len(vals) % 2:
                raise ValueError("need >=3 x,y pairs")
        except (ValueError, IndexError) as exc:
            logger.warning("%s:%d: skipping malformed label line (%s)",
                           path, lineno, exc)
            continue
        poly = vals.reshape(-1, 2) * (img_w, img_h)
        objs.append((cls, poly))
    return objs


def format_label_lines(objs) -> str:
    """[(cls, poly_norm (N,2) in [0,1])] -> YOLO-seg text (clamped, 6dp)."""
    lines = []
    for cls, poly in objs:
        flat = np.clip(np.asarray(poly, dtype=np.float64), 0.0, 1.0).flatten()
        lines.append(str(cls) + " " + " ".join(f"{v:.6f}" for v in flat))
    return "".join(line + "\n" for line in lines)


# ---- polygon geometry ----

def poly_bbox(poly):
    return (float(poly[:, 0].min()), float(poly[:, 1].min()),
            float(poly[:, 0].max()), float(poly[:, 1].max()))


def poly_area(poly) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def clip_polygon(poly, x0, y0, x1, y1):
    """Sutherland-Hodgman clip vs axis-aligned rect; None when empty."""
    def ix_v(a, b, x):
        t = (x - a[0]) / (b[0] - a[0])
        return (x, a[1] + t * (b[1] - a[1]))

    def ix_h(a, b, y):
        t = (y - a[1]) / (b[1] - a[1])
        return (a[0] + t * (b[0] - a[0]), y)

    edges = (
        (lambda p: p[0] >= x0, lambda a, b: ix_v(a, b, x0)),
        (lambda p: p[0] <= x1, lambda a, b: ix_v(a, b, x1)),
        (lambda p: p[1] >= y0, lambda a, b: ix_h(a, b, y0)),
        (lambda p: p[1] <= y1, lambda a, b: ix_h(a, b, y1)),
    )
    pts = [tuple(p) for p in np.asarray(poly, dtype=np.float64)]
    for inside, intersect in edges:
        nxt = []
        n = len(pts)
        for i in range(n):
            a, b = pts[i], pts[(i + 1) % n]
            if inside(a):
                nxt.append(a)
                if not inside(b):
                    nxt.append(intersect(a, b))
            elif inside(b):
                nxt.append(intersect(a, b))
        pts = nxt
        if not pts:
            return None
    return np.array(pts, dtype=np.float64)


# ---- clustering & windows ----

def cluster_boxes(boxes, expand_frac: float):
    """Union-find over boxes whose expanded rects intersect."""
    n = len(boxes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    exp = []
    for x0, y0, x1, y1 in boxes:
        mx, my = (x1 - x0) * expand_frac, (y1 - y0) * expand_frac
        exp.append((x0 - mx, y0 - my, x1 + mx, y1 + my))
    for i in range(n):
        for j in range(i + 1, n):
            a, b = exp[i], exp[j]
            if a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]:
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return sorted((sorted(g) for g in groups.values()), key=lambda g: g[0])


def crop_window(bbox, img_wh, size, margin_range, rng):
    """Square window >= size px covering bbox+margin, clamped to the image."""
    iw, ih = img_wh
    x0, y0, x1, y1 = bbox
    bw, bh = x1 - x0, y1 - y0
    lo, hi = margin_range
    x0 -= bw * float(rng.uniform(lo, hi))
    x1 += bw * float(rng.uniform(lo, hi))
    y0 -= bh * float(rng.uniform(lo, hi))
    y1 += bh * float(rng.uniform(lo, hi))
    side = max(x1 - x0, y1 - y0, float(size))
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    wx0 = round(cx - side / 2.0)
    wy0 = round(cy - side / 2.0)
    wx1 = wx0 + round(side)
    wy1 = wy0 + round(side)
    if wx0 < 0:
        wx1 -= wx0
        wx0 = 0
    if wy0 < 0:
        wy1 -= wy0
        wy0 = 0
    if wx1 > iw:
        wx0 -= wx1 - iw
        wx1 = iw
    if wy1 > ih:
        wy0 -= wy1 - ih
        wy1 = ih
    return max(0, wx0), max(0, wy0), wx1, wy1


def letterbox_to(img, size):
    """Fit img into size x size with gray padding; NEVER upscale content."""
    h, w = img.shape[:2]
    scale = min(size / w, size / h, 1.0)
    nw, nh = round(w * scale), round(h * scale)
    if (nw, nh) != (w, h):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), GRAY, np.uint8)
    dx, dy = (size - nw) // 2, (size - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw] = img
    return canvas, float(scale), dx, dy


# ---- per-image pipeline ----

def generate_crops(img, objs, *, size, margin_range, keep_frac, rng):
    """One (crop, labels) per GT cluster; keep-frac rule gray-fills partials.

    An image with no objects becomes a single letterboxed full-frame
    background crop (empty labels).
    """
    h, w = img.shape[:2]
    if not objs:
        canvas, _, _, _ = letterbox_to(img, size)
        return [(canvas, [])]
    boxes = [poly_bbox(p) for _, p in objs]
    out = []
    for group in cluster_boxes(boxes, expand_frac=margin_range[1]):
        gx0 = min(boxes[i][0] for i in group)
        gy0 = min(boxes[i][1] for i in group)
        gx1 = max(boxes[i][2] for i in group)
        gy1 = max(boxes[i][3] for i in group)
        wx0, wy0, wx1, wy1 = crop_window((gx0, gy0, gx1, gy1), (w, h),
                                         size, margin_range, rng)
        cw, ch = wx1 - wx0, wy1 - wy0
        if cw < 8 or ch < 8:
            continue
        crop = img[wy0:wy1, wx0:wx1].copy()
        kept, fills = [], []
        for cls, poly in objs:
            clipped = clip_polygon(poly - (wx0, wy0), 0, 0, cw, ch)
            if clipped is None or len(clipped) < 3:
                continue
            area = poly_area(clipped)
            if area < 1.0:
                fills.append(clipped)
                continue
            if area / max(poly_area(poly), 1e-9) >= keep_frac:
                kept.append((cls, clipped))
            else:
                fills.append(clipped)
        for f in fills:
            cv2.fillPoly(crop, [np.round(f).astype(np.int32)],
                         (GRAY, GRAY, GRAY))
        canvas, scale, dx, dy = letterbox_to(crop, size)
        labels = [(cls, (p * scale + (dx, dy)) / size) for cls, p in kept]
        out.append((canvas, labels))
    return out


# ---- CLI ----

def bg_split(name: str) -> str:
    """Deterministic 90/10 background split by capture-series prefix, so
    near-duplicate frames from one (cam, zone) series stay in ONE split."""
    group = re.sub(r"_\d+\.[A-Za-z]+$", "", name)
    return "val" if int(hashlib.md5(group.encode()).hexdigest(), 16) % 10 == 0 \
        else "train"


def _iter_images(split_dir: Path):
    return sorted(p for p in split_dir.glob("*")
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Derive an object-centric cropped YOLO-seg dataset from "
                    "GT labels. See module docstring.")
    ap.add_argument("--src", required=True, help="source dataset root")
    ap.add_argument("--out", required=True, help="output dataset root")
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--backgrounds", default=None,
                    help="folder of label-free background images to fold in")
    ap.add_argument("--margin", type=float, nargs=2, default=(0.10, 0.25),
                    metavar=("LO", "HI"))
    ap.add_argument("--keep-frac", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preview", type=int, default=0, metavar="N",
                    help="write N annotated sample crops to <out>/_preview "
                         "and exit (no dataset generated)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    src, out = Path(args.src), Path(args.out)
    if out.exists() and any(p.name != "_preview" for p in out.iterdir()):
        logger.error("refusing: %s exists and is not empty", out)
        return 2
    rng = np.random.default_rng(args.seed)

    pairs = []          # (split, img_path, label_path)
    for split in ("train", "val"):
        for img_path in _iter_images(src / "images" / split):
            pairs.append((split, img_path,
                          src / "labels" / split / (img_path.stem + ".txt")))

    if not pairs:
        logger.error("no images found under %s/images/{train,val}", src)
        return 2

    if args.preview:
        pv = out / "_preview"
        pv.mkdir(parents=True, exist_ok=True)
        idxs = rng.choice(len(pairs), size=min(args.preview, len(pairs)),
                          replace=False)
        written = 0
        for i in idxs:
            split, img_path, lbl_path = pairs[int(i)]
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            objs = (parse_label_file(lbl_path, img.shape[1], img.shape[0])
                    if lbl_path.exists() else [])
            for k, (crop, labels) in enumerate(generate_crops(
                    img, objs, size=args.size, margin_range=tuple(args.margin),
                    keep_frac=args.keep_frac, rng=rng)):
                for cls, poly in labels:
                    pts = np.round(poly * args.size).astype(np.int32)
                    cv2.polylines(crop, [pts], True, (0, 255, 0), 2)
                    cv2.putText(crop, str(cls), tuple(pts[0]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imwrite(str(pv / f"{img_path.stem}_c{k}.jpg"), crop)
                written += 1
                if written >= args.preview:
                    break
            if written >= args.preview:
                break
        logger.info("preview: %d annotated crop(s) in %s", written, pv)
        return 0

    stats = {"images": 0, "unreadable": 0, "crops": 0, "labels": 0,
             "grayfilled_or_dropped": 0, "backgrounds": 0}
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    for split, img_path, lbl_path in pairs:
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning("unreadable image skipped: %s", img_path)
            stats["unreadable"] += 1
            continue
        stats["images"] += 1
        objs = (parse_label_file(lbl_path, img.shape[1], img.shape[0])
                if lbl_path.exists() else [])
        n_src = len(objs)
        n_kept = 0
        for k, (crop, labels) in enumerate(generate_crops(
                img, objs, size=args.size, margin_range=tuple(args.margin),
                keep_frac=args.keep_frac, rng=rng)):
            name = f"{img_path.stem}_c{k}"
            cv2.imwrite(str(out / "images" / split / (name + ".jpg")), crop,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            if labels:
                (out / "labels" / split / (name + ".txt")).write_text(
                    format_label_lines(labels))
            stats["crops"] += 1
            stats["labels"] += len(labels)
            n_kept += len(labels)
        stats["grayfilled_or_dropped"] += max(0, n_src - n_kept)

    if args.backgrounds:
        for p in _iter_images(Path(args.backgrounds)):
            img = cv2.imread(str(p))
            if img is None:
                logger.warning("unreadable background skipped: %s", p)
                continue
            target = out / "images" / bg_split(p.name) / (p.stem + ".jpg")
            if target.exists():
                logger.warning("background name collides with an existing "
                               "crop, skipped: %s", target)
                continue
            canvas, _, _, _ = letterbox_to(img, args.size)
            cv2.imwrite(str(target), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
            stats["backgrounds"] += 1

    names = ["palette", "carton", "polybag"]
    src_yaml = src / "data.yaml"
    if src_yaml.exists():
        loaded = yaml.safe_load(src_yaml.read_text()) or {}
        names = list(loaded.get("names", names))
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\n"
        f"nc: {len(names)}\nnames: {names}\n")

    for split in ("train", "val"):
        n_img = len(list((out / "images" / split).glob("*.jpg")))
        n_lbl = len(list((out / "labels" / split).glob("*.txt")))
        logger.info("%s: %d images (%d labeled, %d backgrounds)",
                    split, n_img, n_lbl, n_img - n_lbl)
    logger.info("summary: %s", stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
