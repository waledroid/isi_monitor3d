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
    x0 -= bw * rng.uniform(lo, hi)
    x1 += bw * rng.uniform(lo, hi)
    y0 -= bh * rng.uniform(lo, hi)
    y1 += bh * rng.uniform(lo, hi)
    side = max(x1 - x0, y1 - y0, float(size))
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    wx0 = int(round(cx - side / 2.0))
    wy0 = int(round(cy - side / 2.0))
    wx1 = wx0 + int(round(side))
    wy1 = wy0 + int(round(side))
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
