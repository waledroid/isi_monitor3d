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
    epsilon = 1e-4  # Push near-1.0 values over the rounding edge
    for cls, poly in objs:
        flat = np.asarray(poly, dtype=np.float64)
        flat = np.clip(flat, 0.0, 1.0)  # Clamp to [0, 1]
        # Add epsilon to values very close to 1.0 to ensure rounding to 1.0
        flat = np.where(flat > (1.0 - epsilon), flat + epsilon, flat)
        flat = np.round(flat, 6)  # Round to 6 decimals
        flat = np.clip(flat, 0.0, 1.0)  # Clamp again to [0, 1]
        flat = flat.flatten()
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
