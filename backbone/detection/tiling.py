"""SAHI-style tiling + tile-merge for zone crops.

A zone crop that is much larger than the model's input gets letterboxed down,
which shrinks small objects (a carton at the far end of an aisle) below the
detector's useful scale. SAHI ("slicing aided hyper inference") instead cuts
the crop into overlapping tiles at roughly the model's native scale, runs the
model on every tile, and merges the results back into crop coordinates.

Cost is paid in BATCH, not in extra calls: every tile of every zone of every
camera goes into ONE inference (see ``ZoneScopedDetector``), which is exactly
what a batched TensorRT engine wants. The batch-bucketing in that module keeps
the tile count from exploding the set of compiled engine shapes.

Merging: tiles overlap, so one physical object can appear in several tiles.
``merge_tiled`` unifies same-class detections that describe one object —
IoU above threshold, or one centre inside the other, or intersection covering
most of the smaller box. The rule is deliberately CONTAINMENT-based rather
than "adjacent boxes stitch together": two pallets standing side by side must
never fuse into one.

**Overlap constraint** (SAHI's own rule, and what makes the merge sound): the
tile overlap must exceed the largest expected object, so at least one tile
sees every object WHOLE. The clipped copies from neighbouring tiles are then
contained in that whole box and get absorbed. With too little overlap, a big
object crossing a seam is only ever seen in halves — and this merger will
(correctly, conservatively) leave them as two detections.
"""

from __future__ import annotations

import numpy as np


def tile_boxes(w: int, h: int, tile: int, overlap: float) -> list[tuple[int, int, int, int]]:
    """Overlapping tile rectangles covering a ``w x h`` crop.

    Returns ``[(x0, y0, x1, y1), ...]``; a crop at or below ``tile`` in both
    dimensions yields a single full-crop tile (i.e. SAHI is a no-op there).
    """
    tile = max(32, int(tile))
    if w <= tile and h <= tile:
        return [(0, 0, w, h)]
    step = max(16, int(tile * (1.0 - min(0.9, max(0.0, overlap)))))
    xs = list(range(0, max(1, w - tile + 1), step)) or [0]
    ys = list(range(0, max(1, h - tile + 1), step)) or [0]
    if xs[-1] + tile < w:
        xs.append(max(0, w - tile))
    if ys[-1] + tile < h:
        ys.append(max(0, h - tile))
    return [(x, y, min(w, x + tile), min(h, y + tile)) for y in ys for x in xs]


def _iou(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _same_object(a, b, iou_thresh: float) -> bool:
    """One physical object seen in two tiles — robust to clipping."""
    if _iou(a, b) > iou_thresh:
        return True
    acx, acy = (a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0
    bcx, bcy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
    if (b[0] <= acx <= b[2] and b[1] <= acy <= b[3]) or (
            a[0] <= bcx <= a[2] and a[1] <= bcy <= a[3]):
        return True
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    smaller = min(area_a, area_b)
    return smaller > 0.0 and inter / smaller > 0.6


def merge_tiled(dets: list, *, iou_thresh: float = 0.5) -> list:
    """Union-merge detections of the same class coming from overlapping tiles.

    Keeps the highest-confidence member's class/mask and takes the UNION box
    (a clipped half plus its other half is the whole object). Detections are
    expected in CROP coordinates already.
    """
    if len(dets) < 2:
        return list(dets)
    order = sorted(dets, key=lambda d: -float(d.confidence))
    kept: list = []
    for d in order:
        merged = False
        for k in kept:
            if str(k.cls) != str(d.cls):
                continue
            if _same_object(k.bbox_xyxy, d.bbox_xyxy, iou_thresh):
                x0 = min(k.bbox_xyxy[0], d.bbox_xyxy[0])
                y0 = min(k.bbox_xyxy[1], d.bbox_xyxy[1])
                x1 = max(k.bbox_xyxy[2], d.bbox_xyxy[2])
                y1 = max(k.bbox_xyxy[3], d.bbox_xyxy[3])
                k.bbox_xyxy = (x0, y0, x1, y1)
                k.foot_uv = ((x0 + x1) / 2.0, y1)
                merged = True
                break
        if not merged:
            kept.append(d)
    return kept


def shift_detection(d, dx: int, dy: int):
    """Translate a detection (box, foot, keypoints, mask origin) in place."""
    x0, y0, x1, y1 = d.bbox_xyxy
    d.bbox_xyxy = (x0 + dx, y0 + dy, x1 + dx, y1 + dy)
    u, v = d.foot_uv
    d.foot_uv = (u + dx, v + dy)
    if d.keypoints_uv is not None:
        kp = np.asarray(d.keypoints_uv, dtype=np.float64).copy()
        kp[:, 0] += dx
        kp[:, 1] += dy
        d.keypoints_uv = kp
    if getattr(d, "mask_offset_xy", None) is not None:
        ox, oy = d.mask_offset_xy
        d.mask_offset_xy = (ox + dx, oy + dy)
    elif d.mask is not None:
        d.mask_offset_xy = (dx, dy)
    return d
