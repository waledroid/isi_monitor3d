"""Zone-scoped detection — the Backbone's object detector sees ONLY the zones.

The system is zone-based: object detection (and therefore tracking, pallet
state, MQTT) applies inside the configured floor zones; the person-pose model
stays global (safety needs eyes everywhere). ``ZoneScopedDetector`` wraps any
``Detector`` plugin:

1. **Build time** — every floor zone polygon (metres, ``zones.yaml``) is
   projected into every camera through the calibration (distortion-aware,
   at ``z = 0`` and ``z = crop_height_m`` so a loaded pallet's full height
   fits) → one pixel crop box per (camera, zone), in CALIBRATION-frame
   pixels. Zones a camera can't see produce no box.
2. **Per pair** — each visible zone is cropped from the (possibly
   ingest-downscaled) frame, all crops ride ONE batched ``detect()`` call on
   the wrapped detector (mixed sizes are letterboxed per-crop, so a small far
   zone gets MORE model pixels than it would inside the full frame), and the
   detections are remapped to frame pixels. Downstream — geometry, ByteTrack,
   zone state, comms — is untouched.

Deliberately NOT a plugin seam: there is one sensible way to scope detection
to zones. The orchestrator composes it around whichever detector plugin the
config names (``detection.scope: zones`` — the default; ``full_frame``
restores the pre-zone-scope behaviour).

Masks (when ``decode_masks`` is on) stay CROP-RELATIVE through the remap,
carrying their crop origin in ``Detection.mask_offset_xy`` — mask area
consumers are offset-agnostic and the observations publisher polygonizes
them into frame coordinates for the wire.

Overlapping zones: a detection inside two overlapping crops is reported twice
(one per zone crop). Zones are physically disjoint in every deployment so no
dedup pass is spent on it; if that changes, dedupe here — not downstream.
"""

from __future__ import annotations

import logging
import math

import cv2
import numpy as np

from backbone.core.types import Detection, Frame, FramePair
from backbone.detection.enhance import enhance_bgr
from backbone.detection.tiling import merge_tiled, shift_detection, tile_boxes
from backbone.shared.geometry import (
    densify_polygon,
    floor_to_pixel,
    has_metric_camera_model,
)

logger = logging.getLogger(__name__)

# Synthetic camera-id separator for crop frames — must never appear in a real
# camera id (YAML keys are operator-typed identifiers like "cam_a").
_SEP = "\x00"
# Prefix for batch-padding frames (TensorRT bucketing) — never a real crop.
_PAD = "__pad__"


def _project_world3(world3, K, D, R, t, image_size_wh) -> np.ndarray:
    """Project 3D world points to RAW (distorted) pixels, pose convention as in
    :mod:`backbone.shared.geometry` (``R, t`` = camera pose, world←camera).

    Same divergence guard as ``floor_to_pixel_distorted``: the distortion
    polynomial explodes (or folds back inside the frame) for points far
    outside the calibrated field, so points whose PINHOLE projection is beyond
    a 25 % margin keep their pinhole coordinates. Behind-camera points → NaN.
    """
    world3 = np.asarray(world3, dtype=np.float64).reshape(-1, 3)
    K = np.asarray(K, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    R_cw = R.T
    t_cw = -R_cw @ t
    cam = (R_cw @ world3.T).T + t_cw
    in_front = cam[:, 2] > 1e-6

    out = np.full((len(world3), 2), np.nan)
    if not in_front.any():
        return out
    # Pinhole baseline (for the guard + the fallback value).
    pin = cam[in_front, :2] / cam[in_front, 2:3]
    pin = (K[:2, :2] @ pin.T).T + K[:2, 2]
    out[in_front] = pin

    w, h = float(image_size_wh[0]), float(image_size_wh[1])
    mx, my = w * 0.25, h * 0.25
    near = np.zeros(len(world3), dtype=bool)
    near[in_front] = ((pin[:, 0] >= -mx) & (pin[:, 0] < w + mx)
                      & (pin[:, 1] >= -my) & (pin[:, 1] < h + my))
    if near.any():
        rvec, _ = cv2.Rodrigues(R_cw)
        duv, _ = cv2.projectPoints(world3[near], rvec, t_cw, K,
                                   np.asarray(D, dtype=np.float64))
        out[near] = duv.reshape(-1, 2)
    return out


def zone_crop_boxes(
    rig,
    zones,
    *,
    crop_height_m: float = 2.0,
    margin_px: int = 16,
    min_side_px: int = 48,
) -> dict[str, list[tuple[str, tuple[int, int, int, int]]]]:
    """Per-camera crop boxes, CALIBRATION-frame pixels: ``{cam: [(zone, box)]}``.

    The box is the union bbox of the zone polygon projected at the floor
    (``z=0``) and at ``crop_height_m`` (objects have height — a loaded pallet
    extends well above its floor footprint in the image), padded by
    ``margin_px`` and clipped to the frame. Zones whose visible box is smaller
    than ``min_side_px`` on either side (barely/not visible) are skipped.

    Mode-1 placeholder extrinsics (only ``H`` is real) can't lift the polygon
    off the floor: the floor bbox is extended UPWARD by its own height as a
    conservative stand-in.
    """
    boxes: dict[str, list[tuple[str, tuple[int, int, int, int]]]] = {}
    for cam_id in rig.camera_ids:
        view = rig[cam_id]
        w, h = int(view.image_size_wh[0]), int(view.image_size_wh[1])
        cam_boxes = []
        for name in zones.names:
            poly = densify_polygon(zones[name].polygon, segments_per_edge=8)
            if has_metric_camera_model(view.K, view.R, view.t):
                floor3 = np.hstack([poly, np.zeros((len(poly), 1))])
                top3 = np.hstack([poly, np.full((len(poly), 1), float(crop_height_m))])
                uv = np.vstack([
                    _project_world3(floor3, view.K, view.D, view.R, view.t,
                                    view.image_size_wh),
                    _project_world3(top3, view.K, view.D, view.R, view.t,
                                    view.image_size_wh),
                ])
                uv = uv[~np.isnan(uv).any(axis=1)]
            else:
                uv = floor_to_pixel(poly, view.H)
            if len(uv) == 0:
                continue
            x0 = math.floor(uv[:, 0].min()) - margin_px
            x1 = math.ceil(uv[:, 0].max()) + margin_px
            y0 = math.floor(uv[:, 1].min()) - margin_px
            y1 = math.ceil(uv[:, 1].max()) + margin_px
            if not has_metric_camera_model(view.K, view.R, view.t):
                y0 -= (y1 - y0)          # Mode 1: height stand-in (see docstring)
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w, x1), min(h, y1)
            if (x1 - x0) < min_side_px or (y1 - y0) < min_side_px:
                continue
            cam_boxes.append((name, (int(x0), int(y0), int(x1), int(y1))))
        boxes[cam_id] = cam_boxes
        logger.info("zone_scope: %s sees %d/%d zones: %s", cam_id,
                    len(cam_boxes), len(zones), [b[0] for b in cam_boxes])
    return boxes


class ZoneScopedDetector:
    """Wrap a ``Detector`` so it runs on zone crops only (see module docstring).

    ``calib_wh_by_camera`` is each camera's calibration frame size — crop boxes
    are calibration-frame pixels, while live frames may be ingest-downscaled
    (source ``output_wh``), so crops scale per frame.
    """

    def __init__(self, detector, boxes_by_camera,
                 calib_wh_by_camera: dict[str, tuple[int, int]],
                 *,
                 sahi: dict | None = None,
                 enhance: dict | None = None,
                 batch_buckets: tuple[int, ...] | None = None) -> None:
        self._detector = detector
        self._boxes = boxes_by_camera
        self._calib_wh = {k: (int(v[0]), int(v[1]))
                          for k, v in calib_wh_by_camera.items()}
        # SAHI: slice big zone crops into overlapping tiles so far/small
        # objects keep their pixels. All tiles ride the SAME batched call.
        sahi = sahi or {}
        self._sahi_on = bool(sahi.get("enabled", False))
        self._tile = int(sahi.get("tile", 0) or 0)      # 0 → the model input size
        self._overlap = float(sahi.get("overlap", 0.2))
        self._merge_iou = float(sahi.get("merge_iou", 0.5))
        # ENH: CLAHE (+ gamma) on each crop/tile before letterboxing.
        enhance = enhance or {}
        self._enh_on = bool(enhance.get("enabled", False))
        self._enh_kwargs = {
            "clip_limit": float(enhance.get("clip_limit", 2.0)),
            "tile_grid": int(enhance.get("tile_grid", 8)),
            "gamma": float(enhance.get("gamma", 1.0)),
        }
        # TensorRT compiles ONE ENGINE PER INPUT SHAPE. A batch that changes
        # with the number of visible zones / motion-gated cameras / SAHI tiles
        # would trigger a multi-minute engine build at every new count. Pad the
        # batch up to the next bucket instead: a handful of engines, built once
        # and cached. (The padding frames are duplicates; their outputs are
        # discarded.) None ⇒ no padding (CUDA EP handles any batch).
        self._buckets = tuple(sorted(batch_buckets)) if batch_buckets else None

    def detect(self, pair: FramePair) -> dict[str, list[Detection]]:
        out: dict[str, list[Detection]] = {cid: [] for cid in pair.frames}
        crops: dict[str, Frame] = {}
        origin: dict[str, tuple[str, int, int]] = {}
        for cam_id, frame in pair.frames.items():
            cam_boxes = self._boxes.get(cam_id) or []
            if not cam_boxes:
                continue
            fh, fw = frame.image.shape[:2]
            # Crop boxes are calibration-frame px; frames may be ingest-downscaled.
            calib_w, calib_h = self._calib_wh.get(cam_id, (fw, fh))
            sx, sy = fw / calib_w, fh / calib_h
            for i, (_zone, (x0, y0, x1, y1)) in enumerate(cam_boxes):
                fx0, fy0 = max(0, int(x0 * sx)), max(0, int(y0 * sy))
                fx1, fy1 = min(fw, math.ceil(x1 * sx)), min(fh, math.ceil(y1 * sy))
                if fx1 - fx0 < 8 or fy1 - fy0 < 8:
                    continue
                crop_img = frame.image[fy0:fy1, fx0:fx1]
                if self._enh_on:
                    crop_img = enhance_bgr(crop_img, **self._enh_kwargs)
                ch, cw = crop_img.shape[:2]
                tile = self._tile or self._model_input_px()
                rects = (tile_boxes(cw, ch, tile, self._overlap)
                         if self._sahi_on else [(0, 0, cw, ch)])
                for t, (tx0, ty0, tx1, ty1) in enumerate(rects):
                    sid = f"{cam_id}{_SEP}{i}{_SEP}{t}"
                    crops[sid] = Frame(camera_id=sid, capture_ts=frame.capture_ts,
                                       frame_idx=frame.frame_idx,
                                       image=crop_img[ty0:ty1, tx0:tx1])
                    # Origin folds the zone-crop offset AND the tile offset, so
                    # a detection maps straight back to frame coordinates.
                    origin[sid] = (cam_id, fx0 + tx0, fy0 + ty0)
        if not crops:
            return out
        raw = self._detect_padded(pair, crops)
        # Tiled zones: merge each zone's tiles before remapping, so an object
        # split across two tiles becomes one detection with the union box.
        if self._sahi_on:
            raw = self._merge_zone_tiles(raw, origin)
        for sid, dets in raw.items():
            cam_id, ox, oy = origin[sid]
            for d in dets:
                d.camera_id = cam_id
                x0, y0, x1, y1 = d.bbox_xyxy
                d.bbox_xyxy = (x0 + ox, y0 + oy, x1 + ox, y1 + oy)
                u, v = d.foot_uv
                d.foot_uv = (u + ox, v + oy)
                if d.keypoints_uv is not None:
                    kp = np.asarray(d.keypoints_uv, dtype=np.float64).copy()
                    kp[:, 0] += ox
                    kp[:, 1] += oy
                    d.keypoints_uv = kp
                if d.mask is not None:
                    # Keep the mask CROP-RELATIVE + record the crop origin —
                    # tiny memory, and downstream (occupancy area, the wire's
                    # mask→polygon) handles the offset. Never allocate a
                    # full-frame canvas per detection.
                    d.mask_offset_xy = (ox, oy)
                out[cam_id].append(d)
        return out

    # ---- internals ----

    def _model_input_px(self) -> int:
        size = getattr(self._detector, "input_size", None)
        if isinstance(size, (tuple, list)) and size:
            return int(size[0])
        return 384

    def _detect_padded(self, pair: FramePair, crops: dict[str, Frame]) -> dict:
        """Run one batched inference, padding the batch to a bucket when the
        backend compiles per-shape engines (TensorRT)."""
        if not self._buckets:
            return self._detector.detect(FramePair(
                capture_ts=pair.capture_ts, frame_idx=pair.frame_idx, frames=crops))
        n = len(crops)
        target = next((b for b in self._buckets if b >= n), None)
        padded = dict(crops)
        if target is not None and target > n:
            filler = next(iter(crops.values()))
            for k in range(target - n):
                padded[f"{_PAD}{k}"] = filler
        elif target is None:
            logger.debug("zone_scope: batch %d exceeds largest bucket %s",
                         n, self._buckets[-1])
        raw = self._detector.detect(FramePair(
            capture_ts=pair.capture_ts, frame_idx=pair.frame_idx, frames=padded))
        return {sid: dets for sid, dets in raw.items() if not sid.startswith(_PAD)}

    def _merge_zone_tiles(self, raw: dict, origin: dict) -> dict:
        """Union-merge the tiles of each zone crop (SAHI). Detections are
        translated into ZONE-CROP coordinates for merging, then re-expressed
        against a synthetic origin so the caller's remap stays unchanged."""
        by_zone: dict[tuple[str, str], list] = {}
        for sid, dets in raw.items():
            cam_id, zone_idx, _tile = sid.split(_SEP)
            by_zone.setdefault((cam_id, zone_idx), []).append((sid, dets))
        merged: dict[str, list] = {}
        for (cam_id, zone_idx), entries in by_zone.items():
            # Anchor: the zone crop's own origin = min tile origin.
            ox0 = min(origin[sid][1] for sid, _ in entries)
            oy0 = min(origin[sid][2] for sid, _ in entries)
            pooled: list = []
            for sid, dets in entries:
                _c, tx, ty = origin[sid]
                for d in dets:
                    pooled.append(shift_detection(d, tx - ox0, ty - oy0))
            anchor = f"{cam_id}{_SEP}{zone_idx}{_SEP}0"
            merged[anchor] = merge_tiled(pooled, iou_thresh=self._merge_iou)
            origin[anchor] = (cam_id, ox0, oy0)
        return merged
