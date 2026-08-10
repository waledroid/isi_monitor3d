"""Zone-scoped detection — the Backbone's object detector sees ONLY the zones.

The system is zone-based: object detection (and therefore tracking, pallet
state, MQTT) applies inside the configured floor zones; the person-pose model
stays global (safety needs eyes everywhere). ``ZoneScopedDetector`` wraps any
``Detector`` plugin:

1. **Build time** — every floor zone polygon (metres, ``zones.yaml``) is
   projected into every camera through the calibration (distortion-aware) →
   the tight rectangle around the projected polygon corners, one pixel crop
   box per (camera, zone), in CALIBRATION-frame pixels. An optional
   ``crop_height_m`` headroom additionally projects the polygon at that
   height for sites with tall loads. Zones a camera can't see produce no box.
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

Overlapping CROPS: zones are physically disjoint, but their crops (with
margin, or with a nonzero ``crop_height_m``) can still overlap, so one
physical object may be detected once per crop (per-crop NMS cannot see
across crops; the cam view drew two boxes on one palette). ``detect()``
therefore MERGES each camera's overlapping same-class detections into one
(union bbox + union mask, max confidence) and drops object detections whose
metric foot lies outside every zone polygon (``zone_filter``).
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
# A zone crop whose long side exceeds this multiple of its short side is
# square-tiled even without global SAHI — see the aspect note in detect().
_MAX_CROP_ASPECT = 2.0
# Polygon fill: crop pixels outside the projected zone polygon (dilated by
# this many metres, converted at the zone's local pixel scale) are blanked to
# neutral gray before inference, so the detector can't see off-zone objects
# the rectangular crop necessarily includes. The dilation keeps an object
# straddling the zone boundary whole.
_FILL_DILATE_M = 0.30
_FILL_GRAY = 114  # YOLO letterbox gray


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


def build_zone_membership_filter(rig, zones, tol_m: float = 0.15):
    """A ``filter(cam_id, foot_uv_calibration_px) -> bool`` closure: True when
    the detection's metric point lies inside ANY zone polygon (± ``tol_m``),
    tested on EACH ZONE'S OWN PLANE (``Zone.z_base_m`` via
    :class:`~backbone.shared.zones.ZoneAwareProjector`): floor zones keep the
    exact undistort+H floor path; a raised zone (a platform, a shelf) tests
    the ray/plane intersection at its own height, so an on-platform object
    whose FLOOR projection overshoots the polygon is still kept.

    The semantic guarantee behind zone-scoped detection: a zone never reports
    an object that is not metrically inside a zone, no matter how far its
    (rectangular, height-extruded) pixel crop had to reach. Tolerance is
    sampled as a 5-point cross (center ± tol on each axis) so an object
    straddling the boundary by projection error is kept.
    """
    from backbone.shared.zones import ZoneAwareProjector

    projector = ZoneAwareProjector(rig)
    zone_list = [zones[name] for name in zones.names]
    offsets = ((0.0, 0.0), (tol_m, 0.0), (-tol_m, 0.0),
               (0.0, tol_m), (0.0, -tol_m))

    def _filter(cam_id: str, foot_uv) -> bool:
        if cam_id not in projector:
            return True                       # unknown camera: never drop
        try:
            # One projection per distinct plane height (all-floor configs
            # project exactly once, as before).
            plane_xy: dict[float, tuple[float, float] | None] = {}
            degenerate = False
            for zone in zone_list:
                z = float(zone.z_base_m)
                if z not in plane_xy:
                    plane_xy[z] = projector.position_on_plane(cam_id, foot_uv, z)
                xy = plane_xy[z]
                if xy is None:
                    degenerate = True         # fail open — never lose a det to NaN
                    continue
                for dx, dy in offsets:
                    if zone.contains((float(xy[0] + dx), float(xy[1] + dy))):
                        return True
            return degenerate
        except Exception:
            return True                       # fail open on any geometry error
    return _filter


def _clipped_same(a: Detection, b: Detection) -> bool:
    """One object seen as offset PARTIAL boxes by two overlapping crops.

    Offset clips overlap only ~50%, evading the generic same-object tests
    (2026-07-22 live case: Sortie_1's full-height strip vs Sortie_2's crop cut
    at y=679 put two boxes on one pallet). The rule: a box lying ON its own
    crop edge, with the other box extending past that edge and a real overlap
    between them (≥30% of the smaller box), is a cut-off view of the same
    object."""
    eps = 2.0
    ax0, ay0, ax1, ay1 = a.bbox_xyxy
    bx0, by0, bx1, by1 = b.bbox_xyxy
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    smaller = min((ax1 - ax0) * (ay1 - ay0), (bx1 - bx0) * (by1 - by0))
    if smaller <= 0.0 or (ix * iy) / smaller < 0.3:
        return False
    for (x0, y0, x1, y1), other, crop in (
            (a.bbox_xyxy, b.bbox_xyxy, getattr(a, "crop_xyxy", None)),
            (b.bbox_xyxy, a.bbox_xyxy, getattr(b, "crop_xyxy", None))):
        if crop is None:
            continue
        cx0, cy0, cx1, cy1 = crop
        if ((abs(y1 - cy1) <= eps and other[3] > y1 + eps)
                or (abs(y0 - cy0) <= eps and other[1] < y0 - eps)
                or (abs(x1 - cx1) <= eps and other[2] > x1 + eps)
                or (abs(x0 - cx0) <= eps and other[0] < x0 - eps)):
            return True
    return False


def _iou_xyxy(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-9)


def _merge_two(a: Detection, b: Detection) -> Detection:
    """One detection from two views of the same object: UNION box (recovers
    the true extent two crop-clipped partials each lost), max confidence,
    union-composed mask, foot at the union's bottom-center."""
    ax0, ay0, ax1, ay1 = a.bbox_xyxy
    bx0, by0, bx1, by1 = b.bbox_xyxy
    ux0, uy0 = min(ax0, bx0), min(ay0, by0)
    ux1, uy1 = max(ax1, bx1), max(ay1, by1)
    lead = a if float(a.confidence) >= float(b.confidence) else b

    mask, offset = lead.mask, getattr(lead, "mask_offset_xy", None)
    if a.mask is not None and b.mask is not None:
        exts = []
        for d in (a, b):
            ox, oy = getattr(d, "mask_offset_xy", None) or (0, 0)
            mh, mw = d.mask.shape[:2]
            exts.append((int(ox), int(oy), int(ox) + mw, int(oy) + mh))
        cx0 = min(e[0] for e in exts)
        cy0 = min(e[1] for e in exts)
        cx1 = max(e[2] for e in exts)
        cy1 = max(e[3] for e in exts)
        canvas = np.zeros((cy1 - cy0, cx1 - cx0), dtype=bool)
        for d, (ox, oy, ex, ey) in zip((a, b), exts, strict=True):
            canvas[oy - cy0:ey - cy0, ox - cx0:ex - cx0] |= d.mask.astype(bool)
        mask, offset = canvas, (cx0, cy0)

    return Detection(
        camera_id=lead.camera_id, capture_ts=lead.capture_ts, cls=lead.cls,
        confidence=max(float(a.confidence), float(b.confidence)),
        bbox_xyxy=(ux0, uy0, ux1, uy1),
        foot_uv=((ux0 + ux1) / 2.0, uy1),
        keypoints_uv=lead.keypoints_uv,
        mask=mask, mask_offset_xy=offset,
        crop_xyxy=None,          # merged view spans crops — clip rule off
    )


def _dedup_across_crops(dets: list[Detection]) -> list[Detection]:
    """One detection per physical object per camera: MERGE same-class
    duplicates from overlapping zone crops into a union detection (box, mask,
    max confidence) — a class appears once per physical object with its true
    extent; non-overlapping same-class boxes are distinct objects and are
    never touched. Merge triggers: the generic same-object geometry, the
    crop-edge clip rule (`_clipped_same`), or plain IoU ≥ 0.10 (author rule:
    same-class overlap in a zone means one object; the floor ignores
    pixel-touching neighbors). Iterates to a fixed point."""
    if len(dets) < 2:
        return dets
    from backbone.detection.tiling import _same_object

    def _same(x: Detection, y: Detection) -> bool:
        return (str(x.cls) == str(y.cls)
                and (_same_object(x.bbox_xyxy, y.bbox_xyxy, 0.5)
                     or _clipped_same(x, y)
                     or _iou_xyxy(x.bbox_xyxy, y.bbox_xyxy) >= 0.10))

    pool = sorted(dets, key=lambda d: -float(d.confidence))
    merged = True
    while merged:
        merged = False
        for i in range(len(pool)):
            for j in range(i + 1, len(pool)):
                if _same(pool[i], pool[j]):
                    union = _merge_two(pool[i], pool[j])
                    pool = [d for k, d in enumerate(pool) if k not in (i, j)]
                    pool.append(union)
                    pool.sort(key=lambda d: -float(d.confidence))
                    merged = True
                    break
            if merged:
                break
    return pool


def zone_crop_boxes(
    rig,
    zones,
    *,
    crop_height_m: float = 0.0,
    margin_px: int = 16,
    min_side_px: int = 48,
) -> dict[str, list[tuple[str, tuple[int, int, int, int]]]]:
    """Per-camera crop boxes, CALIBRATION-frame pixels: ``{cam: [(zone, box)]}``.

    The box is the tight bbox of the zone polygon's corners projected at the
    ZONE'S OWN base plane (``Zone.z_base_m``; 0 = the floor, matching the
    pre-z_base behavior exactly), padded by ``margin_px`` and clipped to the
    frame. Zones whose visible box is smaller than ``min_side_px`` on either
    side (barely/not visible) are skipped.

    ``crop_height_m`` default 0 (config ``detection.zone_crop_height_m``):
    the crop IS the base-polygon rectangle — no headroom. Nonzero values
    additionally project the polygon at ``z_base_m + crop_height_m`` and take
    the union bbox (for sites whose tall loads must stay fully in-crop).
    History: 2.0 m ballooned a small floor zone into a frame-spanning strip
    at low camera angles (2026-07-23 live case: Sortie_1 on cam_b covered
    (0,0,336,1080)).

    Mode-1 placeholder extrinsics (only ``H`` is real) can't lift the polygon
    off the floor: a raised zone projects at the floor like before, and with a
    nonzero ``crop_height_m`` the floor bbox is extended UPWARD by its own
    height as a conservative stand-in.
    """
    boxes: dict[str, list[tuple[str, tuple[int, int, int, int]]]] = {}
    for cam_id in rig.camera_ids:
        view = rig[cam_id]
        w, h = int(view.image_size_wh[0]), int(view.image_size_wh[1])
        cam_boxes = []
        for name in zones.names:
            zone = zones[name]
            poly = densify_polygon(zone.polygon, segments_per_edge=8)
            z0 = float(zone.z_base_m)
            if has_metric_camera_model(view.K, view.R, view.t):
                base3 = np.hstack([poly, np.full((len(poly), 1), z0)])
                pts3 = [_project_world3(base3, view.K, view.D, view.R, view.t,
                                        view.image_size_wh)]
                if crop_height_m > 0:
                    top3 = np.hstack(
                        [poly,
                         np.full((len(poly), 1), z0 + float(crop_height_m))])
                    pts3.append(_project_world3(top3, view.K, view.D, view.R,
                                                view.t, view.image_size_wh))
                uv = np.vstack(pts3)
                uv = uv[~np.isnan(uv).any(axis=1)]
            else:
                uv = floor_to_pixel(poly, view.H)
            if len(uv) == 0:
                continue
            x0 = math.floor(uv[:, 0].min()) - margin_px
            x1 = math.ceil(uv[:, 0].max()) + margin_px
            y0 = math.floor(uv[:, 1].min()) - margin_px
            y1 = math.ceil(uv[:, 1].max()) + margin_px
            if crop_height_m > 0 and not has_metric_camera_model(
                    view.K, view.R, view.t):
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


def zone_fill_polygons(
    rig, zones, *, crop_height_m: float = 0.0,
) -> dict[str, dict[str, tuple[np.ndarray, float]]]:
    """Projected zone polygons for the crop outside-fill:
    ``{cam: {zone: (poly_px Nx2, dilate_px)}}``, CALIBRATION-frame pixels.

    The rectangular crop covers more than the polygon (its corner triangles,
    ~35-45 % of the area on this rig); with fill on, ``detect()`` blanks those
    pixels to neutral gray so the detector cannot see off-zone objects at all
    (the membership filter then only mops up boundary cases). ``dilate_px`` is
    ``_FILL_DILATE_M`` metres at the zone's local pixel scale. Each zone
    projects at ITS OWN base plane (``Zone.z_base_m``; Mode-1 H-only rigs
    fall back to the floor). With ``crop_height_m`` > 0 the fill region is
    the convex hull of the base and top (``z_base_m + crop_height_m``)
    projections. A zone whose projection is partially invalid (behind the
    camera / non-finite) gets NO fill entry — its crop stays unfilled
    (fail-open, same spirit as the membership filter).
    """
    out: dict[str, dict[str, tuple[np.ndarray, float]]] = {}
    for cam_id in rig.camera_ids:
        view = rig[cam_id]
        cam_polys: dict[str, tuple[np.ndarray, float]] = {}
        for name in zones.names:
            zone = zones[name]
            poly = densify_polygon(zone.polygon, segments_per_edge=8)
            z0 = float(zone.z_base_m)
            metric_area = float(cv2.contourArea(
                np.asarray(poly, dtype=np.float32)))
            metric = has_metric_camera_model(view.K, view.R, view.t)
            if metric:
                base3 = np.hstack([poly, np.full((len(poly), 1), z0)])
                uv = _project_world3(base3, view.K, view.D, view.R, view.t,
                                     view.image_size_wh)
            else:
                uv = floor_to_pixel(poly, view.H)
            if len(uv) < 3 or not np.isfinite(uv).all():
                continue
            fill_uv = uv
            if crop_height_m > 0 and metric:
                top3 = np.hstack(
                    [poly, np.full((len(poly), 1), z0 + float(crop_height_m))])
                top = _project_world3(top3, view.K, view.D, view.R, view.t,
                                      view.image_size_wh)
                if not np.isfinite(top).all():
                    continue
                fill_uv = cv2.convexHull(
                    np.vstack([uv, top]).astype(np.float32)).reshape(-1, 2)
            px_area = float(cv2.contourArea(fill_uv.astype(np.float32)))
            if metric_area <= 0 or px_area <= 0:
                continue
            dilate_px = _FILL_DILATE_M * math.sqrt(px_area / metric_area)
            cam_polys[name] = (np.asarray(fill_uv, dtype=np.float64),
                               float(dilate_px))
        out[cam_id] = cam_polys
    return out


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
                 batch_buckets: tuple[int, ...] | None = None,
                 max_crop_aspect: float = _MAX_CROP_ASPECT,
                 zone_filter=None,
                 fill_polys: dict | None = None) -> None:
        self._detector = detector
        self._boxes = boxes_by_camera
        # In-zone membership filter (build_zone_membership_filter): drops
        # object detections whose metric foot lies outside every zone polygon.
        # Called with CALIBRATION-frame pixels; detect() rescales.
        self._zone_filter = zone_filter
        # Polygon fill (zone_fill_polygons): blank crop pixels outside the
        # dilated zone polygon before inference. Outside-masks are rasterized
        # once per (cam, zone, crop shape) and cached — frame sizes are stable
        # at runtime, so the cache stays bounded.
        self._fill_polys = fill_polys or {}
        self._fill_cache: dict[tuple, np.ndarray] = {}
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
        # Extreme-aspect crops square-tile themselves (see detect()); 0/None
        # disables (hermetic tests with static-batch stub models need a
        # stable batch; production dynamic exports don't care).
        self._max_aspect = float(max_crop_aspect or 0.0)
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
        crop_rect: dict[str, tuple[float, float, float, float]] = {}
        for cam_id, frame in pair.frames.items():
            cam_boxes = self._boxes.get(cam_id) or []
            if not cam_boxes:
                continue
            fh, fw = frame.image.shape[:2]
            # Crop boxes are calibration-frame px; frames may be ingest-downscaled.
            calib_w, calib_h = self._calib_wh.get(cam_id, (fw, fh))
            sx, sy = fw / calib_w, fh / calib_h
            for i, (zone_name, (x0, y0, x1, y1)) in enumerate(cam_boxes):
                fx0, fy0 = max(0, int(x0 * sx)), max(0, int(y0 * sy))
                fx1, fy1 = min(fw, math.ceil(x1 * sx)), min(fh, math.ceil(y1 * sy))
                if fx1 - fx0 < 8 or fy1 - fy0 < 8:
                    continue
                crop_img = frame.image[fy0:fy1, fx0:fx1]
                fill = (self._fill_polys.get(cam_id) or {}).get(zone_name)
                if fill is not None:
                    crop_img = self._fill_outside(
                        cam_id, zone_name, crop_img, fill, sx, sy, fx0, fy0)
                if self._enh_on:
                    crop_img = enhance_bgr(crop_img, **self._enh_kwargs)
                ch, cw = crop_img.shape[:2]
                tile = self._tile or self._model_input_px()
                if self._sahi_on:
                    rects = tile_boxes(cw, ch, tile, self._overlap)
                elif self._max_aspect and max(cw, ch) > self._max_aspect * min(cw, ch):
                    # Extreme-aspect crop (an edge-on zone whose z-extruded
                    # projection is a tall/wide strip): letterboxing the whole
                    # strip into the square model input shrinks the objects
                    # ~3x and the detector goes blind (measured: a palette at
                    # 0.56 in a square crop scored 0.0 in the 1:3.2 strip).
                    # Square-tile JUST this crop — the tiles ride the same
                    # batched call and merge below. Not the global SAHI knob:
                    # this is geometry-triggered, per crop, always on.
                    rects = tile_boxes(cw, ch, min(cw, ch), self._overlap)
                else:
                    rects = [(0, 0, cw, ch)]
                for t, (tx0, ty0, tx1, ty1) in enumerate(rects):
                    sid = f"{cam_id}{_SEP}{i}{_SEP}{t}"
                    crops[sid] = Frame(camera_id=sid, capture_ts=frame.capture_ts,
                                       frame_idx=frame.frame_idx,
                                       image=crop_img[ty0:ty1, tx0:tx1])
                    # Origin folds the zone-crop offset AND the tile offset, so
                    # a detection maps straight back to frame coordinates.
                    origin[sid] = (cam_id, fx0 + tx0, fy0 + ty0)
                    # The crop's window in frame px — the deduper's clip rule
                    # needs to know where each detection's view was cut off.
                    crop_rect[sid] = (float(fx0 + tx0), float(fy0 + ty0),
                                      float(fx0 + tx1), float(fy0 + ty1))
        if not crops:
            return out
        raw = self._detect_padded(pair, crops)
        # Tiled zones: merge each zone's tiles before remapping, so an object
        # split across two tiles becomes one detection with the union box.
        # (Global SAHI tiles everything; extreme-aspect crops tile themselves —
        # any sid with tile index > 0 means a merge pass is needed.)
        if self._sahi_on or any(sid.rsplit(_SEP, 1)[1] != "0" for sid in raw):
            raw = self._merge_zone_tiles(raw, origin)
        for sid, dets in raw.items():
            cam_id, ox, oy = origin[sid]
            for d in dets:
                d.camera_id = cam_id
                d.crop_xyxy = crop_rect.get(sid)
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
                    # full-frame canvas per detection. COMPOSE with any tile
                    # offset already on the det (shift_detection stored the
                    # mask's tile position within the zone crop) — overwriting
                    # it anchored every tiled mask at the crop origin, up to a
                    # crop-height away from the object ('boxes but no masks').
                    tx, ty = getattr(d, "mask_offset_xy", None) or (0, 0)
                    d.mask_offset_xy = (ox + tx, oy + ty)
                out[cam_id].append(d)
        for cam_id, dets in out.items():
            # In-zone guarantee: drop detections whose metric foot is outside
            # every zone polygon (the rectangular crop necessarily covers more
            # than the zone; the polygon is the truth). Persons never pass
            # through zone scope, so no exemption is needed here.
            if self._zone_filter is not None and dets:
                frame = pair.frames.get(cam_id)
                if frame is not None:
                    fh, fw = frame.image.shape[:2]
                    cw, ch = self._calib_wh.get(cam_id, (fw, fh))
                    kx, ky = cw / float(fw), ch / float(fh)
                    dets = [d for d in dets
                            if self._zone_filter(
                                cam_id,
                                (d.foot_uv[0] * kx, d.foot_uv[1] * ky))]
            out[cam_id] = _dedup_across_crops(dets)
        return out

    # ---- internals ----

    def _fill_outside(self, cam_id, zone_name, crop_img, fill,
                      sx, sy, fx0, fy0) -> np.ndarray:
        """Blank crop pixels outside the (dilated) zone polygon to neutral
        gray. Returns a COPY — the crop is a view into the shared frame."""
        ch, cw = crop_img.shape[:2]
        key = (cam_id, zone_name, ch, cw, fx0, fy0)
        outside = self._fill_cache.get(key)
        if outside is None:
            poly_px, dilate_px = fill
            pts = np.round(poly_px * (sx, sy) - (fx0, fy0)).astype(np.int32)
            inside = np.zeros((ch, cw), np.uint8)
            cv2.fillPoly(inside, [pts], 255)
            r = max(1, round(dilate_px * (sx + sy) / 2.0))
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
            inside = cv2.dilate(inside, kernel)
            outside = inside == 0
            self._fill_cache[key] = outside
        crop_img = crop_img.copy()
        crop_img[outside] = _FILL_GRAY
        return crop_img

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
