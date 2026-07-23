"""Frame annotation — the OVERLAY stage: boxes/masks/pose drawing, distance
lines, pallet occupancy, annotate_frame, and the UI display preferences.
Pure pixels in, pixels out; sessions live in engines.py. Split out of
detection_overlay.py.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from .engines import (
    _GPU_MIN_FREE_MB,
    gpu_inference_safe,
)
from .model_store import (
    read_backbone,
)
from .yaml_cache import load_yaml_cached

logger = logging.getLogger(__name__)


_BOX_COLOR = (80, 220, 80)  # BGR — default / unknown class


_gpu_skip_log_ts = 0.0


# THE canonical per-class overlay palette, as #rrggbb. Every renderer — the
# server-side zone panels (Python/OpenCV) AND the client-side cam views
# (JS/canvas) — must use these, or the same pallet appears green in one view
# and blue in another (it did). Served to the browser via /api/ui-settings so
# there is exactly one source of truth; the JS keeps an identical fallback
# table, pinned equal by a test.
CLASS_COLORS_HEX: dict[str, str] = {
    "palette": "#50dc50",    # green
    "pallet": "#50dc50",     # alias
    "carton": "#ff7878",     # light red
    "polybag": "#78b4ff",    # light blue
    "person": "#ffd54f",     # amber
    "forklift": "#ff7043",   # orange
}
DEFAULT_CLASS_COLOR_HEX = "#ffffff"


def _hex_to_bgr(value: str) -> tuple[int, int, int]:
    v = value.lstrip("#")
    r, g, b = int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16)
    return (b, g, r)


_CLASS_COLORS = {k: _hex_to_bgr(v) for k, v in CLASS_COLORS_HEX.items()}


def _color_for(cls) -> tuple[int, int, int]:
    return _CLASS_COLORS.get(str(cls).lower(), _BOX_COLOR)


# Detection classes treated as a "pallet" for person↔pallet distance lines.
_PALLET_CLASSES = {"palette", "pallet", "palette_vide"}


def draw(image, det, show_nodes: bool = True, show_masks: bool = True,
         show_boxes: bool = True, mask_clip=None) -> None:
    """``mask_clip`` (optional uint8 stencil, 255 inside the zones) bounds the
    mask fill: a zone-based system shows nothing outside the zone, and a mask
    hugging the boundary must not spill past the dashed outline even though
    the detection itself (foot inside) is legitimately kept."""
    color = _color_for(det.cls)   # per-class: palette=green, carton=light-red, polybag=light-blue
    # Segmentation mask underlay — only set by `yolo_onnx_seg` (or a future seg
    # plugin); detect detectors leave `det.mask = None` and this branch is skipped.
    if show_masks and getattr(det, "mask", None) is not None:
        m = det.mask
        # Blend the class colour only inside the mask. `addWeighted` is the
        # cheapest cv2 path; the boolean mask keeps the blend localised.
        if m.shape == image.shape[:2]:
            if mask_clip is not None:
                m = m & (mask_clip > 0)
            overlay = image.copy()
            overlay[m] = color
            cv2.addWeighted(overlay, 0.35, image, 0.65, 0, dst=image)
    elif show_masks and getattr(det, "mask_poly", None):
        # Wire observations carry the instance mask as a simplified POLYGON
        # (never a bitmap) — fill it with the same translucent class colour.
        pts = np.asarray(det.mask_poly, dtype=np.int32).reshape(-1, 1, 2)
        if len(pts) >= 3:
            region = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(region, [pts], 255)
            if mask_clip is not None:
                cv2.bitwise_and(region, mask_clip, dst=region)
            overlay = image.copy()
            overlay[region > 0] = color
            cv2.addWeighted(overlay, 0.35, image, 0.65, 0, dst=image)
    x1, y1, x2, y2 = (int(v) for v in det.bbox_xyxy)
    # The bounding box is optional (Settings toggle) — with a seg model the mask +
    # label alone often reads cleaner. The class-name label is always drawn,
    # anchored at the box's top-left even when the box itself is hidden.
    if show_boxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    label = f"{det.cls} {det.confidence:.2f}"
    cv2.putText(image, label, (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    # Object foot/edge nodes are intentionally NOT drawn — the only node point
    # shown is the PERSON foot node (drawn by the pose overlay). The pallet's
    # bbox-edge nodes are used only to anchor the nearest distance line, not shown
    # as points. (`show_nodes` is kept for signature compatibility / future use.)
    _ = show_nodes


def _project_to_floor(feet_uv, K, D, H):
    """Project pixel foot points to floor metres via undistort → homography.
    Reuses the SAME backbone geometry the metric pipeline uses, so the metres
    match what the Backbone would compute."""
    from backbone.shared.geometry import pixel_to_floor, undistort_points

    pts = np.asarray(feet_uv, dtype=np.float64).reshape(-1, 2)
    return pixel_to_floor(undistort_points(pts, K, D), H)


def compute_person_pallet_distances(person_feet, pallet_feet, view, frame_wh, *,
                                    max_m: float = 6.0):
    """Return ``[(person_uv, pallet_uv, d_m), ...]`` for every person↔pallet pair
    within ``max_m`` metres, using the floor homography. Pure (no drawing) so it's
    unit-testable. ``view`` carries the calibrated ``K, D, H, image_size_wh``.

    Frame-size guard: when the live frame size differs from the calibration size,
    ``H`` is rescaled (``H @ diag(cal_w/iw, cal_h/ih, 1)``) so actual-frame pixels
    map to the right metres — the same guard the MAP warp uses."""
    if not person_feet or not pallet_feet:
        return []
    H = np.asarray(view.H, dtype=np.float64)
    iw, ih = int(frame_wh[0]), int(frame_wh[1])
    cal_w, cal_h = int(view.image_size_wh[0]), int(view.image_size_wh[1])
    if (iw, ih) != (cal_w, cal_h):
        H = H @ np.diag([cal_w / iw, cal_h / ih, 1.0])
    K = np.asarray(view.K, dtype=np.float64)
    D = np.asarray(view.D, dtype=np.float64)
    persons_m = _project_to_floor(person_feet, K, D, H)
    pallets_m = _project_to_floor(pallet_feet, K, D, H)
    out = []
    for i, puv in enumerate(person_feet):
        for j, quv in enumerate(pallet_feet):
            d_m = float(np.hypot(persons_m[i, 0] - pallets_m[j, 0],
                                 persons_m[i, 1] - pallets_m[j, 1]))
            if d_m <= max_m:
                out.append((puv, quv, d_m))
    return out


def _filled_rounded_rect(img, x1, y1, x2, y2, r, color) -> None:
    """Filled rounded rectangle (cv2 has no native one): two rects + 4 corner discs."""
    r = max(0, min(r, (x2 - x1) // 2, (y2 - y1) // 2))
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
    for ccx, ccy in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)):
        cv2.circle(img, (ccx, ccy), r, color, -1, cv2.LINE_AA)


def _draw_distance(image, p1, p2, d_m: float, *, style=None) -> None:
    """Elastic line ``p1→p2`` + a white rounded centre badge with black ``'X.X m'``
    text. Line look (opacity / colour / thickness) comes from ``style`` (UI-settings
    via :func:`distance_line_style`); defaults to faint white 2 px."""
    opacity = float((style or {}).get("opacity", 0.25))
    color = (style or {}).get("color", (255, 255, 255))
    thickness = int((style or {}).get("thickness", 2))
    h, w = image.shape[:2]
    # Blend the line over the frame at `opacity` (line drawn on a copy → only its
    # pixels are blended).
    overlay = image.copy()
    cv2.line(overlay, p1, p2, color, thickness, cv2.LINE_AA)
    cv2.addWeighted(overlay, opacity, image, 1.0 - opacity, 0, dst=image)
    fs = max(0.4, min(h, w) / 1400.0)            # font scales with frame
    th = max(1, round(min(h, w) / 720.0))
    text = f"{d_m:.1f} m"
    (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
    cx, cy = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
    pad = max(3, round(min(h, w) / 360.0))
    rad = max(4, round(min(h, w) / 220.0))
    _filled_rounded_rect(image, cx - tw // 2 - pad, cy - tht // 2 - pad,
                         cx + tw // 2 + pad, cy + tht // 2 + pad, rad, (255, 255, 255))
    cv2.putText(image, text, (cx - tw // 2, cy + tht // 2),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th, cv2.LINE_AA)


def _bbox_edge_nodes(bbox) -> list[tuple[float, float]]:
    """The 4 edge-midpoint nodes of a bbox (pixel uv): top-mid, bottom-mid, left-mid,
    right-mid. Used so the distance line attaches to the pallet edge NEAREST the
    person, not always the bottom-centre."""
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return [(cx, y1), (cx, y2), (x1, cy), (x2, cy)]


def draw_person_pallet_distances(image, detections, poses, view, *, max_m: float = 6.0,
                                 style=None) -> None:
    """Draw ONE white line + metre badge per pallet: from the nearest person foot node
    to that pallet's NEAREST bbox edge-midpoint node (of its 4 edge mids). For each
    pallet we measure the metric distance from every person to each of its 4 edge nodes
    and keep only the single shortest — so there's exactly one line per pallet, to the
    closest edge. Distances are metric via the floor homography in ``view``; no-op if
    either list is empty / projection fails."""
    pallets = [d for d in detections
               if str(getattr(d, "cls", "")).lower() in _PALLET_CLASSES
               and getattr(d, "bbox_xyxy", None) is not None]
    person_feet = [p.foot_uv for p in poses if getattr(p, "foot_uv", None) is not None]
    if not pallets or not person_feet:
        return
    frame_wh = (image.shape[1], image.shape[0])
    for pal in pallets:
        nodes = _bbox_edge_nodes(pal.bbox_xyxy)        # 4 edge midpoints
        try:
            # distances from every person to each of this pallet's 4 nodes (within max_m)
            pairs = compute_person_pallet_distances(person_feet, nodes, view, frame_wh, max_m=max_m)
        except Exception:
            logger.warning("distance overlay: projection failed", exc_info=True)
            continue
        if not pairs:
            continue
        # retain only the lowest-distance line for this bbox → one line per pallet
        puv, quv, d_m = min(pairs, key=lambda t: t[2])
        _draw_distance(image, (int(puv[0]), int(puv[1])),
                       (int(quv[0]), int(quv[1])), d_m, style=style)


# Load objects + per-state badge colours (BGR). Matches the 2D-map cue:
# empty=green · carton=amber · polybag=blue.
_OBJECT_CLASSES = {"carton", "polybag"}


_OCC_COLORS = {"empty": (80, 220, 80), "loaded": (53, 171, 245)}   # BGR: green / amber


# Canonical content order so a multi-load label is stable frame-to-frame (never
# "palette_polybag_carton" one frame and "palette_carton_polybag" the next).
_CONTENT_ORDER = ("carton", "polybag")


def _occupancy_label(contents) -> str:
    """Pallet label depicting presence + load: ``palette_vide`` when empty, else
    ``palette_<loads>`` in canonical order — e.g. ``palette_carton``,
    ``palette_polybag``, ``palette_carton_polybag``."""
    if not contents:
        return "palette_vide"
    ordered = [c for c in _CONTENT_ORDER if c in contents]
    ordered += sorted(c for c in contents if c not in _CONTENT_ORDER)   # unknowns last
    return "palette_" + "_".join(ordered)


def image_occupancy(detections, *, k: float = 1.5, a_min: float = 0.2):
    """Per-pallet empty/full from IMAGE OVERLAP (the A estimator) on one frame's
    detections — self-contained, no calibration. An object is "on" a pallet if its
    base sits in the pallet's box extended upward by ``k*`` its height and they
    overlap horizontally (fraction ≥ ``a_min``). Returns ``[(pallet_det, label), ...]``
    where label depicts pallet presence + its full load set — ``palette_vide`` when
    empty, else ``palette_<loads>`` (e.g. ``palette_carton``, ``palette_carton_polybag``).

    This mirrors the Backbone's A-association so the CAM preview agrees with the
    metric pipeline, without importing the homography layer (process boundary)."""
    pallets = [d for d in detections if str(getattr(d, "cls", "")).lower() in _PALLET_CLASSES]
    objects = [d for d in detections if str(getattr(d, "cls", "")).lower() in _OBJECT_CLASSES]
    loads: dict[int, list] = {i: [] for i in range(len(pallets))}
    for obj in objects:
        ox1, _oy1, ox2, oy2 = obj.bbox_xyxy
        best_i, best_s = None, a_min
        for i, pal in enumerate(pallets):
            px1, py1, px2, py2 = pal.bbox_xyxy
            h = max(1e-6, py2 - py1)
            if not (py1 - k * h <= oy2 <= py2 + 0.5 * h):
                continue
            overlap = max(0.0, min(ox2, px2) - max(ox1, px1)) / max(1e-6, ox2 - ox1)
            if overlap >= best_s:
                best_s, best_i = overlap, i
        if best_i is not None:
            loads[best_i].append(obj)

    out = []
    for i, pal in enumerate(pallets):
        contents = {str(o.cls).lower() for o in loads[i]}
        out.append((pal, _occupancy_label(contents)))
    return out


def _draw_occupancy_badge(image, pallet_det, label) -> None:
    """A small filled colour badge ('palette_vide' / 'palette_carton' / …) above a
    pallet box — green when empty, amber when loaded."""
    color = _OCC_COLORS["empty"] if label == "palette_vide" else _OCC_COLORS["loaded"]
    x1, y1, _x2, _y2 = (int(v) for v in pallet_det.bbox_xyxy)
    h, w = image.shape[:2]
    fs = max(0.45, min(h, w) / 1300.0)
    th = max(1, round(min(h, w) / 800.0))
    (tw, tht), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
    pad = max(3, round(min(h, w) / 360.0))
    by2 = max(tht + 2 * pad, y1)                 # badge sits just above the box
    cv2.rectangle(image, (x1, by2 - (tht + 2 * pad)), (x1 + tw + 2 * pad, by2), color, -1)
    cv2.putText(image, label, (x1 + pad, by2 - pad),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (20, 20, 20), th, cv2.LINE_AA)


def annotate_frame(image, detector, cam_id: str = "cam",
                   show_nodes: bool = True, show_masks: bool = True,
                   show_boxes: bool = True, pose_detector=None,
                   dist_view=None, dist_max_m: float = 6.0,
                   show_occupancy: bool = False, detections=None, dist_style=None,
                   mask_clip=None):
    """Run detection on one BGR frame and draw masks (seg only) + boxes + foot
    nodes in place. When ``pose_detector`` is given, also run person-pose and draw
    skeletons + foot nodes. When ``dist_view`` (a calibrated camera) is given, draw
    a white line + metre badge from each person to each pallet. When
    ``show_occupancy``, draw a pallet empty/full badge. Never raises on a bad frame
    — returns the (possibly un-annotated) image so a stream won't break."""
    from backbone.core.types import Frame, FramePair

    # GPU-pressure guard: if the card is nearly full, skip this frame's inference
    # (return the raw image) rather than risk the OOM that corrupts the CUDA
    # context. The preview yields to the live Backbone; boxes simply don't refresh
    # for the throttled frames.
    if (detector is not None or pose_detector is not None) and not gpu_inference_safe():
        global _gpu_skip_log_ts
        now = time.time()
        if now - _gpu_skip_log_ts > 10.0:
            logger.warning(
                "GPU low on VRAM (<%d MB free) — preview skipping inference to "
                "protect the CUDA context", _GPU_MIN_FREE_MB,
            )
            _gpu_skip_log_ts = now
        return image

    ts = time.time()
    pair = FramePair(
        capture_ts=ts, frame_idx=0,
        frames={cam_id: Frame(camera_id=cam_id, capture_ts=ts, frame_idx=0, image=image)},
    )
    # `detections` may be PRE-COMPUTED (e.g. zone-based detection mapped back to the
    # full frame — cam1 then runs no heavy full-frame detector, only pose). When it
    # is None, detect on the full frame as usual.
    incoming = detections
    detections = []
    if incoming is not None:
        detections = list(incoming)
    elif detector is not None:
        try:
            detections = list(detector.detect(pair).get(cam_id, []))
        except Exception:
            logger.warning("detection overlay: detect failed", exc_info=True)
            detections = []
    for det in detections:
        draw(image, det, show_nodes=show_nodes, show_masks=show_masks,
             show_boxes=show_boxes, mask_clip=mask_clip)
    poses = []
    if pose_detector is not None:
        try:
            poses = pose_detector.predict(image)
            pose_detector.draw(image, poses)
        except Exception:
            logger.warning("pose overlay: draw failed", exc_info=True)
            poses = []
    # Person↔pallet distance lines need both lists + a calibrated view.
    if dist_view is not None and detections and poses:
        try:
            draw_person_pallet_distances(image, detections, poses, dist_view,
                                         max_m=dist_max_m, style=dist_style)
        except Exception:
            logger.warning("distance overlay: draw failed", exc_info=True)
    # Pallet empty/full badge (image-space association — no calibration needed).
    if show_occupancy and detections:
        try:
            for pal, label in image_occupancy(detections):
                _draw_occupancy_badge(image, pal, label)
        except Exception:
            logger.warning("occupancy overlay: draw failed", exc_info=True)
    return image


def _ui_pref(cfg, key: str, default: bool = True) -> bool:
    """Read one boolean preference from the UI-settings YAML; default if missing.

    Cached by file mtime (``yaml_cache``): this is called several times PER
    FRAME PER STREAM, and an uncached parse of the settings file costs ~18 ms
    of GIL-held work — enough to starve the event loop outright.
    """
    data = load_yaml_cached(cfg.ui_settings_path)
    if not data:
        return default
    return bool(data.get(key, default))


def nodes_enabled(cfg) -> bool:
    """Dashboard preference: draw the white foot-node disc on each detection."""
    return _ui_pref(cfg, "show_nodes", True)


def masks_enabled(cfg) -> bool:
    """Dashboard preference: draw the seg mask overlay (only meaningful for
    seg detectors — detect detectors have no mask)."""
    return _ui_pref(cfg, "show_masks", True)


def floor_zones_enabled(cfg) -> bool:
    """Dashboard preference: draw the projected FLOOR-ZONE outlines on the cam
    views. Off by default — the operator's dashed zone patches stay the always-
    visible boundary; this adds the metric zone geometry on demand. Display
    only: detections + masks stay zone-clipped regardless."""
    return _ui_pref(cfg, "show_floor_zones", False)


def zone_fill_dim_enabled(cfg) -> bool:
    """Dashboard preference: on the ZONE panels, darken the pixels the
    producer's polygon fill blanks before inference (zone_crop_polygon_fill) —
    shows the detector's true field of view. Slightly conservative: the
    producer dilates the polygon ~0.3 m, so the dimmed band is a little wider
    than what the detector actually loses. Off by default. Display only."""
    return _ui_pref(cfg, "show_zone_fill", False)


def occupancy_enabled(cfg) -> bool:
    """Dashboard preference: draw the pallet empty/full badge on the CAM overlay."""
    return _ui_pref(cfg, "show_occupancy", True)


def boxes_enabled(cfg) -> bool:
    """Dashboard preference: draw the detection bounding box. Off ⇒ mask + class
    label only (cleaner with a seg model)."""
    return _ui_pref(cfg, "show_boxes", True)


def distances_enabled(cfg) -> bool:
    """Dashboard preference: draw the person↔pallet distance lines (needs a pose
    model + calibration; degrades to no lines otherwise)."""
    return _ui_pref(cfg, "show_distances", True)




def person_pallet_max_m(cfg) -> float:
    """Max person↔pallet distance (m) to draw a line for — caps clutter. Reads
    ``detection.person_pallet_max_distance_m`` from backbone.yaml (default 6.0)."""
    det = read_backbone(cfg).get("detection") or {}
    try:
        return float(det.get("person_pallet_max_distance_m", 6.0))
    except (TypeError, ValueError):
        return 6.0


def _hex_to_bgr(s, default=(255, 255, 255)) -> tuple[int, int, int]:
    """'#rrggbb' (or '#rgb') → (B, G, R) for cv2. Bad input → default."""
    try:
        h = str(s).strip().lstrip("#")
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))   # B, G, R
    except (ValueError, TypeError, IndexError):
        return default


def distance_line_style(cfg) -> dict:
    """Person↔pallet distance-line look from the UI-settings YAML: ``opacity`` [0..1],
    ``color`` (``#rrggbb`` → BGR tuple), ``thickness`` px. Defaults: 0.25 / white / 2."""
    path = Path(cfg.ui_settings_path)
    opacity, color, thickness = 0.25, (255, 255, 255), 2
    if path.exists():
        try:
            data = load_yaml_cached(path)
            if isinstance(data, dict):
                if data.get("distance_line_opacity") is not None:
                    opacity = float(data["distance_line_opacity"])
                if data.get("distance_line_color"):
                    color = _hex_to_bgr(data["distance_line_color"])
                if data.get("distance_line_thickness") is not None:
                    thickness = int(data["distance_line_thickness"])
        except (OSError, yaml.YAMLError, TypeError, ValueError):
            pass
    return {"opacity": max(0.05, min(1.0, opacity)), "color": color,
            "thickness": max(1, min(8, thickness))}
