"""Project the metric floor zones into a camera — the SINGLE source of truth for
the cam-view zone outline AND the detection clip.

Zone-based system: on the cam view a detection is shown ONLY when its foot point
lands inside a zone, and each zone is drawn as a dashed outline. Both come from
the SAME floor zones the producer scopes detection to (``zones.yaml``), projected
through calibration — so there is nothing to desync (no dependency on the
pixel-space ``zone_patches``, which the cam view no longer needs).

isistream detects inside each zone's bounding-box CROP (a rectangle a bit larger
than the polygon), so its observations can carry objects in the rectangular
margin outside the zone shape. Clipping the display to the projected polygon is
what enforces "no detections outside the zone".
"""

from __future__ import annotations

import cv2
import numpy as np
from backbone.shared.geometry import (
    densify_polygon,
    floor_to_pixel,
    has_metric_camera_model,
    project_floor_polygon_distorted,
)


def project_zone_polygons(rig, zones, camera_id: str) -> list:
    """Per-zone pixel polygons in the camera's CALIBRATION frame.

    Returns ``[(zone_id, name, poly)]`` where ``poly`` is an ``(N, 2)`` float
    array of pixel vertices. Mode 2 (real ``K, D, R, t``) projects distortion-
    aware and clipped to the reliably-projectable field; Mode 1 (only ``H``)
    maps through the homography. Zones the camera cannot see are dropped.
    """
    if rig is None or camera_id not in rig:
        return []
    view = rig[camera_id]
    out: list = []
    for name in zones.names:
        poly_m = densify_polygon(
            np.asarray(zones[name].polygon, dtype=np.float64), segments_per_edge=8)
        if has_metric_camera_model(view.K, view.R, view.t):
            px = project_floor_polygon_distorted(
                poly_m, view.K, view.D, view.R, view.t, view.image_size_wh)
        else:
            px = floor_to_pixel(poly_m, view.H)
        if px is None or len(px) < 3:
            continue
        out.append((zones.id_of(name) or name, name, np.asarray(px, dtype=np.float64)))
    return out


def scale_polygons(polys: list, sx: float, sy: float) -> list:
    """Scale calibration-frame polygons into another frame (e.g. the display)."""
    s = np.array([sx, sy], dtype=np.float64)
    return [(zid, name, poly * s) for zid, name, poly in polys]


def zone_of_point(pt, polys) -> str | None:
    """The id of the first zone whose polygon contains ``pt`` (or ``None``)."""
    for zid, _name, poly in polys:
        if len(poly) >= 3 and cv2.pointPolygonTest(
                poly.astype(np.float32), (float(pt[0]), float(pt[1])), False) >= 0:
            return zid
    return None


def clip_to_zones(dets: list, polys: list) -> list:
    """Keep only detections whose FOOT point falls inside a zone polygon — a
    zone-based cam view shows nothing outside the zones. The foot point is the
    object's ground contact, so membership matches the metric floor zone. Each
    surviving det is tagged with ``zone_id``. No zones ⇒ nothing shown."""
    kept: list = []
    for d in dets:
        zid = zone_of_point(d.foot_uv, polys)
        if zid is not None:
            d.zone_id = zid
            kept.append(d)
    return kept


def draw_zone_outlines(image: np.ndarray, polys: list, *, color=(0, 220, 255),
                       thickness: int = 2, dash: int = 14, gap: int = 9) -> None:
    """Draw each zone as a DASHED closed polygon — the user-facing boundary."""
    for _zid, _name, poly in polys:
        pts = poly.astype(int)
        n = len(pts)
        for i in range(n):
            _dashed_line(image, pts[i], pts[(i + 1) % n], color, thickness, dash, gap)


def _dashed_line(image, p0, p1, color, thickness, dash, gap) -> None:
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    dist = float(np.hypot(*(p1 - p0)))
    if dist < 1e-6:
        return
    unit = (p1 - p0) / dist
    step = dash + gap
    s = 0.0
    while s < dist:
        e = min(s + dash, dist)
        a = tuple((p0 + unit * s).astype(int))
        b = tuple((p0 + unit * e).astype(int))
        cv2.line(image, a, b, color, thickness)
        s += step


def zone_stencil(shape_hw, polys) -> np.ndarray:
    """Rasterize zone polygons into a uint8 stencil (255 inside a zone) — the
    mask-clip handed to the overlay so no mask pixel renders outside a zone."""
    m = np.zeros((int(shape_hw[0]), int(shape_hw[1])), dtype=np.uint8)
    for _zid, _name, poly in polys:
        if len(poly) >= 3:
            cv2.fillPoly(m, [np.asarray(poly, dtype=np.int32)], 255)
    return m


# Boundary tolerance for METRIC zone membership on the cam views. Measured on
# the rig: genuine boundary cases (an object physically in the zone whose
# per-camera foot projection lands just outside the polygon) miss by
# 0.05-0.11 m; crop-margin junk sits 0.38 m+ out. 0.15 m splits them cleanly.
_ZONE_TOL_M = 0.15


def clip_to_zones_metric(dets: list, view, display_wh, zones,
                         tol_m: float = _ZONE_TOL_M) -> list:
    """Zone membership in METRES — the same undistort+H the metric engine
    uses, so the cam view agrees with the fused zone state / COMMS card.

    A per-camera PIXEL polygon test is boundary-fragile: calibration skew puts
    the same physical object's foot inside one camera's projected polygon and
    a few dozen px outside the other's (box shown on cam2, dropped on cam1).
    Projecting the foot to the floor and testing against the metric zone
    polygon (± ``tol_m``) is camera-invariant. Each kept det gets ``zone_id``
    (nearest zone). No zones ⇒ nothing shown."""
    if not dets or len(zones) == 0:
        return []
    from backbone.shared.geometry import pixel_to_floor, undistort_points
    cw, ch = float(view.image_size_wh[0]), float(view.image_size_wh[1])
    fw, fh = float(display_wh[0]), float(display_wh[1])
    uv = np.array([[d.foot_uv[0] * cw / fw, d.foot_uv[1] * ch / fh]
                   for d in dets], dtype=np.float64)
    xy = pixel_to_floor(undistort_points(uv, view.K, view.D), view.H)
    polys = [(zones.id_of(n) or n, np.asarray(zones[n].polygon, np.float32))
             for n in zones.names]
    kept: list = []
    for d, (x, y) in zip(dets, xy, strict=True):
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        best = None
        for zid, poly in polys:
            dist = cv2.pointPolygonTest(poly, (float(x), float(y)), True)
            if best is None or dist > best[1]:
                best = (zid, dist)
        if best is not None and best[1] >= -float(tol_m):
            d.zone_id = best[0]
            kept.append(d)
    return kept


def crop_box_stencil(shape_hw, crop_boxes, scale_xy) -> np.ndarray:
    """Mask-clip stencil from the zones' CROP BOXES (z=0..2m projection union
    rects, calibration px, scaled to the display frame).

    The flat floor POLYGON is the wrong mask boundary: a pallet's mask rises
    above its floor footprint (obliquely-seen zones lost their whole mask),
    and a field-clipped polygon (edge-on zone) under-covers the metric zone
    (measured 0% mask survival on genuinely in-zone objects). Every wire mask
    was detected INSIDE a zone crop, so the crop-box union bounds every
    legitimate mask while still confining rendering to the zones' visual
    regions. Junk outside the zones is dropped by the metric clip upstream."""
    m = np.zeros((int(shape_hw[0]), int(shape_hw[1])), dtype=np.uint8)
    sx, sy = float(scale_xy[0]), float(scale_xy[1])
    for (x0, y0, x1, y1) in crop_boxes:
        m[int(y0 * sy):int(np.ceil(y1 * sy)), int(x0 * sx):int(np.ceil(x1 * sx))] = 255
    return m


def zone_of_foot_metric(view, display_wh, zones, foot_uv,
                        tol_m: float = _ZONE_TOL_M) -> str | None:
    """Single-foot version of :func:`clip_to_zones_metric` — the ONE membership
    rule (foot → floor via the camera's undistort+H → nearest zones.yaml
    polygon ± ``tol_m``) shared by the cam views AND the zone worker, so a
    detection can never be 'in the zone' on one surface and outside on
    another. Returns the zone id, or ``None``."""
    from backbone.shared.geometry import pixel_to_floor, undistort_points
    cw, ch = float(view.image_size_wh[0]), float(view.image_size_wh[1])
    fw, fh = float(display_wh[0]), float(display_wh[1])
    uv = np.array([[foot_uv[0] * cw / fw, foot_uv[1] * ch / fh]], dtype=np.float64)
    xy = pixel_to_floor(undistort_points(uv, view.K, view.D), view.H)[0]
    if not (np.isfinite(xy[0]) and np.isfinite(xy[1])):
        return None
    best = None
    for name in zones.names:
        poly = np.asarray(zones[name].polygon, np.float32)
        dist = cv2.pointPolygonTest(poly, (float(xy[0]), float(xy[1])), True)
        if best is None or dist > best[1]:
            best = (zones.id_of(name) or name, dist)
    if best is not None and best[1] >= -float(tol_m):
        return best[0]
    return None
