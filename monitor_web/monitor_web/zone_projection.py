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
