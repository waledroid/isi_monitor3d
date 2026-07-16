"""Project the metric floor zones into a camera — the SINGLE source of truth for
the cam-view zone outline AND the detection clip.

Zone-based system: on the cam view a detection is shown ONLY when its foot point
lands inside a zone, and each zone is drawn as a dashed outline. Both come from
the SAME floor zones the producer scopes detection to (``zones.yaml``), projected
through calibration — so there is nothing to desync (no dependency on the
pixel-space ``zone_patches``, which the cam view no longer needs).

isistream detects inside each zone's bounding-box CROP (a rectangle a bit larger
than the polygon), so its observations can carry objects in the rectangular
margin outside the zone shape. The METRIC membership test (clip_to_zones_metric
/ zone_of_foot_metric, foot → floor → zones.yaml polygon ± tolerance) is what
enforces "no detections outside the zone" — identically on every surface.
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


def project_zone_hulls(rig, zones, camera_id: str, *,
                       height_m: float = 2.0) -> list:
    """Per-zone EXTRUDED hulls in the camera's calibration frame: the convex
    hull of the zone polygon projected at the floor (z=0) AND at ``height_m``.

    This is the mask-clip boundary. The flat floor polygon cuts the body off
    a tall object (a mask rises above its floor footprint) and a field-
    clipped projection under-covers an edge-on zone (0% mask survival on
    genuinely in-zone objects); the zones' crop-box RECTS over-cover (their
    union spans most of the frame — masks visibly outside the zones again).
    The hull is tight laterally like the polygon but tall enough for the
    objects standing in the zone. Same projection as ``zone_crop_boxes``
    (pinhole-fallback, no field clip). Returns ``[(zone_id, name, hull)]``.
    """
    from backbone.detection.zone_scope import _project_world3
    from backbone.shared.geometry import (
        densify_polygon,
        floor_to_pixel,
        has_metric_camera_model,
    )
    if rig is None or camera_id not in rig:
        return []
    view = rig[camera_id]
    out: list = []
    for name in zones.names:
        poly_m = densify_polygon(
            np.asarray(zones[name].polygon, dtype=np.float64), segments_per_edge=8)
        if has_metric_camera_model(view.K, view.R, view.t):
            floor3 = np.hstack([poly_m, np.zeros((len(poly_m), 1))])
            top3 = np.hstack([poly_m, np.full((len(poly_m), 1), float(height_m))])
            uv = np.vstack([
                _project_world3(floor3, view.K, view.D, view.R, view.t,
                                view.image_size_wh),
                _project_world3(top3, view.K, view.D, view.R, view.t,
                                view.image_size_wh),
            ])
            uv = uv[~np.isnan(uv).any(axis=1)]
        else:
            # Mode 1 (H only): floor polygon + the same polygon shifted up by
            # its own pixel height — the zone_crop_boxes stand-in.
            base = floor_to_pixel(poly_m, view.H)
            lift = base.copy()
            lift[:, 1] -= (base[:, 1].max() - base[:, 1].min())
            uv = np.vstack([base, lift])
        if len(uv) < 3:
            continue
        # An edge zone's z=2m lift can explode via the pinhole fallback
        # (measured a hull 3000x the crop rect). Drop only the numerically
        # unhinged points (beyond 8x the frame), hull the rest, then take the
        # EXACT convex intersection with the frame rectangle — the visible
        # footprint stays correct however far off-frame the tops project
        # (a tight margin filter here collapsed oblique cameras' hulls to
        # floor-only, re-clipping tall objects' masks).
        w, h = float(view.image_size_wh[0]), float(view.image_size_wh[1])
        sane = uv[(np.abs(uv[:, 0]) < 8 * w) & (np.abs(uv[:, 1]) < 8 * h)]
        if len(sane) < 3:
            continue
        hull = cv2.convexHull(sane.astype(np.float32))
        frame_rect = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
                              dtype=np.float32).reshape(-1, 1, 2)
        _area, clipped = cv2.intersectConvexConvex(hull, frame_rect)
        if clipped is None or len(clipped) < 3:
            continue
        out.append((zones.id_of(name) or name, name,
                    clipped.reshape(-1, 2).astype(np.float64)))
    return out


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
