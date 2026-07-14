"""Zone-scoped cam view: project floor zones → clip detections → dashed outline.

The core guarantee (a zone-based system): a detection is shown ONLY when its
foot point falls inside a zone polygon — nothing outside the zone, even though
isistream detects inside the zone's larger bounding-box crop.
"""

from types import SimpleNamespace

import numpy as np

from monitor_web.zone_projection import (
    clip_to_zones,
    draw_zone_outlines,
    project_zone_polygons,
    scale_polygons,
    zone_of_point,
)


def _det(cls="palette", foot=(50.0, 50.0)):
    return SimpleNamespace(cls=cls, confidence=0.9, foot_uv=foot,
                           bbox_xyxy=(foot[0] - 10, foot[1] - 20, foot[0] + 10, foot[1]))


# a single square zone (0,0)-(100,100) in calibration px
_SQUARE = [("z1", "Zone 1", np.array([[0, 0], [100, 0], [100, 100], [0, 100]], float))]


def test_detection_inside_zone_is_kept_and_tagged():
    d = _det(foot=(50.0, 50.0))
    out = clip_to_zones([d], _SQUARE)
    assert out == [d]
    assert d.zone_id == "z1"


def test_detection_outside_zone_is_dropped():
    """The whole point — an object in the crop margin but OUTSIDE the zone
    polygon must not appear on the cam view."""
    out = clip_to_zones([_det(foot=(150.0, 150.0))], _SQUARE)
    assert out == []


def test_no_zones_shows_nothing():
    """Zone-based: with no zone polygons, nothing is drawn (fail closed)."""
    assert clip_to_zones([_det(), _det(foot=(10.0, 10.0))], []) == []


def test_boundary_point_counts_as_inside():
    # cv2.pointPolygonTest >= 0 includes the edge.
    assert zone_of_point((0.0, 50.0), _SQUARE) == "z1"


def test_scale_polygons_matches_display_frame():
    scaled = scale_polygons(_SQUARE, 2.0, 3.0)
    assert np.allclose(scaled[0][2], [[0, 0], [200, 0], [200, 300], [0, 300]])
    # a foot that was outside at calib scale can be inside after scaling
    assert zone_of_point((150.0, 250.0), scaled) == "z1"


def test_draw_outline_marks_pixels_without_crashing():
    img = np.zeros((120, 120, 3), dtype=np.uint8)
    draw_zone_outlines(img, _SQUARE, color=(0, 220, 255))
    assert img.any()   # something was drawn


def test_project_zone_polygons_absent_camera_is_empty():
    rig = {"cam_a": object()}   # supports __contains__

    class _Reg:
        pass
    assert project_zone_polygons(rig, _Reg(), "cam_zzz") == []
    assert project_zone_polygons(None, _Reg(), "cam_a") == []


# ---- metric membership (clip_to_zones_metric) -------------------------------


class _LookDownView:
    """Synthetic look-down camera at 3m: u = 1000*X/3 + 500 (no distortion)."""

    def __init__(self):
        from backbone.shared.geometry import floor_homography_from_K_R_t
        self.K = np.array([[1000.0, 0.0, 500.0], [0.0, 1000.0, 500.0], [0.0, 0.0, 1.0]])
        self.D = np.zeros(5)
        self.R = np.diag([1.0, -1.0, -1.0])
        self.t = np.array([0.0, 0.0, 3.0])
        self.H = floor_homography_from_K_R_t(self.K, self.R, self.t)
        self.image_size_wh = (1000, 1000)


def _floor_to_px(x, y):
    return (1000.0 * x / 3.0 + 500.0, -1000.0 * y / 3.0 + 500.0)


def _mzones(poly):
    from backbone.shared.zones import ZoneRegistry
    return ZoneRegistry.from_dict(
        {"zones": [{"id": "z1", "name": "Z1", "type": "palette", "polygon": poly}]})


def test_metric_clip_keeps_inside_and_near_boundary():
    """Membership is in METRES: a foot 0.1m outside (per-camera projection
    skew, measured 0.05-0.11m on the rig) is kept; 0.5m-out junk is dropped."""
    from monitor_web.zone_projection import clip_to_zones_metric
    view = _LookDownView()
    zones = _mzones([[0.5, -0.5], [1.5, -0.5], [1.5, 0.5], [0.5, 0.5]])
    def det(fx, fy):
        u, v = _floor_to_px(fx, fy)
        return SimpleNamespace(cls="palette", confidence=0.9, foot_uv=(u, v),
                               bbox_xyxy=(u - 10, v - 20, u + 10, v))
    inside, near, junk = det(1.0, 0.0), det(1.6, 0.0), det(2.2, 0.0)
    kept = clip_to_zones_metric([inside, near, junk], view, (1000, 1000), zones)
    assert kept == [inside, near]
    assert inside.zone_id == "z1" and near.zone_id == "z1"


def test_metric_clip_scales_display_to_calibration():
    """A half-size display frame (500x500 vs the 1000x1000 calibration) still
    projects the foot to the right metres."""
    from monitor_web.zone_projection import clip_to_zones_metric
    view = _LookDownView()
    zones = _mzones([[0.5, -0.5], [1.5, -0.5], [1.5, 0.5], [0.5, 0.5]])
    u, v = _floor_to_px(1.0, 0.0)
    d = SimpleNamespace(cls="palette", confidence=0.9, foot_uv=(u / 2.0, v / 2.0),
                        bbox_xyxy=(0, 0, 10, 10))
    assert clip_to_zones_metric([d], view, (500, 500), zones) == [d]


def test_metric_clip_no_zones_shows_nothing():
    from backbone.shared.zones import ZoneRegistry

    from monitor_web.zone_projection import clip_to_zones_metric
    d = SimpleNamespace(cls="palette", confidence=0.9, foot_uv=(500.0, 500.0),
                        bbox_xyxy=(0, 0, 10, 10))
    assert clip_to_zones_metric([d], _LookDownView(), (1000, 1000),
                                ZoneRegistry.empty()) == []


def test_crop_box_stencil_covers_boxes_scaled():
    from monitor_web.zone_projection import crop_box_stencil
    st = crop_box_stencil((100, 200), [(100, 100, 300, 300)], (0.5, 0.5))
    assert st[75, 100] == 255      # inside the scaled box (50..150 x 50..150)
    assert st[75, 25] == 0         # outside
