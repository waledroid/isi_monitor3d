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
