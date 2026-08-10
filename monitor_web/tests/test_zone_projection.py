"""Zone-scoped cam view: project floor zones → clip detections → dashed outline.

The core guarantee (a zone-based system): a detection is shown ONLY when its
foot point falls inside a zone polygon — nothing outside the zone, even though
isistream detects inside the zone's larger bounding-box crop.

Plane-aware (zone-base-height, decision 5): every projection/membership
function tests a zone on ITS OWN PLANE (``Zone.z_base_m``) via
``backbone.shared.zones.ZoneAwareProjector`` — the same helper the Backbone's
``zone_scope.build_zone_membership_filter`` and ``PalletStateManager`` use.
"""

from types import SimpleNamespace

import numpy as np

from monitor_web.zone_projection import (
    draw_zone_outlines,
    project_zone_polygons,
    scale_polygons,
)


def _det(cls="palette", foot=(50.0, 50.0)):
    return SimpleNamespace(cls=cls, confidence=0.9, foot_uv=foot,
                           bbox_xyxy=(foot[0] - 10, foot[1] - 20, foot[0] + 10, foot[1]))


# a single square zone (0,0)-(100,100) in calibration px
_SQUARE = [("z1", "Zone 1", np.array([[0, 0], [100, 0], [100, 100], [0, 100]], float))]




def test_scale_polygons_scales_vertices():
    scaled = scale_polygons(_SQUARE, 2.0, 3.0)
    assert np.allclose(scaled[0][2], [[0, 0], [200, 0], [200, 300], [0, 300]])


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


# ---- metric membership (clip_to_zones_metric / zone_of_foot_metric) --------
#
# Rig geometry mirrors tests/test_zone_scope.py (backbone side): a look-down
# camera at the origin, z=3, f=1000, c=(500, 500) — world (X, Y, z) projects
# to u = 1000*X/(3-z) + 500. Raising the plane brings it closer to the
# camera, so a raised zone's footprint MAGNIFIES outward vs its z=0 projection.

_PLATFORM_Z = 0.304          # the live sortie_machine_1 platform height


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


class _ViewH:
    """Mode-1 placeholder extrinsics — only ``H`` is real (K=I, R=I, t=0)."""

    def __init__(self):
        from backbone.shared.geometry import floor_homography_from_K_R_t
        K = np.array([[1000.0, 0.0, 500.0], [0.0, 1000.0, 500.0], [0.0, 0.0, 1.0]])
        R_look_down = np.diag([1.0, -1.0, -1.0])
        self.K = np.eye(3)
        self.D = np.zeros(5)
        self.R = np.eye(3)
        self.t = np.zeros(3)
        self.H = floor_homography_from_K_R_t(K, R_look_down, np.array([0.0, 0.0, 3.0]))
        self.image_size_wh = (1000, 1000)


class _FakeRig:
    def __init__(self, views: dict):
        self._views = views

    @property
    def camera_ids(self):
        return tuple(self._views)

    def __getitem__(self, cam_id):
        return self._views[cam_id]

    def __contains__(self, cam_id):
        return cam_id in self._views


def _floor_to_px(x, y):
    return (1000.0 * x / 3.0 + 500.0, -1000.0 * y / 3.0 + 500.0)


def _mzones(poly, z_base_m: float = 0.0):
    from backbone.shared.zones import ZoneRegistry
    return ZoneRegistry.from_dict(
        {"zones": [{"id": "z1", "name": "Z1", "type": "palette", "polygon": poly,
                    "z_base_m": z_base_m}]})


def test_metric_clip_keeps_inside_and_near_boundary():
    """Membership is in METRES: a foot 0.1m outside (per-camera projection
    skew, measured 0.05-0.11m on the rig) is kept; 0.5m-out junk is dropped."""
    from monitor_web.zone_projection import clip_to_zones_metric
    rig = _FakeRig({"cam_a": _LookDownView()})
    zones = _mzones([[0.5, -0.5], [1.5, -0.5], [1.5, 0.5], [0.5, 0.5]])
    def det(fx, fy):
        u, v = _floor_to_px(fx, fy)
        return SimpleNamespace(cls="palette", confidence=0.9, foot_uv=(u, v),
                               bbox_xyxy=(u - 10, v - 20, u + 10, v))
    inside, near, junk = det(1.0, 0.0), det(1.6, 0.0), det(2.2, 0.0)
    kept = clip_to_zones_metric([inside, near, junk], rig, "cam_a", (1000, 1000), zones)
    assert kept == [inside, near]
    assert inside.zone_id == "z1" and near.zone_id == "z1"


def test_metric_clip_scales_display_to_calibration():
    """A half-size display frame (500x500 vs the 1000x1000 calibration) still
    projects the foot to the right metres."""
    from monitor_web.zone_projection import clip_to_zones_metric
    rig = _FakeRig({"cam_a": _LookDownView()})
    zones = _mzones([[0.5, -0.5], [1.5, -0.5], [1.5, 0.5], [0.5, 0.5]])
    u, v = _floor_to_px(1.0, 0.0)
    d = SimpleNamespace(cls="palette", confidence=0.9, foot_uv=(u / 2.0, v / 2.0),
                        bbox_xyxy=(0, 0, 10, 10))
    assert clip_to_zones_metric([d], rig, "cam_a", (500, 500), zones) == [d]


def test_metric_clip_no_zones_shows_nothing():
    from backbone.shared.zones import ZoneRegistry

    from monitor_web.zone_projection import clip_to_zones_metric
    d = SimpleNamespace(cls="palette", confidence=0.9, foot_uv=(500.0, 500.0),
                        bbox_xyxy=(0, 0, 10, 10))
    rig = _FakeRig({"cam_a": _LookDownView()})
    assert clip_to_zones_metric([d], rig, "cam_a", (1000, 1000),
                                ZoneRegistry.empty()) == []


def test_metric_clip_unknown_camera_shows_nothing():
    from monitor_web.zone_projection import clip_to_zones_metric
    d = SimpleNamespace(cls="palette", confidence=0.9, foot_uv=(500.0, 500.0),
                        bbox_xyxy=(0, 0, 10, 10))
    rig = _FakeRig({"cam_a": _LookDownView()})
    zones = _mzones([[0.5, -0.5], [1.5, -0.5], [1.5, 0.5], [0.5, 0.5]])
    assert clip_to_zones_metric([d], rig, "cam_zzz", (1000, 1000), zones) == []


def test_zone_hull_covers_height_but_stays_lateral():
    """The mask stencil comes from the EXTRUDED hull: it must reach ABOVE the
    flat floor polygon (an object's mask rises above its footprint) yet stay
    far tighter laterally than the zone's bounding-rect crop box."""
    from monitor_web.zone_projection import project_zone_hulls

    rig = _FakeRig({"cam_a": _LookDownView()})
    zones = _mzones([[0.3, -0.3], [0.9, -0.3], [0.9, 0.3], [0.3, 0.3]])
    (zid, _name, hull), = project_zone_hulls(rig, zones, "cam_a")
    assert zid == "z1"
    # Floor footprint u-range: 600..800 px; the z=2m projection (depth 1m)
    # stretches u to 1000*x/1+500 = 800..1400 → the hull must extend well
    # beyond the floor-only footprint, but stay clipped to the frame.
    floor_u_max = 1000.0 * 0.9 / 3.0 + 500.0
    assert hull[:, 0].max() > floor_u_max + 100
    assert hull[:, 0].max() <= 999.0 and hull[:, 1].min() >= 0.0


# ---- plane-aware projection & membership (z_base_m) ------------------------


def test_membership_accepts_platform_detection_floor_would_misplace():
    """The live platform case, synthesized: an object at (3, 0) ON a 0.304 m
    platform. Its foot pixel floor-projects to X≈3.34 — outside the zone even
    with the ±0.15 m cross — but plane-projects to exactly (3.0, 0), inside.
    The z_base-aware clip keeps it; a floor (z=0) zone still drops it."""
    from monitor_web.zone_projection import clip_to_zones_metric

    rig = _FakeRig({"cam_a": _LookDownView()})
    poly = [[2.9, -0.1], [3.1, -0.1], [3.1, 0.1], [2.9, 0.1]]
    foot = (1000.0 * 3.0 / (3.0 - _PLATFORM_Z) + 500.0, 500.0)
    d = SimpleNamespace(cls="palette", confidence=0.9, foot_uv=foot,
                        bbox_xyxy=(0, 0, 10, 10))
    raised = _mzones(poly, _PLATFORM_Z)
    kept = clip_to_zones_metric([d], rig, "cam_a", (1000, 1000), raised)
    assert kept == [d]

    d2 = SimpleNamespace(cls="palette", confidence=0.9, foot_uv=foot,
                         bbox_xyxy=(0, 0, 10, 10))
    floor_zone = _mzones(poly, 0.0)
    assert clip_to_zones_metric([d2], rig, "cam_a", (1000, 1000), floor_zone) == []


def test_zone_of_foot_metric_accepts_platform_detection():
    """Same platform case as above, single-foot API."""
    from monitor_web.zone_projection import zone_of_foot_metric

    rig = _FakeRig({"cam_a": _LookDownView()})
    poly = [[2.9, -0.1], [3.1, -0.1], [3.1, 0.1], [2.9, 0.1]]
    foot = (1000.0 * 3.0 / (3.0 - _PLATFORM_Z) + 500.0, 500.0)
    raised = _mzones(poly, _PLATFORM_Z)
    assert zone_of_foot_metric(rig, "cam_a", (1000, 1000), raised, foot) == "z1"
    floor_zone = _mzones(poly, 0.0)
    assert zone_of_foot_metric(rig, "cam_a", (1000, 1000), floor_zone, foot) is None


def test_metric_membership_floor_zones_bit_identical_to_old_way():
    """z_base_m=0 zones keep the EXACT old undistort+H membership semantics —
    replicated here over a grid of inside/boundary/outside cases."""
    from backbone.shared.geometry import pixel_to_floor, undistort_points

    from monitor_web.zone_projection import clip_to_zones_metric, zone_of_foot_metric

    rig = _FakeRig({"cam_a": _LookDownView()})
    view = rig["cam_a"]
    zones = _mzones([[0.5, -0.5], [1.5, -0.5], [1.5, 0.5], [0.5, 0.5]])
    for foot in [(1000.0, 500.0), (1300.0, 500.0), (1600.0, 500.0), (1900.0, 500.0)]:
        px = np.asarray([foot], dtype=np.float64)
        xy = pixel_to_floor(undistort_points(px, view.K, view.D), view.H)[0]
        expected = zones["Z1"].contains((float(xy[0]), float(xy[1])))
        # 5-point ±tol cross, same as clip_to_zones_metric / build_zone_membership_filter.
        if not expected:
            for dx, dy in ((0.15, 0.0), (-0.15, 0.0), (0.0, 0.15), (0.0, -0.15)):
                if zones["Z1"].contains((float(xy[0] + dx), float(xy[1] + dy))):
                    expected = True
                    break
        d = SimpleNamespace(cls="palette", confidence=0.9, foot_uv=foot,
                            bbox_xyxy=(0, 0, 10, 10))
        kept = clip_to_zones_metric([d], rig, "cam_a", (1000, 1000), zones)
        assert (kept == [d]) is expected, f"clip diverged at {foot}"
        zid = zone_of_foot_metric(rig, "cam_a", (1000, 1000), zones, foot)
        assert (zid == "z1") is expected, f"zone_of_foot diverged at {foot}"


def test_project_zone_polygons_outline_shifts_for_raised_zone():
    """A raised zone's outline magnifies outward vs its floor projection —
    the exact plane math ``project_floor_polygon_distorted`` now threads."""
    from monitor_web.zone_projection import project_zone_polygons

    rig = _FakeRig({"cam_a": _LookDownView()})
    poly = [[0.8, -0.2], [1.2, -0.2], [1.2, 0.2], [0.8, 0.2]]
    floor_polys = project_zone_polygons(rig, _mzones(poly, 0.0), "cam_a")
    raised_polys = project_zone_polygons(rig, _mzones(poly, _PLATFORM_Z), "cam_a")
    assert len(floor_polys) == 1 and len(raised_polys) == 1
    floor_u_max = floor_polys[0][2][:, 0].max()
    raised_u_max = raised_polys[0][2][:, 0].max()
    assert raised_u_max > floor_u_max
    # Exact plane: max u = 1000*1.2/(3-0.304) + 500.
    assert abs(raised_u_max - (1000 * 1.2 / (3.0 - _PLATFORM_Z) + 500)) < 1e-3


def test_project_zone_hulls_extrudes_from_own_base_plane():
    """A raised zone's hull base sits at z_base_m (not the floor) — its lower
    edge should already be magnified outward vs the floor zone's."""
    from monitor_web.zone_projection import project_zone_hulls

    rig = _FakeRig({"cam_a": _LookDownView()})
    poly = [[0.3, -0.3], [0.9, -0.3], [0.9, 0.3], [0.3, 0.3]]
    floor_hull = project_zone_hulls(rig, _mzones(poly, 0.0), "cam_a")[0][2]
    raised_hull = project_zone_hulls(rig, _mzones(poly, _PLATFORM_Z), "cam_a")[0][2]
    assert raised_hull[:, 0].max() >= floor_hull[:, 0].max()


def test_mode1_h_only_raised_zone_keeps_floor_behavior():
    """Mode-1 rigs (H only) cannot lift a ray off the floor: a raised zone
    projects/clips/membership-tests exactly like a floor zone."""
    from monitor_web.zone_projection import (
        clip_to_zones_metric,
        project_zone_hulls,
        project_zone_polygons,
        zone_of_foot_metric,
    )

    rig = _FakeRig({"cam_a": _ViewH()})
    poly = [[0.8, -0.2], [1.2, -0.2], [1.2, 0.2], [0.8, 0.2]]
    floor_zone = _mzones(poly, 0.0)
    raised_zone = _mzones(poly, _PLATFORM_Z)

    floor_polys = project_zone_polygons(rig, floor_zone, "cam_a")
    raised_polys = project_zone_polygons(rig, raised_zone, "cam_a")
    assert np.allclose(floor_polys[0][2], raised_polys[0][2])

    floor_hull = project_zone_hulls(rig, floor_zone, "cam_a")[0][2]
    raised_hull = project_zone_hulls(rig, raised_zone, "cam_a")[0][2]
    assert np.allclose(floor_hull, raised_hull)

    foot = (850.0, 500.0)
    d_f = SimpleNamespace(cls="palette", confidence=0.9, foot_uv=foot,
                          bbox_xyxy=(0, 0, 10, 10))
    d_r = SimpleNamespace(cls="palette", confidence=0.9, foot_uv=foot,
                          bbox_xyxy=(0, 0, 10, 10))
    kept_floor = clip_to_zones_metric([d_f], rig, "cam_a", (1000, 1000), floor_zone)
    kept_raised = clip_to_zones_metric([d_r], rig, "cam_a", (1000, 1000), raised_zone)
    assert (kept_floor == [d_f]) is (kept_raised == [d_r])
    assert zone_of_foot_metric(rig, "cam_a", (1000, 1000), floor_zone, foot) == \
           zone_of_foot_metric(rig, "cam_a", (1000, 1000), raised_zone, foot)
