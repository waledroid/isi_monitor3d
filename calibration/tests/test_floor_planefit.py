"""Hermetic tests for the floor-plane world frame (Stage 2).

No rig, no camera. Synthetic floor points on a KNOWN plane in the rig frame
(arbitrary tilt/offset) + injected outliers/noise/bad-quality points exercise:

* the RANSAC plane fit recovers the plane and the world frame maps floor points
  to Z ~= 0, with an orthonormal X/Y basis and det(R) = +1;
* the quality pre-filter drops the injected high-error / low-confidence /
  bad-depth points (counts asserted);
* the emitted ``FloorAnchor`` flows through ``assemble_calibration`` (reusing the
  Stage-1 synthetic ``MultiCalSolution``) into a valid ``calibration.json`` with
  sane H/P and sensible camera translations in the plane-fit world frame.
"""

from __future__ import annotations

import numpy as np
import pytest

from calibration.calibrate import assemble_calibration, compose_camera_in_world
from calibration.floor_planefit import (
    FloorPointFilter,
    estimate_floor_anchor_planefit,
    prefilter_floor_points,
    ransac_plane_fit,
)
from calibration.schema import CalibrationFile

# --- synthetic floor plane in the rig frame ----------------------------------


def _tilted_plane_basis():
    """A known floor plane in the rig frame: tilted normal + non-zero offset.

    Normal points generally 'up' toward the rig origin (negative rig +Z, since
    the floor is in front of / below the cameras), with a deliberate tilt so the
    fit has to recover a non-axis-aligned plane.
    """
    normal = np.array([0.1, -0.85, -0.5])
    normal = normal / np.linalg.norm(normal)
    # A point the plane passes through (the floor is ~3 m ahead of the cameras).
    plane_point = np.array([0.2, 1.4, 3.0])
    return normal, plane_point


def _floor_points(rng: np.random.Generator, n: int = 300) -> np.ndarray:
    """``n`` points sampled ON the tilted floor plane, in the rig frame."""
    normal, plane_point = _tilted_plane_basis()
    # Two in-plane directions.
    a = np.cross(normal, [1.0, 0.0, 0.0])
    a = a / np.linalg.norm(a)
    b = np.cross(normal, a)
    u = rng.uniform(-2.0, 2.0, n)
    v = rng.uniform(-2.0, 2.0, n)
    return plane_point[None, :] + u[:, None] * a[None, :] + v[:, None] * b[None, :]


# --- plane fit + world frame -------------------------------------------------


def test_world_frame_maps_floor_to_z_zero() -> None:
    rng = np.random.default_rng(0)
    pts = _floor_points(rng, 300)
    pts += rng.normal(scale=0.002, size=pts.shape)  # noise 2 mm
    anchor, res = estimate_floor_anchor_planefit(pts, return_result=True)

    R, t = anchor.R_world_from_rig, anchor.t_world_from_rig
    world = pts @ R.T + t
    # Floor points land near Z=0 in world coords.
    assert np.abs(world[:, 2]).max() < 0.02
    assert res.rms_inlier_dist_m < 0.01


def test_recovered_normal_matches_plane() -> None:
    rng = np.random.default_rng(1)
    pts = _floor_points(rng, 300)
    _anchor, res = estimate_floor_anchor_planefit(pts, return_result=True)
    normal, _ = _tilted_plane_basis()
    # Recovered normal parallel to the true normal (sign chosen 'up').
    cos = abs(float(np.dot(res.normal, normal)))
    assert cos > 0.999


def test_normal_oriented_up_toward_cameras() -> None:
    rng = np.random.default_rng(2)
    pts = _floor_points(rng, 300)
    _anchor, res = estimate_floor_anchor_planefit(pts, return_result=True)
    # World +Z (= plane normal) points from the floor centroid toward the rig
    # origin (the camera), i.e. it has a positive component along (origin - centroid).
    to_camera = -res.centroid
    assert float(res.normal @ to_camera) > 0


def test_rotation_is_orthonormal_and_proper() -> None:
    rng = np.random.default_rng(3)
    pts = _floor_points(rng, 300)
    anchor = estimate_floor_anchor_planefit(pts)
    R = anchor.R_world_from_rig
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-9)


def test_origin_lies_on_the_plane() -> None:
    rng = np.random.default_rng(4)
    pts = _floor_points(rng, 300)
    anchor, res = estimate_floor_anchor_planefit(pts, return_result=True)
    # The world origin (centroid) maps to world Z ~= 0.
    world_origin = res.centroid @ anchor.R_world_from_rig.T + anchor.t_world_from_rig
    assert abs(float(world_origin[2])) < 1e-6


# --- RANSAC rejects gross outliers -------------------------------------------


def test_ransac_rejects_off_plane_outliers() -> None:
    rng = np.random.default_rng(5)
    pts = _floor_points(rng, 200)
    # Inject 40 gross outliers well off the plane (e.g. a wall / clutter).
    outliers = pts[:40] + np.array([0.0, 0.0, 1.0])  # 1 m off in rig +Z
    noisy = np.vstack([pts, outliers])
    normal, _offset, _centroid, inliers = ransac_plane_fit(noisy, threshold_m=0.03)
    # The 200 on-plane points are inliers; the 40 shifted ones are not.
    assert inliers[:200].sum() > 190
    assert inliers[200:].sum() < 5
    true_normal, _ = _tilted_plane_basis()
    assert abs(float(np.dot(normal / np.linalg.norm(normal), true_normal))) > 0.999


def test_planefit_ignores_outliers_in_world_frame() -> None:
    rng = np.random.default_rng(6)
    pts = _floor_points(rng, 200)
    outliers = pts[:30] + np.array([0.5, 0.5, 0.8])
    noisy = np.vstack([pts, outliers])
    anchor, res = estimate_floor_anchor_planefit(noisy, threshold_m=0.03, return_result=True)
    world = pts @ anchor.R_world_from_rig.T + anchor.t_world_from_rig
    # The true floor points still map to Z ~= 0 despite the outliers.
    assert np.abs(world[:, 2]).max() < 0.03
    assert res.n_outliers >= 25


# --- quality pre-filter ------------------------------------------------------


def test_prefilter_drops_high_reproj_error() -> None:
    rng = np.random.default_rng(7)
    pts = _floor_points(rng, 100)
    err = np.full(100, 0.5)
    err[:15] = 8.0  # 15 points reproject badly
    keep = prefilter_floor_points(pts, reproj_error_px=err,
                                  filt=FloorPointFilter(max_reproj_error_px=3.0))
    assert keep.sum() == 85
    assert not keep[:15].any()


def test_prefilter_drops_low_confidence() -> None:
    rng = np.random.default_rng(8)
    pts = _floor_points(rng, 100)
    conf = np.full(100, 0.9)
    conf[:20] = 0.05  # 20 low-confidence matches
    keep = prefilter_floor_points(pts, confidence=conf,
                                  filt=FloorPointFilter(min_confidence=0.2))
    assert keep.sum() == 80
    assert not keep[:20].any()


def test_prefilter_drops_bad_depth() -> None:
    rng = np.random.default_rng(9)
    pts = _floor_points(rng, 100)
    # Force implausible depths (rig +Z) on some points.
    pts[:5, 2] = -1.0      # behind the camera
    pts[5:10, 2] = 200.0   # absurdly far
    keep = prefilter_floor_points(pts, filt=FloorPointFilter(min_depth_m=0.1, max_depth_m=50.0))
    assert not keep[:10].any()
    assert keep[10:].all()


def test_estimate_reports_dropped_counts() -> None:
    rng = np.random.default_rng(10)
    pts = _floor_points(rng, 120)
    err = np.full(120, 0.5)
    err[:18] = 9.0
    conf = np.full(120, 0.8)
    conf[18:30] = 0.01
    _anchor, res = estimate_floor_anchor_planefit(
        pts, reproj_error_px=err, confidence=conf, return_result=True,
    )
    assert res.n_input == 120
    assert res.n_dropped_filter == 30       # 18 bad reproj + 12 low conf
    assert res.n_after_filter == 90


def test_too_few_after_filter_raises() -> None:
    rng = np.random.default_rng(11)
    pts = _floor_points(rng, 10)
    conf = np.zeros(10)  # everything filtered out
    with pytest.raises(RuntimeError, match="pre-filter"):
        estimate_floor_anchor_planefit(pts, confidence=conf)


def test_too_few_input_points_raises() -> None:
    with pytest.raises(ValueError, match=">=3"):
        estimate_floor_anchor_planefit(np.zeros((2, 3)))


# --- end-to-end seam: MultiCalSolution + plane-fit anchor -> calibration.json -

# Reuse the Stage-1 synthetic solution builder so the two stages compose exactly
# as they will on the rig.
from calibration.tests.test_feature_extrinsics import _full_solution  # noqa: E402


def _planefit_anchor_for_synthetic_rig():
    """Build a plane-fit anchor from floor points consistent with the Stage-1 rig.

    Stage-1's synthetic cameras sit at the rig origin looking down +Z, with the
    3D cloud at depth ~4-9 m. We place a floor plane at ~6.5 m depth (mid-cloud)
    tilted like a real down-looking rig, then feed its points to the plane fit.
    """
    rng = np.random.default_rng(20)
    # Floor roughly perpendicular to the viewing (+Z) axis, ~6.5 m ahead, small tilt.
    normal = np.array([0.05, 0.05, -1.0])
    normal = normal / np.linalg.norm(normal)
    plane_point = np.array([0.0, 0.0, 6.5])
    a = np.cross(normal, [1.0, 0.0, 0.0])
    a /= np.linalg.norm(a)
    b = np.cross(normal, a)
    u = rng.uniform(-2.0, 2.0, 200)
    v = rng.uniform(-2.0, 2.0, 200)
    pts = plane_point[None, :] + u[:, None] * a[None, :] + v[:, None] * b[None, :]
    pts += rng.normal(scale=0.003, size=pts.shape)
    return estimate_floor_anchor_planefit(pts)


def test_planefit_anchor_assembles_into_valid_calibration(tmp_path) -> None:
    res = _full_solution()
    anchor = _planefit_anchor_for_synthetic_rig()
    assert anchor.method == "planefit"

    calib = assemble_calibration(res.solution, anchor)
    assert isinstance(calib, CalibrationFile)
    assert calib.floor_anchor_method == "planefit"
    assert set(calib.cameras) == {"cam_a", "cam_b"}

    out = tmp_path / "calibration.json"
    calib.write(out)
    reloaded = CalibrationFile.read(out)
    for cam_id in ("cam_a", "cam_b"):
        cam = reloaded.cameras[cam_id]
        H = cam.H_np()
        P = cam.P_np()
        assert H.shape == (3, 3) and P.shape == (3, 4)
        assert np.isfinite(H).all() and np.isfinite(P).all()


def test_cameras_sit_above_floor_in_world_frame() -> None:
    """Both cameras' world-frame Z (height above the floor plane) is positive and sane."""
    res = _full_solution()
    anchor = _planefit_anchor_for_synthetic_rig()
    for cam_id in ("cam_a", "cam_b"):
        cam = res.solution.cameras[cam_id]
        _R_world_cam, t_world_cam = compose_camera_in_world(cam, anchor)
        # Camera centre in world coords: C = -R_world_cam.T @ t_world_cam.
        C = -_R_world_cam.T @ t_world_cam
        # The floor is ~6.5 m ahead along the look-down axis, so the cameras are
        # ~6.5 m above the world floor plane (positive Z, up).
        assert C[2] > 3.0
        assert abs(C[2]) < 12.0


def test_planefit_H_and_P_geometrically_consistent() -> None:
    from backbone.shared.geometry import pixel_to_floor, project_world_to_pixel

    res = _full_solution()
    anchor = _planefit_anchor_for_synthetic_rig()
    calib = assemble_calibration(res.solution, anchor)
    cam = calib.cameras["cam_a"]
    P = cam.P_np()
    H = cam.H_np()

    world_xy = np.array([[0.4, -0.3]])
    world_xyz = np.array([[0.4, -0.3, 0.0]])
    uv = project_world_to_pixel(world_xyz, P)
    back = pixel_to_floor(uv, H)
    np.testing.assert_allclose(back, world_xy, atol=1e-6)
