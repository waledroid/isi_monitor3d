"""Geometry helpers — verified against a synthetic camera with known truth.

The synthetic rig:
    Camera A at world (0, 0, 3) looking straight down (-Z).
    Camera B at world (2, 0, 3) looking straight down (-Z).
    Both: focal 1000 px, principal point (500, 500), 1000x1000 image,
    no distortion.

For a camera looking straight down, ``R_world_from_camera = diag(1, -1, -1)``
(camera +X → world +X, camera +Y → world -Y, camera +Z forward → world -Z).
"""

from __future__ import annotations

import numpy as np
import pytest

from backbone.shared.geometry import (
    epipolar_line,
    floor_homography_from_K_R_t,
    floor_to_pixel,
    fundamental_from_projections,
    pixel_to_floor,
    point_line_distance_px,
    project_world_to_pixel,
    projection_from_K_R_t,
    reprojection_error_px,
    undistort_points,
)

K = np.array([[1000.0, 0.0, 500.0], [0.0, 1000.0, 500.0], [0.0, 0.0, 1.0]])
R_LOOK_DOWN = np.diag([1.0, -1.0, -1.0])
ZERO_D = np.zeros(5)


def _camera_at(xy: tuple[float, float], z: float = 3.0):
    t = np.array([xy[0], xy[1], z])
    P = projection_from_K_R_t(K, R_LOOK_DOWN, t)
    H = floor_homography_from_K_R_t(K, R_LOOK_DOWN, t)
    return P, H


# ---------- undistort ----------


def test_undistort_zero_distortion_is_identity() -> None:
    pts = np.array([[100.0, 200.0], [500.0, 500.0], [900.0, 100.0]])
    out = undistort_points(pts, K, ZERO_D)
    np.testing.assert_allclose(out, pts, atol=1e-6)


# ---------- pixel ↔ floor round-trip ----------


def test_pixel_to_floor_origin_at_image_center() -> None:
    _P, H = _camera_at((0.0, 0.0))
    xy = pixel_to_floor(np.array([[500.0, 500.0]]), H)
    np.testing.assert_allclose(xy, [[0.0, 0.0]], atol=1e-6)


def test_pixel_to_floor_camera_b_offset() -> None:
    """Camera B at (2, 0, 3) — the world origin is below cam A, so cam B sees
    the world origin offset to the left in its image. Verifies the homography
    correctly accounts for camera translation."""
    _P, H = _camera_at((2.0, 0.0))
    xy = pixel_to_floor(np.array([[500.0, 500.0]]), H)
    # cam B's image center is its own footprint on the floor → world (2, 0).
    np.testing.assert_allclose(xy, [[2.0, 0.0]], atol=1e-6)


def test_pixel_floor_round_trip() -> None:
    _P, H = _camera_at((0.0, 0.0))
    pts_uv = np.array([[100.0, 200.0], [500.0, 500.0], [800.0, 750.0]])
    xy = pixel_to_floor(pts_uv, H)
    back = floor_to_pixel(xy, H)
    np.testing.assert_allclose(back, pts_uv, atol=1e-6)


# ---------- 3D projection ----------


def test_project_world_origin_is_image_center() -> None:
    P, _H = _camera_at((0.0, 0.0))
    uv = project_world_to_pixel(np.array([[0.0, 0.0, 0.0]]), P)
    np.testing.assert_allclose(uv, [[500.0, 500.0]], atol=1e-6)


def test_project_off_axis_floor_point() -> None:
    """Looking down from (0, 0, 3), a floor point at world (0.3, 0, 0) projects
    to pixel (0.3 * 1000/3 + 500, 500) = (600, 500)."""
    P, _H = _camera_at((0.0, 0.0))
    uv = project_world_to_pixel(np.array([[0.3, 0.0, 0.0]]), P)
    np.testing.assert_allclose(uv, [[600.0, 500.0]], atol=1e-6)


def test_homography_and_projection_agree_on_floor() -> None:
    """A floor point projected via H must match its 3D projection (Z=0) via P."""
    P, H = _camera_at((1.0, 0.5))
    xy = np.array([[0.3, -0.4]])
    via_H = floor_to_pixel(xy, H)
    via_P = project_world_to_pixel(np.column_stack([xy, np.zeros(1)]), P)
    np.testing.assert_allclose(via_H, via_P, atol=1e-6)


# ---------- reprojection error ----------


def test_reprojection_error_zero_for_perfect_observation() -> None:
    P_a, _H_a = _camera_at((0.0, 0.0))
    P_b, _H_b = _camera_at((2.0, 0.0))
    world_pt = np.array([0.5, 0.3, 0.0])
    uv_a = project_world_to_pixel(world_pt.reshape(1, 3), P_a).reshape(2)
    uv_b = project_world_to_pixel(world_pt.reshape(1, 3), P_b).reshape(2)
    errs = reprojection_error_px(
        world_pt,
        {"cam_a": uv_a, "cam_b": uv_b},
        {"cam_a": P_a, "cam_b": P_b},
    )
    assert max(errs.values()) < 1e-6


def test_reprojection_error_increases_with_offset() -> None:
    P_a, _H_a = _camera_at((0.0, 0.0))
    P_b, _H_b = _camera_at((2.0, 0.0))
    world_pt = np.array([0.5, 0.3, 0.0])
    uv_a = project_world_to_pixel(world_pt.reshape(1, 3), P_a).reshape(2)
    uv_b = project_world_to_pixel(world_pt.reshape(1, 3), P_b).reshape(2)
    # Add 5 px of noise to cam_a's observation.
    uv_a_noisy = uv_a + np.array([5.0, 0.0])
    errs = reprojection_error_px(
        world_pt,
        {"cam_a": uv_a_noisy, "cam_b": uv_b},
        {"cam_a": P_a, "cam_b": P_b},
    )
    assert errs["cam_a"] == pytest.approx(5.0, abs=1e-6)
    assert errs["cam_b"] == pytest.approx(0.0, abs=1e-6)


# ---------- epipolar geometry ----------


def test_fundamental_satisfies_correspondence() -> None:
    P_a, _H_a = _camera_at((0.0, 0.0))
    P_b, _H_b = _camera_at((2.0, 0.0))
    F = fundamental_from_projections(P_a, P_b)
    world_pts = np.array([[0.5, 0.3, 0.0], [-0.4, 0.2, 0.0], [0.1, -0.6, 0.0]])
    for w in world_pts:
        ua = project_world_to_pixel(w.reshape(1, 3), P_a).reshape(2)
        ub = project_world_to_pixel(w.reshape(1, 3), P_b).reshape(2)
        constraint = np.array([ub[0], ub[1], 1.0]) @ F @ np.array([ua[0], ua[1], 1.0])
        assert abs(constraint) < 1e-6


def test_epipolar_line_passes_through_correspondence() -> None:
    P_a, _H_a = _camera_at((0.0, 0.0))
    P_b, _H_b = _camera_at((2.0, 0.0))
    F = fundamental_from_projections(P_a, P_b)
    w = np.array([0.5, 0.3, 0.0])
    ua = project_world_to_pixel(w.reshape(1, 3), P_a).reshape(2)
    ub = project_world_to_pixel(w.reshape(1, 3), P_b).reshape(2)
    line = epipolar_line(ua, F)
    assert point_line_distance_px(ub, line) < 1e-6
