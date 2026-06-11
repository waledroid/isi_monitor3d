"""Pure geometric helpers used by both the homography and triangulation layers.

Stateless, NumPy/OpenCV in, NumPy out. No I/O, no logging — these functions
are called per detection at high frequency and any sneaky side effect would
show up in the latency budget.

Conventions:
    * Pixel coordinates: (u, v) in pixels, origin top-left, +u right, +v down.
    * World coordinates: (X, Y, Z) in meters; the floor plane is Z = 0.
    * "Floor (X, Y)" means dropping Z from the world frame.
    * Distortion model: OpenCV plumb-bob (k1, k2, p1, p2[, k3, ...]).
"""

from __future__ import annotations

import cv2
import numpy as np


def undistort_points(
    points_uv: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
) -> np.ndarray:
    """Correct pixel coordinates for lens distortion.

    Skipping this step costs several centimeters of metric accuracy at frame
    edges — non-negotiable for the ≤2 px reprojection KPI.

    Args:
        points_uv: ``(N, 2)`` array of pixel coordinates.
        K: 3x3 intrinsic matrix.
        D: distortion coefficients (k1, k2, p1, p2, k3, ...).

    Returns:
        ``(N, 2)`` array of undistorted pixel coordinates (still in the same
        pixel frame, i.e. multiplied back by ``K``).
    """
    pts = np.asarray(points_uv, dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.undistortPoints(pts, K, D, P=K)
    return out.reshape(-1, 2)


def pixel_to_floor(
    points_uv: np.ndarray,
    H: np.ndarray,
) -> np.ndarray:
    """Project pixels to floor coordinates (X, Y) in meters via the homography.

    Expects ``points_uv`` to be already undistorted — apply :func:`undistort_points`
    first. The pipeline is: detection → bottom-center of bbox → undistort → here.

    Args:
        points_uv: ``(N, 2)`` undistorted pixel coordinates.
        H: 3x3 homography matrix, pixel → world (meters).

    Returns:
        ``(N, 2)`` floor coordinates in meters.
    """
    pts = np.asarray(points_uv, dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, H)
    return out.reshape(-1, 2)


def floor_to_pixel(
    points_xy_m: np.ndarray,
    H: np.ndarray,
) -> np.ndarray:
    """Project floor (X, Y) meters back to pixels. Used by the dashboard overlay."""
    H_inv = np.linalg.inv(H)
    pts = np.asarray(points_xy_m, dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, H_inv)
    return out.reshape(-1, 2)


def project_world_to_pixel(
    points_xyz_m: np.ndarray,
    P: np.ndarray,
) -> np.ndarray:
    """Project 3D world points to pixels with the full projection matrix.

    Used by the reprojection-error gate after triangulation. The gate is the
    only thing standing between bad input and silent bad 3D output.

    Args:
        points_xyz_m: ``(N, 3)`` world points in meters.
        P: 3x4 projection matrix.

    Returns:
        ``(N, 2)`` pixel coordinates.
    """
    pts = np.asarray(points_xyz_m, dtype=np.float64)
    homog = np.hstack([pts, np.ones((pts.shape[0], 1))])
    proj = homog @ P.T
    return proj[:, :2] / proj[:, 2:3]


def reprojection_error_px(
    point_xyz_m: np.ndarray,
    observations_uv: dict[str, np.ndarray],
    P_by_camera: dict[str, np.ndarray],
) -> dict[str, float]:
    """Per-camera reprojection error, in pixels, for a single 3D point.

    Returns a dict keyed by camera_id. The reprojection gate rejects the
    triangulation if ``max(values()) > threshold``.
    """
    errors: dict[str, float] = {}
    for cam_id, obs_uv in observations_uv.items():
        P = P_by_camera[cam_id]
        proj = project_world_to_pixel(point_xyz_m.reshape(1, 3), P).reshape(2)
        errors[cam_id] = float(np.linalg.norm(np.asarray(obs_uv, dtype=np.float64) - proj))
    return errors


def fundamental_from_projections(P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
    """Compute the fundamental matrix F such that x2.T @ F @ x1 = 0.

    Derived directly from the two 3x4 projection matrices — no need for the
    eight-point algorithm or live observations. Used by the keypoint
    associator's epipolar fallback (left/right limb disambiguation).
    """

    def _drop_row(M: np.ndarray, row: int) -> np.ndarray:
        return np.delete(M, row, axis=0)

    F = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            M = np.vstack([_drop_row(P1, i), _drop_row(P2, j)])
            F[j, i] = ((-1) ** (i + j)) * np.linalg.det(M)
    return F


def epipolar_line(point_uv: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Epipolar line ``(a, b, c)`` in the second view such that ``a u + b v + c = 0``.

    ``point_uv`` is in the first view.
    """
    u, v = float(point_uv[0]), float(point_uv[1])
    line = F @ np.array([u, v, 1.0])
    return line


def point_line_distance_px(point_uv: np.ndarray, line_abc: np.ndarray) -> float:
    """Perpendicular pixel distance from a point to an epipolar line."""
    a, b, c = line_abc
    u, v = float(point_uv[0]), float(point_uv[1])
    denom = float(np.hypot(a, b))
    if denom == 0.0:
        return float("inf")
    return abs(a * u + b * v + c) / denom


def projection_from_K_R_t(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Compose ``P = K @ [R | t]`` — used during calibration export.

    Note: ``R, t`` are world ← camera in this codebase. The projection matrix
    that maps world points to pixels needs the inverse: ``R^T, -R^T t``.
    """
    R_cw = R.T
    t_cw = -R_cw @ t.reshape(3)
    Rt = np.hstack([R_cw, t_cw.reshape(3, 1)])
    return K @ Rt


def floor_homography_from_K_R_t(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Compose the floor-plane homography (pixel → world meters at Z=0).

    Given camera projection columns ``[r1 r2 r3 | t]`` of ``[R_cw | t_cw]``,
    the floor homography (world → pixel) is ``K @ [r1 r2 t_cw]``. Inverting
    it yields pixel → world. Returned in the pixel→world direction so it
    drops directly into :func:`pixel_to_floor`.
    """
    R_cw = R.T
    t_cw = -R_cw @ t.reshape(3)
    H_world_to_pixel = K @ np.column_stack([R_cw[:, 0], R_cw[:, 1], t_cw])
    return np.linalg.inv(H_world_to_pixel)
