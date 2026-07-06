"""``calibration.refine`` — rigid floor-alignment fit + extrinsic bake."""

from __future__ import annotations

import numpy as np
import pytest

from backbone.shared.geometry import (
    floor_homography_from_K_R_t,
    pixel_to_floor,
    projection_from_K_R_t,
    undistort_points,
)
from calibration.refine import apply_floor_alignment, fit_rigid_floor_alignment
from calibration.schema import CALIBRATION_VERSION, CalibrationFile, CameraCalibration

K = np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]])
R_LOOK_DOWN = np.diag([1.0, -1.0, -1.0])


def _rot2(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def test_fit_recovers_known_rigid_transform() -> None:
    rng = np.random.default_rng(7)
    target = rng.uniform(-2, 2, size=(6, 2))
    theta, t = np.radians(3.5), np.array([0.12, -0.08])
    ref = target @ _rot2(theta).T + t

    fit = fit_rigid_floor_alignment(target, ref)
    assert fit.theta_rad == pytest.approx(theta, abs=1e-9)
    assert fit.tx_m == pytest.approx(t[0], abs=1e-9)
    assert fit.ty_m == pytest.approx(t[1], abs=1e-9)
    assert fit.max_residual_m < 1e-9


def test_fit_residuals_expose_bad_pairs() -> None:
    """A mismatched pair (operator clicked the wrong spot) shows up as a big
    residual — the gate the UI refuses on."""
    target = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    ref = target + np.array([0.1, 0.0])
    ref[2] += [0.5, 0.4]                    # one bad correspondence
    fit = fit_rigid_floor_alignment(target, ref)
    assert fit.max_residual_m > 0.15


def test_fit_never_returns_reflection() -> None:
    target = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    ref = target[:, ::-1]                   # mirrored points try to force det<0
    fit = fit_rigid_floor_alignment(target, ref)
    R = _rot2(fit.theta_rad)
    assert np.linalg.det(R) == pytest.approx(1.0)


def _cal_file(offset_cam_b: tuple[float, float] = (0.0, 0.0),
              yaw_cam_b_deg: float = 0.0) -> CalibrationFile:
    """A 2-cam file; cam_b's pose optionally CORRUPTED by a rigid floor error
    (the thing the fine-tune corrects)."""
    def cam(cid, center, extra_yaw=0.0, extra_t=(0.0, 0.0)):
        c, s = np.cos(extra_yaw), np.sin(extra_yaw)
        T = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        R = T @ R_LOOK_DOWN
        t = T @ np.array([center[0], center[1], 2.5]) + [extra_t[0], extra_t[1], 0.0]
        return CameraCalibration(
            camera_id=cid, image_size_wh=(1920, 1080),
            K=K.tolist(), D=[0.0] * 5, R=R.tolist(), t=t.tolist(),
            H=floor_homography_from_K_R_t(K, R, t).tolist(),
            P=projection_from_K_R_t(K, R, t).tolist(),
            reprojection_rms_px=0.5,
        )
    return CalibrationFile(
        version=CALIBRATION_VERSION, created_at="2026-07-03T00:00:00Z",
        floor_anchor_method="synthetic", floor_origin_note="test",
        cameras={
            "cam_a": cam("cam_a", (0.0, 0.0)),
            "cam_b": cam("cam_b", (2.0, 0.0), np.radians(yaw_cam_b_deg), offset_cam_b),
        },
    )


def test_apply_alignment_collapses_cross_camera_error() -> None:
    """End-to-end: corrupt cam_b by a rigid floor error, fit from point pairs
    (as the tool does), apply — the cross-camera disagreement collapses to ~0
    and H/P stay mutually consistent."""
    corrupted = _cal_file(offset_cam_b=(0.10, -0.06), yaw_cam_b_deg=2.0)
    truth = _cal_file()   # what cam_b SHOULD be

    def author(calfile, cid, world_pts):
        """world → pixels via the TRUE geometry, back to world via calfile."""
        cam_true = truth.cameras[cid]
        P = np.asarray(cam_true.P)
        w3 = np.hstack([world_pts, np.ones((len(world_pts), 1)) * 0.0])
        pix = (P @ np.hstack([w3, np.ones((len(w3), 1))]).T).T
        pix = pix[:, :2] / pix[:, 2:3]
        cam = calfile.cameras[cid]
        return pixel_to_floor(
            undistort_points(pix, np.asarray(cam.K), np.asarray(cam.D)),
            np.asarray(cam.H),
        )

    pts = np.array([[0.5, -0.5], [1.5, -0.5], [1.5, 0.5], [0.5, 0.5]])
    xa = author(corrupted, "cam_a", pts)          # reference (identity here)
    xb = author(corrupted, "cam_b", pts)          # corrupted mapping
    assert np.linalg.norm(xa - xb, axis=1).max() > 0.05   # visibly misaligned

    fit = fit_rigid_floor_alignment(xb, xa)
    refined = apply_floor_alignment(corrupted, "cam_b", fit)

    xb2 = author(refined, "cam_b", pts)
    assert np.linalg.norm(xa - xb2, axis=1).max() < 1e-6  # aligned

    # H and P moved TOGETHER: P re-projects a floor point to the pixel that H
    # maps back to it.
    cam = refined.cameras["cam_b"]
    P = np.asarray(cam.P)
    w = np.array([1.0, 0.2, 0.0, 1.0])
    pix = P @ w
    pix = pix[:2] / pix[2]
    back = pixel_to_floor(np.asarray([pix]), np.asarray(cam.H))[0]
    assert np.allclose(back, w[:2], atol=1e-9)

    # cam_a untouched; base file unmodified.
    assert refined.cameras["cam_a"].R == corrupted.cameras["cam_a"].R
    assert corrupted.cameras["cam_b"].H != refined.cameras["cam_b"].H


def test_apply_alignment_unknown_camera_rejected() -> None:
    with pytest.raises(ValueError, match="cam_z"):
        apply_floor_alignment(_cal_file(), "cam_z",
                              fit_rigid_floor_alignment(
                                  np.zeros((2, 2)), np.zeros((2, 2))))
