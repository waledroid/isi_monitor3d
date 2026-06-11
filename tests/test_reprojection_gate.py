"""``ReprojectionGate`` — accept on agreement, reject above threshold."""

from __future__ import annotations

import numpy as np
import pytest

from backbone.shared.camera_rig import CameraRig
from backbone.shared.geometry import (
    floor_homography_from_K_R_t,
    project_world_to_pixel,
    projection_from_K_R_t,
)
from backbone.triangulation.reprojection_gate import ReprojectionGate
from calibration.schema import (
    CALIBRATION_VERSION,
    CalibrationFile,
    CameraCalibration,
)

K = np.array([[1000.0, 0.0, 500.0], [0.0, 1000.0, 500.0], [0.0, 0.0, 1.0]])
R_LOOK_DOWN = np.diag([1.0, -1.0, -1.0])


def _camera_cal(camera_id: str, position_xy: tuple[float, float]) -> CameraCalibration:
    t = np.array([position_xy[0], position_xy[1], 3.0])
    return CameraCalibration(
        camera_id=camera_id,
        image_size_wh=(1000, 1000),
        K=K.tolist(),
        D=[0.0, 0.0, 0.0, 0.0, 0.0],
        R=R_LOOK_DOWN.tolist(),
        t=t.tolist(),
        H=floor_homography_from_K_R_t(K, R_LOOK_DOWN, t).tolist(),
        P=projection_from_K_R_t(K, R_LOOK_DOWN, t).tolist(),
        reprojection_rms_px=0.1,
    )


def _rig() -> CameraRig:
    return CameraRig(CalibrationFile(
        version=CALIBRATION_VERSION,
        created_at="2026-05-18T00:00:00Z",
        floor_anchor_method="synthetic",
        floor_origin_note="test",
        cameras={
            "cam_a": _camera_cal("cam_a", (0.0, 0.0)),
            "cam_b": _camera_cal("cam_b", (2.0, 0.0)),
        },
    ))


def _perfect_obs(xyz: np.ndarray, rig: CameraRig) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for cam_id in rig.camera_ids:
        uv = project_world_to_pixel(xyz.reshape(1, 3), rig[cam_id].P).reshape(2)
        out[cam_id] = (float(uv[0]), float(uv[1]))
    return out


def test_perfect_observations_pass_gate() -> None:
    rig = _rig()
    gate = ReprojectionGate(rig, max_error_px=5.0)
    xyz = np.array([1.0, 0.5, 0.0])
    assert gate.check(xyz, _perfect_obs(xyz, rig))
    assert gate.rejected_count == 0
    assert gate.last_max_error_px == pytest.approx(0.0, abs=1e-6)


def test_camera_disagreement_above_threshold_rejected() -> None:
    rig = _rig()
    gate = ReprojectionGate(rig, max_error_px=5.0)
    xyz = np.array([1.0, 0.5, 0.0])
    obs = _perfect_obs(xyz, rig)
    obs["cam_a"] = (obs["cam_a"][0] + 20.0, obs["cam_a"][1])  # 20 px shift
    assert not gate.check(xyz, obs)
    assert gate.rejected_count == 1
    assert gate.last_max_error_px > 5.0


def test_disagreement_within_threshold_accepted() -> None:
    rig = _rig()
    gate = ReprojectionGate(rig, max_error_px=5.0)
    xyz = np.array([1.0, 0.5, 0.0])
    obs = _perfect_obs(xyz, rig)
    obs["cam_a"] = (obs["cam_a"][0] + 3.0, obs["cam_a"][1])  # 3 px — within tolerance
    assert gate.check(xyz, obs)
    assert gate.rejected_count == 0
    assert gate.last_max_error_px > 0


def test_custom_threshold_respected() -> None:
    rig = _rig()
    strict = ReprojectionGate(rig, max_error_px=1.0)
    lax = ReprojectionGate(rig, max_error_px=10.0)
    xyz = np.array([1.0, 0.5, 0.0])
    obs = _perfect_obs(xyz, rig)
    obs["cam_a"] = (obs["cam_a"][0] + 5.0, obs["cam_a"][1])
    assert not strict.check(xyz, obs)
    assert lax.check(xyz, obs)


def test_invalid_threshold_rejected() -> None:
    rig = _rig()
    with pytest.raises(ValueError, match="positive"):
        ReprojectionGate(rig, max_error_px=0.0)


def test_rejected_count_accumulates() -> None:
    rig = _rig()
    gate = ReprojectionGate(rig, max_error_px=1.0)
    xyz = np.array([1.0, 0.5, 0.0])
    obs = _perfect_obs(xyz, rig)
    obs["cam_a"] = (obs["cam_a"][0] + 50.0, obs["cam_a"][1])
    for _ in range(4):
        gate.check(xyz, obs)
    assert gate.rejected_count == 4
