"""``OpencvDltTriangulator`` — verify the linear DLT against known geometry."""

from __future__ import annotations

import numpy as np
import pytest

from backbone.core.interfaces import triangulator_registry
from backbone.shared.camera_rig import CameraRig
from backbone.shared.geometry import (
    floor_homography_from_K_R_t,
    project_world_to_pixel,
    projection_from_K_R_t,
)
from backbone.triangulation.opencv_dlt import OpencvDltTriangulator
from calibration.schema import (
    CALIBRATION_VERSION,
    CalibrationFile,
    CameraCalibration,
)

K = np.array([[1000.0, 0.0, 500.0], [0.0, 1000.0, 500.0], [0.0, 0.0, 1.0]])
R_LOOK_DOWN = np.diag([1.0, -1.0, -1.0])


def _camera_cal(camera_id: str, position_xy: tuple[float, float], z: float = 3.0) -> CameraCalibration:
    t = np.array([position_xy[0], position_xy[1], z])
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


def test_plugin_registered_under_opencv_dlt() -> None:
    import backbone.triangulation  # noqa: F401

    assert "opencv_dlt" in triangulator_registry


def test_triangulate_recovers_known_point() -> None:
    rig = _rig()
    tri = OpencvDltTriangulator(rig)
    truth = np.array([0.5, 0.3, 0.4])    # 0.4 m above the floor

    obs = {}
    for cam_id in rig.camera_ids:
        cam = rig[cam_id]
        uv = project_world_to_pixel(truth.reshape(1, 3), cam.P).reshape(2)
        obs[cam_id] = (float(uv[0]), float(uv[1]))

    xyz = tri.triangulate_point(obs)
    assert xyz is not None
    np.testing.assert_allclose(xyz, truth, atol=1e-6)


def test_triangulate_floor_point() -> None:
    """Z = 0 (a foot on the floor) — must come back near zero."""
    rig = _rig()
    tri = OpencvDltTriangulator(rig)
    truth = np.array([1.0, 0.2, 0.0])
    obs = {}
    for cam_id in rig.camera_ids:
        uv = project_world_to_pixel(truth.reshape(1, 3), rig[cam_id].P).reshape(2)
        obs[cam_id] = (float(uv[0]), float(uv[1]))
    xyz = tri.triangulate_point(obs)
    assert xyz is not None
    np.testing.assert_allclose(xyz, truth, atol=1e-5)


def test_too_few_cameras_returns_none() -> None:
    rig = _rig()
    tri = OpencvDltTriangulator(rig)
    assert tri.triangulate_point({"cam_a": (500.0, 500.0)}) is None
    assert tri.triangulate_point({}) is None


def test_three_plus_cameras_raises() -> None:
    rig = CameraRig(CalibrationFile(
        version=CALIBRATION_VERSION,
        created_at="2026-05-18T00:00:00Z",
        floor_anchor_method="synthetic",
        floor_origin_note="test",
        cameras={
            "cam_a": _camera_cal("cam_a", (0.0, 0.0)),
            "cam_b": _camera_cal("cam_b", (2.0, 0.0)),
            "cam_c": _camera_cal("cam_c", (0.0, 2.0)),
        },
    ))
    tri = OpencvDltTriangulator(rig)
    obs = {cam_id: (500.0, 500.0) for cam_id in rig.camera_ids}
    with pytest.raises(NotImplementedError, match=r"S5\.5"):
        tri.triangulate_point(obs)


