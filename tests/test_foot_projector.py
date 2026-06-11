"""``FootProjector`` — Detection → (X, Y) m via undistort + homography.

Reuses the synthetic camera fixtures from ``test_geometry.py``: a 1000x1000
camera at world (0, 0, 3) looking straight down. With zero distortion and
the camera's homography, an image-centre foot pixel must project to (0, 0).
"""

from __future__ import annotations

import numpy as np

from backbone.core.types import Detection
from backbone.homography.foot_projector import FootProjector
from backbone.shared.camera_rig import CameraRig
from backbone.shared.geometry import (
    floor_homography_from_K_R_t,
    projection_from_K_R_t,
)
from calibration.schema import (
    CALIBRATION_VERSION,
    CalibrationFile,
    CameraCalibration,
)

K = np.array([[1000.0, 0.0, 500.0], [0.0, 1000.0, 500.0], [0.0, 0.0, 1.0]])
R_LOOK_DOWN = np.diag([1.0, -1.0, -1.0])


def _calibration_for(xy: tuple[float, float], z: float = 3.0) -> CameraCalibration:
    t = np.array([xy[0], xy[1], z])
    P = projection_from_K_R_t(K, R_LOOK_DOWN, t)
    H = floor_homography_from_K_R_t(K, R_LOOK_DOWN, t)
    return CameraCalibration(
        camera_id="cam_a",
        image_size_wh=(1000, 1000),
        K=K.tolist(),
        D=[0.0, 0.0, 0.0, 0.0, 0.0],
        R=R_LOOK_DOWN.tolist(),
        t=t.tolist(),
        H=H.tolist(),
        P=P.tolist(),
        reprojection_rms_px=0.1,
    )


def _rig_at(xy: tuple[float, float], cam_id: str = "cam_a") -> CameraRig:
    base = _calibration_for(xy)
    cal = CameraCalibration(
        camera_id=cam_id,
        image_size_wh=base.image_size_wh,
        K=base.K,
        D=base.D,
        R=base.R,
        t=base.t,
        H=base.H,
        P=base.P,
        reprojection_rms_px=base.reprojection_rms_px,
    )
    file = CalibrationFile(
        version=CALIBRATION_VERSION,
        created_at="2026-05-16T00:00:00Z",
        floor_anchor_method="synthetic",
        floor_origin_note="test",
        cameras={cam_id: cal},
    )
    return CameraRig(file)


def _det(camera_id: str, foot_uv: tuple[float, float], cls: str = "person") -> Detection:
    return Detection(
        camera_id=camera_id,
        capture_ts=0.0,
        cls=cls,
        confidence=0.9,
        bbox_xyxy=(foot_uv[0] - 5.0, foot_uv[1] - 50.0, foot_uv[0] + 5.0, foot_uv[1]),
        foot_uv=foot_uv,
    )


def test_image_centre_projects_to_world_origin() -> None:
    rig = _rig_at((0.0, 0.0))
    projector = FootProjector(rig)
    det = _det("cam_a", (500.0, 500.0))
    x, y = projector.project(det)
    assert x == pytest.approx(0.0, abs=1e-6)
    assert y == pytest.approx(0.0, abs=1e-6)


def test_off_axis_pixel_matches_known_geometry() -> None:
    """At f=1000, h=3, 100 px horizontal offset → 0.3 m world X."""
    rig = _rig_at((0.0, 0.0))
    projector = FootProjector(rig)
    det = _det("cam_a", (600.0, 500.0))
    x, y = projector.project(det)
    assert x == pytest.approx(0.3, abs=1e-6)
    assert y == pytest.approx(0.0, abs=1e-6)


def test_camera_offset_projects_to_camera_footprint() -> None:
    """Camera at (2, 0, 3) sees its own footprint at image centre → world (2, 0)."""
    rig = _rig_at((2.0, 0.0))
    projector = FootProjector(rig)
    det = _det("cam_a", (500.0, 500.0))
    x, y = projector.project(det)
    assert x == pytest.approx(2.0, abs=1e-6)
    assert y == pytest.approx(0.0, abs=1e-6)


def test_project_batch_preserves_order_and_pairs_detections() -> None:
    rig = _rig_at((0.0, 0.0))
    projector = FootProjector(rig)
    dets = [
        _det("cam_a", (500.0, 500.0)),
        _det("cam_a", (600.0, 500.0)),
        _det("cam_a", (400.0, 500.0)),
    ]
    out = projector.project_batch(dets)
    assert [d for d, _ in out] == dets
    xs = [xy[0] for _, xy in out]
    assert xs == pytest.approx([0.0, 0.3, -0.3], abs=1e-6)


def test_unknown_camera_id_raises_keyerror() -> None:
    rig = _rig_at((0.0, 0.0))
    projector = FootProjector(rig)
    det = _det("cam_ghost", (500.0, 500.0))
    with pytest.raises(KeyError, match="cam_ghost"):
        projector.project(det)


# pytest must be imported for the approx machinery used in the asserts above.
import pytest  # noqa: E402
