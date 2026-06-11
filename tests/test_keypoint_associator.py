"""``KeypointAssociator`` — pick the per-camera Detection that matches a Track2D.

Uses the same synthetic look-down rig as the earlier geometry tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from backbone.core.types import Detection, Track2D
from backbone.homography.foot_projector import FootProjector
from backbone.shared.camera_rig import CameraRig
from backbone.shared.geometry import (
    floor_homography_from_K_R_t,
    projection_from_K_R_t,
)
from backbone.triangulation.keypoint_associator import KeypointAssociator
from calibration.schema import (
    CALIBRATION_VERSION,
    CalibrationFile,
    CameraCalibration,
)

K_INTRINSIC = np.array([[1000.0, 0.0, 500.0], [0.0, 1000.0, 500.0], [0.0, 0.0, 1.0]])
R_LOOK_DOWN = np.diag([1.0, -1.0, -1.0])


def _camera_cal(camera_id: str, position_xy: tuple[float, float], z: float = 3.0) -> CameraCalibration:
    t = np.array([position_xy[0], position_xy[1], z])
    P = projection_from_K_R_t(K_INTRINSIC, R_LOOK_DOWN, t)
    H = floor_homography_from_K_R_t(K_INTRINSIC, R_LOOK_DOWN, t)
    return CameraCalibration(
        camera_id=camera_id,
        image_size_wh=(1000, 1000),
        K=K_INTRINSIC.tolist(),
        D=[0.0, 0.0, 0.0, 0.0, 0.0],
        R=R_LOOK_DOWN.tolist(),
        t=t.tolist(),
        H=H.tolist(),
        P=P.tolist(),
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


def _det(camera_id: str, foot_uv: tuple[float, float], cls: str = "person", conf: float = 0.9) -> Detection:
    return Detection(
        camera_id=camera_id,
        capture_ts=0.0,
        cls=cls,
        confidence=conf,
        bbox_xyxy=(foot_uv[0] - 5.0, foot_uv[1] - 100.0, foot_uv[0] + 5.0, foot_uv[1]),
        foot_uv=foot_uv,
    )


def _track(xy_m: tuple[float, float], cameras_seeing: tuple[str, ...] = ("cam_a", "cam_b")) -> Track2D:
    return Track2D(
        track_id=1,
        cls="person",
        capture_ts=0.0,
        xy_m=xy_m,
        vxy_m=(0.0, 0.0),
        confidence=0.9,
        cameras_seeing=cameras_seeing,
    )


def test_associator_picks_only_candidate_per_camera() -> None:
    rig = _rig()
    associator = KeypointAssociator(rig, FootProjector(rig))
    # cam_a image centre maps to world (0, 0); cam_b image centre maps to world (2, 0).
    # Track at world (0, 0) — cam_a sees centre, cam_b sees (-2, 0) m offset.
    detections = {
        "cam_a": [_det("cam_a", (500.0, 500.0))],
        "cam_b": [_det("cam_b", (-1000.0 + 500.0, 500.0))],   # off-frame but math holds
    }
    # cam_b's image-centre footprint is world (2, 0); for world (0,0) cam_b's foot is
    # at u = (0 - 2) * 1000 / 3 + 500 = -666.67 + 500 = -166.67.
    detections["cam_b"] = [_det("cam_b", (-166.6667, 500.0))]
    out = associator.resolve_foot_uv(_track(xy_m=(0.0, 0.0)), detections)
    assert set(out) == {"cam_a", "cam_b"}
    assert out["cam_a"] == (500.0, 500.0)
    assert out["cam_b"][0] == pytest.approx(-166.6667, abs=1e-3)


def test_associator_picks_nearest_when_multiple_candidates() -> None:
    rig = _rig()
    associator = KeypointAssociator(rig, FootProjector(rig))
    detections = {
        "cam_a": [
            _det("cam_a", (200.0, 200.0)),   # projects to (-0.9, 0.9) m
            _det("cam_a", (500.0, 500.0)),   # projects to (0, 0) m — the right one
            _det("cam_a", (800.0, 800.0)),   # projects to (0.9, -0.9) m
        ],
        "cam_b": [_det("cam_b", (-166.6667, 500.0))],
    }
    out = associator.resolve_foot_uv(_track(xy_m=(0.0, 0.0)), detections)
    assert out["cam_a"] == (500.0, 500.0)


def test_associator_ignores_wrong_class() -> None:
    rig = _rig()
    associator = KeypointAssociator(rig, FootProjector(rig))
    detections = {
        "cam_a": [_det("cam_a", (500.0, 500.0), cls="forklift")],  # right position, wrong class
        "cam_b": [_det("cam_b", (-166.6667, 500.0))],
    }
    out = associator.resolve_foot_uv(_track(xy_m=(0.0, 0.0)), detections)
    assert "cam_a" not in out
    assert "cam_b" in out


def test_associator_omits_camera_when_no_close_match() -> None:
    rig = _rig()
    associator = KeypointAssociator(rig, FootProjector(rig), max_match_distance_m=0.5)
    detections = {
        "cam_a": [_det("cam_a", (800.0, 800.0))],  # ~1.3 m from track world position
        "cam_b": [_det("cam_b", (-166.6667, 500.0))],
    }
    out = associator.resolve_foot_uv(_track(xy_m=(0.0, 0.0)), detections)
    assert "cam_a" not in out


def test_associator_ignores_cameras_not_in_track_cameras_seeing() -> None:
    rig = _rig()
    associator = KeypointAssociator(rig, FootProjector(rig))
    track = _track(xy_m=(0.0, 0.0), cameras_seeing=("cam_a",))
    detections = {
        "cam_a": [_det("cam_a", (500.0, 500.0))],
        "cam_b": [_det("cam_b", (-166.6667, 500.0))],
    }
    out = associator.resolve_foot_uv(track, detections)
    assert set(out) == {"cam_a"}


