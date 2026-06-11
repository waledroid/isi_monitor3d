"""Hermetic end-to-end test of the homography layer.

Builds a synthetic 2-camera rig (look-down geometry), spawns "people" at
known world (X, Y) coordinates, projects them through each camera into pixel
coordinates, wraps them in ``Detection`` objects, and runs the full S4
pipeline. The published ``Track2D.xy_m`` must match the ground-truth world
positions within a tight tolerance — this is the load-bearing test the plan
calls out as "produces metric tracks within 1 mm of synthetic ground truth".

Real on-site E2E against tape-measured ground truth is deferred; this proves
the pipeline math end-to-end without needing a recording.
"""

from __future__ import annotations

import numpy as np
import pytest

from backbone.core.types import Detection
from backbone.homography import (
    ByteTrackMeters,
    CrossCamFusion,
    DisagreementGate,
    FootProjector,
    TemporalStabilizer,
    TrackConfig,
)
from backbone.shared.camera_rig import CameraRig
from backbone.shared.geometry import (
    floor_homography_from_K_R_t,
    project_world_to_pixel,
    projection_from_K_R_t,
)
from calibration.schema import (
    CALIBRATION_VERSION,
    CalibrationFile,
    CameraCalibration,
)

K = np.array([[1000.0, 0.0, 500.0], [0.0, 1000.0, 500.0], [0.0, 0.0, 1.0]])
R_LOOK_DOWN = np.diag([1.0, -1.0, -1.0])


def _camera_cal(camera_id: str, position_xy: tuple[float, float], z: float = 3.0) -> CameraCalibration:
    t = np.array([position_xy[0], position_xy[1], z])
    P = projection_from_K_R_t(K, R_LOOK_DOWN, t)
    H = floor_homography_from_K_R_t(K, R_LOOK_DOWN, t)
    return CameraCalibration(
        camera_id=camera_id,
        image_size_wh=(1000, 1000),
        K=K.tolist(),
        D=[0.0, 0.0, 0.0, 0.0, 0.0],
        R=R_LOOK_DOWN.tolist(),
        t=t.tolist(),
        H=H.tolist(),
        P=P.tolist(),
        reprojection_rms_px=0.1,
    )


def _two_camera_rig() -> CameraRig:
    """Two cameras: cam_a at world origin, cam_b 2 m east — both looking down."""
    file = CalibrationFile(
        version=CALIBRATION_VERSION,
        created_at="2026-05-16T00:00:00Z",
        floor_anchor_method="synthetic",
        floor_origin_note="test",
        cameras={
            "cam_a": _camera_cal("cam_a", (0.0, 0.0)),
            "cam_b": _camera_cal("cam_b", (2.0, 0.0)),
        },
    )
    return CameraRig(file)


def _project_to_pixel(world_xyz: np.ndarray, camera: object) -> tuple[float, float]:
    P = np.asarray(camera.P, dtype=np.float64)
    uv = project_world_to_pixel(world_xyz.reshape(1, 3), P).reshape(2)
    return float(uv[0]), float(uv[1])


def _detection_from_world(
    rig: CameraRig,
    camera_id: str,
    world_xy: tuple[float, float],
    capture_ts: float,
    cls: str = "person",
    confidence: float = 0.9,
    pixel_noise: float = 0.0,
    rng: np.random.Generator | None = None,
) -> Detection:
    """Synthesize a Detection whose foot_uv exactly projects the given world (X, Y)."""
    cam = rig[camera_id]
    foot_world = np.array([world_xy[0], world_xy[1], 0.0])
    foot_uv = _project_to_pixel(foot_world, cam)
    if pixel_noise > 0.0:
        gen = rng if rng is not None else np.random.default_rng(0)
        foot_uv = (
            foot_uv[0] + float(gen.normal(0.0, pixel_noise)),
            foot_uv[1] + float(gen.normal(0.0, pixel_noise)),
        )
    return Detection(
        camera_id=camera_id,
        capture_ts=capture_ts,
        cls=cls,
        confidence=confidence,
        bbox_xyxy=(foot_uv[0] - 10.0, foot_uv[1] - 100.0, foot_uv[0] + 10.0, foot_uv[1]),
        foot_uv=foot_uv,
    )


def _build_pipeline(rig: CameraRig) -> tuple[FootProjector, CrossCamFusion, DisagreementGate, ByteTrackMeters, TemporalStabilizer]:
    projector = FootProjector(rig)
    fusion = CrossCamFusion()
    gate = DisagreementGate()
    tracker = ByteTrackMeters(
        track_config=TrackConfig(
            min_hits_to_confirm=1,
            class_history_window=5,
            measurement_noise_m=0.01,
        ),
    )
    stabilizer = TemporalStabilizer(tracker, min_frames_confirmed=1)
    return projector, fusion, gate, tracker, stabilizer


def _step(
    pipeline: tuple,
    rig: CameraRig,
    capture_ts: float,
    world_positions: list[tuple[float, float]],
    pixel_noise: float = 0.0,
    rng: np.random.Generator | None = None,
) -> list:
    """Run one frame end-to-end. Returns the published Track2D list."""
    projector, fusion, gate, tracker, stabilizer = pipeline
    all_dets = []
    for camera_id in rig.camera_ids:
        for xy in world_positions:
            all_dets.append(_detection_from_world(rig, camera_id, xy, capture_ts, pixel_noise=pixel_noise, rng=rng))
    pairs_with_floor = projector.project_batch(all_dets)
    fused = fusion.fuse(pairs_with_floor)
    gated = gate.check(fused)
    obs_tuples = [(o.cls, o.xy_m, o.confidence, o.cameras_seeing) for o in gated]
    raw_tracks = tracker.update(capture_ts, obs_tuples)
    return stabilizer.stabilize(raw_tracks)


# ---------- zero-noise (math correctness) ----------


def test_static_person_tracked_to_ground_truth_zero_noise() -> None:
    rig = _two_camera_rig()
    pipeline = _build_pipeline(rig)
    truth = (1.0, 0.3)
    last_out: list = []
    for i in range(5):
        last_out = _step(pipeline, rig, capture_ts=i * 0.033, world_positions=[truth])
    assert len(last_out) == 1
    x, y = last_out[0].xy_m
    assert x == pytest.approx(truth[0], abs=1e-3)  # 1 mm
    assert y == pytest.approx(truth[1], abs=1e-3)


def test_two_static_people_get_two_stable_tracks() -> None:
    rig = _two_camera_rig()
    pipeline = _build_pipeline(rig)
    truths = [(0.5, 0.5), (1.5, -0.5)]
    last_out: list = []
    for i in range(10):
        last_out = _step(pipeline, rig, capture_ts=i * 0.033, world_positions=truths)
    assert len(last_out) == 2
    ids = {t.track_id for t in last_out}
    assert len(ids) == 2
    # Positions match truth — assignment may swap order so match by nearest.
    emitted = sorted([(t.xy_m[0], t.xy_m[1]) for t in last_out])
    expected = sorted(truths)
    for got, exp in zip(emitted, expected, strict=True):
        assert got[0] == pytest.approx(exp[0], abs=1e-3)
        assert got[1] == pytest.approx(exp[1], abs=1e-3)


def test_moving_person_tracked_with_velocity_estimate() -> None:
    rig = _two_camera_rig()
    pipeline = _build_pipeline(rig)
    # Walks +X at 0.5 m/s.
    last_out: list = []
    for i in range(20):
        ts = i * 0.033
        truth = (0.0 + 0.5 * ts, 0.0)
        last_out = _step(pipeline, rig, capture_ts=ts, world_positions=[truth])
    assert len(last_out) == 1
    vx, vy = last_out[0].vxy_m
    # Velocity estimate should be near 0.5 m/s in X, near 0 in Y. Wide tolerance
    # because the Kalman needs a few frames to converge on velocity.
    assert vx == pytest.approx(0.5, abs=0.05)
    assert abs(vy) < 0.05


# ---------- noise robustness ----------


def test_noisy_detections_still_within_10cm() -> None:
    """2 px Gaussian pixel noise on every detection → tracker output within 10 cm."""
    rig = _two_camera_rig()
    pipeline = _build_pipeline(rig)
    rng = np.random.default_rng(42)
    truth = (1.0, 0.3)
    last_out: list = []
    for i in range(15):
        last_out = _step(
            pipeline, rig, capture_ts=i * 0.033, world_positions=[truth],
            pixel_noise=2.0, rng=rng,
        )
    assert len(last_out) == 1
    x, y = last_out[0].xy_m
    err = float(np.hypot(x - truth[0], y - truth[1]))
    assert err < 0.10
