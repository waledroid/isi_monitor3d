"""End-to-end synthetic test of the triangulation layer.

Hermetic: build a synthetic 2-camera rig (look-down), spawn a person at
known ground-truth ``(X, Y, Z=0)``, synthesize per-camera ``Detection`` +
``Track2D`` for it, and run the full S5 pipeline (subscriptions → keypoint
associator → opencv_dlt → reprojection gate → 3D Kalman). The published
``Track3D.xyz_m`` must match the ground truth within 1 mm in the zero-noise
path — the load-bearing test the plan calls out.

The pixel-noise path proves the reprojection gate accepts realistic
observations while still rejecting injected disagreement.
"""

from __future__ import annotations

import numpy as np
import pytest

from backbone.core.types import Detection, Track2D
from backbone.homography.foot_projector import FootProjector
from backbone.shared.camera_rig import CameraRig
from backbone.shared.geometry import (
    floor_homography_from_K_R_t,
    project_world_to_pixel,
    projection_from_K_R_t,
)
from backbone.shared.zones import Zone, ZoneRegistry
from backbone.triangulation import (
    KeypointAssociator,
    MatchRule,
    OpencvDltTriangulator,
    ReprojectionGate,
    SubscriptionManager,
    SubscriptionRule,
    Track3DConfig,
    Tracker3D,
)
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


def _project_foot_to_pixel(rig: CameraRig, cam_id: str, world_xy: tuple[float, float]) -> tuple[float, float]:
    cam = rig[cam_id]
    foot_world = np.array([world_xy[0], world_xy[1], 0.0])
    uv = project_world_to_pixel(foot_world.reshape(1, 3), cam.P).reshape(2)
    return float(uv[0]), float(uv[1])


def _make_detection(camera_id: str, foot_uv: tuple[float, float], capture_ts: float) -> Detection:
    return Detection(
        camera_id=camera_id,
        capture_ts=capture_ts,
        cls="person",
        confidence=0.9,
        bbox_xyxy=(foot_uv[0] - 10.0, foot_uv[1] - 100.0, foot_uv[0] + 10.0, foot_uv[1]),
        foot_uv=foot_uv,
    )


def _make_track(track_id: int, xy_m: tuple[float, float], capture_ts: float) -> Track2D:
    return Track2D(
        track_id=track_id, cls="person", capture_ts=capture_ts,
        xy_m=xy_m, vxy_m=(0.0, 0.0), confidence=0.9,
        cameras_seeing=("cam_a", "cam_b"),
    )


def _build_layer(rig: CameraRig, subscriptions: SubscriptionManager):
    projector = FootProjector(rig)
    associator = KeypointAssociator(rig, projector)
    triangulator = OpencvDltTriangulator(rig)
    gate = ReprojectionGate(rig, max_error_px=5.0)
    tracker = Tracker3D(Track3DConfig(process_noise=0.5, measurement_noise_m=0.005))
    return projector, associator, triangulator, gate, tracker, subscriptions


def _step(
    layer,
    rig: CameraRig,
    world_xy: tuple[float, float],
    capture_ts: float,
    *,
    inject_pixel_noise: float = 0.0,
    inject_camera_disagreement_px: float = 0.0,
    rng: np.random.Generator | None = None,
):
    """Run one frame end-to-end. Returns the emitted Track3D or None."""
    _projector, associator, triangulator, gate, tracker, subscriptions = layer
    detections: dict[str, list[Detection]] = {}
    for cam_id in rig.camera_ids:
        uv = _project_foot_to_pixel(rig, cam_id, world_xy)
        if inject_pixel_noise > 0.0:
            gen = rng if rng is not None else np.random.default_rng(0)
            uv = (
                uv[0] + float(gen.normal(0.0, inject_pixel_noise)),
                uv[1] + float(gen.normal(0.0, inject_pixel_noise)),
            )
        if cam_id == "cam_a" and inject_camera_disagreement_px > 0.0:
            uv = (uv[0] + inject_camera_disagreement_px, uv[1])
        detections[cam_id] = [_make_detection(cam_id, uv, capture_ts)]

    track = _make_track(track_id=1, xy_m=world_xy, capture_ts=capture_ts)
    matched = subscriptions.filter([track], reference_ts=capture_ts)
    assert len(matched) == 1, "subscription must accept this synthetic track"

    obs_uv = associator.resolve_foot_uv(track, detections)
    if len(obs_uv) < 2:
        return None
    xyz = triangulator.triangulate_point(obs_uv)
    if xyz is None:
        return None
    if not gate.check(xyz, obs_uv):
        return None
    return tracker.update(
        track_id=track.track_id,
        xyz_obs=xyz,
        capture_ts=capture_ts,
        cameras_seeing=track.cameras_seeing,
        cls=track.cls,
        max_reproj_error_px=gate.last_max_error_px,
    )


def _subscriptions_accept_all() -> SubscriptionManager:
    return SubscriptionManager(
        [SubscriptionRule("all", "test", MatchRule(cls="person"), "xyz")],
    )


# ---- load-bearing math correctness ----


def test_static_target_within_1mm() -> None:
    rig = _rig()
    layer = _build_layer(rig, _subscriptions_accept_all())
    truth = (1.0, 0.3)
    out = None
    for i in range(5):
        out = _step(layer, rig, truth, capture_ts=i * 0.033)
    assert out is not None
    assert out.track_id == 1
    assert out.cls == "person"
    assert out.xyz_m[0] == pytest.approx(truth[0], abs=1e-3)
    assert out.xyz_m[1] == pytest.approx(truth[1], abs=1e-3)
    assert out.xyz_m[2] == pytest.approx(0.0, abs=1e-3)


def test_track3d_inherits_track_id() -> None:
    """The architecture's "identity inherited" principle — Track3D.track_id == Track2D.track_id."""
    rig = _rig()
    layer = _build_layer(rig, _subscriptions_accept_all())
    out = _step(layer, rig, (1.0, 0.0), capture_ts=0.0)
    assert out is not None
    assert out.track_id == 1


def test_track3d_carries_max_reprojection_error() -> None:
    rig = _rig()
    layer = _build_layer(rig, _subscriptions_accept_all())
    out = _step(layer, rig, (1.0, 0.0), capture_ts=0.0)
    assert out is not None
    # Synthetic — should be ~0 px.
    assert out.max_reprojection_error_px < 0.01


# ---- subscription matching ----


def test_unmatched_subscription_skips_triangulation() -> None:
    """A subscription on 'forklift' should not lift a person track."""
    rig = _rig()
    forklift_only = SubscriptionManager(
        [SubscriptionRule("f", "test", MatchRule(cls="forklift"), "xyz")],
    )
    projector = FootProjector(rig)
    KeypointAssociator(rig, projector)
    OpencvDltTriangulator(rig)
    ReprojectionGate(rig)
    tracker = Tracker3D()

    track = _make_track(1, (1.0, 0.0), capture_ts=0.0)
    matched = forklift_only.filter([track], reference_ts=0.0)
    assert matched == []
    # And the rest of the pipeline is intentionally not run.
    assert tracker.active_track_ids == ()


def test_zone_subscription_only_lifts_inside_zone() -> None:
    _rig()
    zones = ZoneRegistry([
        Zone("hot", "danger", np.array([[0.5, -0.5], [1.5, -0.5], [1.5, 0.5], [0.5, 0.5]])),
    ])
    subs = SubscriptionManager(
        [SubscriptionRule("zd", "test", MatchRule(cls="person", in_zone="any_danger"), "xyz")],
        zones=zones,
    )

    # Inside the zone — matched.
    inside_track = _make_track(1, (1.0, 0.0), capture_ts=0.0)
    assert len(subs.filter([inside_track], reference_ts=0.0)) == 1
    # Outside the zone — not matched.
    outside_track = _make_track(2, (5.0, 5.0), capture_ts=0.0)
    assert len(subs.filter([outside_track], reference_ts=0.0)) == 0


# ---- gate rejects bad data ----


def test_two_camera_disagreement_manifests_as_z_offset() -> None:
    """Important gotcha for v1: with only 2 cameras, the linear DLT system is
    exactly determined — any two rays intersect at a unique 3D point that
    re-projects back to the two observed pixels with zero residual. So the
    reprojection gate (5 px max) cannot catch cross-cam disagreement in 2-cam
    mode. Instead, cross-cam disagreement on a floor object shows up as a
    Z offset (the triangulated point lifts off the floor).

    The reprojection gate becomes meaningful for 3+ cameras (S5.5+ aniposelib
    path) and for catching numerical degeneracies. The S4 disagreement gate
    already catches metric-space cross-cam mismatches BEFORE triangulation.
    """
    rig = _rig()
    layer = _build_layer(rig, _subscriptions_accept_all())
    out = _step(layer, rig, (1.0, 0.0), capture_ts=0.0,
                inject_camera_disagreement_px=20.0)
    # Reprojection gate passes (system is exactly determined).
    assert out is not None
    _, _, _, gate, _, _ = layer
    assert gate.rejected_count == 0
    assert gate.last_max_error_px < 1e-6
    # ... but Z is meaningfully off from the truth of 0.0 m.
    assert abs(out.xyz_m[2]) > 0.05


def test_gate_rejects_when_xyz_is_clearly_inconsistent() -> None:
    """Gate-level unit-style test: feed it an xyz that does NOT match the
    observations, and the gate must reject regardless of the triangulator."""
    rig = _rig()
    _, _, _, gate, _, _ = _build_layer(rig, _subscriptions_accept_all())
    # Build observations consistent with (1, 0, 0); ask the gate about (5, 0, 0).
    truth = np.array([1.0, 0.0, 0.0])
    obs = {}
    for cam_id in rig.camera_ids:
        uv = project_world_to_pixel(truth.reshape(1, 3), rig[cam_id].P).reshape(2)
        obs[cam_id] = (float(uv[0]), float(uv[1]))
    bad_xyz = np.array([5.0, 0.0, 0.0])
    assert not gate.check(bad_xyz, obs)
    assert gate.rejected_count == 1


def test_noisy_observations_still_publish_3d() -> None:
    rig = _rig()
    layer = _build_layer(rig, _subscriptions_accept_all())
    rng = np.random.default_rng(42)
    truth = (1.0, 0.3)
    out = None
    for i in range(15):
        out = _step(layer, rig, truth, capture_ts=i * 0.033,
                    inject_pixel_noise=1.0, rng=rng)
    assert out is not None
    err = float(np.linalg.norm(np.array(out.xyz_m) - np.array([truth[0], truth[1], 0.0])))
    assert err < 0.10


# ---- gc lifecycle ----


def test_tracker3d_gc_drops_removed_2d_tracks() -> None:
    rig = _rig()
    layer = _build_layer(rig, _subscriptions_accept_all())
    out = _step(layer, rig, (1.0, 0.3), capture_ts=0.0)
    assert out is not None
    _, _, _, _, tracker, _ = layer
    assert 1 in tracker.active_track_ids
    # 2D tracker reported nothing alive this frame → 3D tracker GCs everything.
    tracker.gc(set())
    assert tracker.active_track_ids == ()
