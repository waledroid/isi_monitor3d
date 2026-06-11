"""``Tracker3D`` — per-track 3D Kalman keyed by 2D ``track_id``."""

from __future__ import annotations

import numpy as np
import pytest

from backbone.triangulation.tracker_3d import Track3DConfig, Tracker3D


def _cfg() -> Track3DConfig:
    return Track3DConfig(process_noise=0.05, measurement_noise_m=0.02)


def test_new_track_id_spawns_kalman() -> None:
    tracker = Tracker3D(_cfg())
    out = tracker.update(
        track_id=42,
        xyz_obs=np.array([1.0, 0.5, 0.0]),
        capture_ts=0.0,
        cameras_seeing=("cam_a", "cam_b"),
        cls="person",
        max_reproj_error_px=0.3,
    )
    assert out.track_id == 42
    assert out.cls == "person"
    assert out.contributing_cameras == ("cam_a", "cam_b")
    assert out.max_reprojection_error_px == pytest.approx(0.3)
    assert out.xyz_m == pytest.approx((1.0, 0.5, 0.0), abs=1e-6)
    assert 42 in tracker.active_track_ids


def test_repeated_updates_share_track_id() -> None:
    tracker = Tracker3D(_cfg())
    a = tracker.update(
        track_id=7, xyz_obs=np.array([0.0, 0.0, 0.0]),
        capture_ts=0.0, cameras_seeing=("cam_a", "cam_b"),
        cls="person", max_reproj_error_px=0.5,
    )
    b = tracker.update(
        track_id=7, xyz_obs=np.array([0.1, 0.0, 0.0]),
        capture_ts=0.033, cameras_seeing=("cam_a", "cam_b"),
        cls="person", max_reproj_error_px=0.4,
    )
    assert a.track_id == b.track_id == 7
    assert len(tracker.active_track_ids) == 1


def test_kalman_converges_on_velocity() -> None:
    """Walks +X at 0.5 m/s. After ~10 frames the velocity estimate should match."""
    tracker = Tracker3D(Track3DConfig(process_noise=0.5, measurement_noise_m=0.005))
    out = None
    for i in range(20):
        ts = i * 0.033
        xyz = np.array([0.5 * ts, 0.0, 0.0])
        out = tracker.update(
            track_id=1, xyz_obs=xyz, capture_ts=ts,
            cameras_seeing=("cam_a", "cam_b"),
            cls="person", max_reproj_error_px=0.3,
        )
    assert out is not None
    vx, vy, vz = out.vxyz_m
    assert vx == pytest.approx(0.5, abs=0.05)
    assert abs(vy) < 0.05
    assert abs(vz) < 0.05


def test_gc_drops_inactive_tracks() -> None:
    tracker = Tracker3D(_cfg())
    for tid in [1, 2, 3]:
        tracker.update(
            track_id=tid, xyz_obs=np.array([1.0, 0.0, 0.0]),
            capture_ts=0.0, cameras_seeing=("cam_a", "cam_b"),
            cls="person", max_reproj_error_px=0.1,
        )
    tracker.gc({2})
    assert tracker.active_track_ids == (2,)


def test_drop_removes_one_track() -> None:
    tracker = Tracker3D(_cfg())
    tracker.update(
        track_id=5, xyz_obs=np.array([0.0, 0.0, 0.0]),
        capture_ts=0.0, cameras_seeing=("cam_a",),
        cls="person", max_reproj_error_px=0.1,
    )
    assert 5 in tracker.active_track_ids
    tracker.drop(5)
    assert 5 not in tracker.active_track_ids


def test_emitted_track3d_has_zero_velocity_before_second_update() -> None:
    """The very first observation cannot have a velocity estimate."""
    tracker = Tracker3D(_cfg())
    out = tracker.update(
        track_id=1, xyz_obs=np.array([1.0, 0.0, 0.0]),
        capture_ts=0.0, cameras_seeing=("cam_a", "cam_b"),
        cls="person", max_reproj_error_px=0.2,
    )
    assert out.vxyz_m == pytest.approx((0.0, 0.0, 0.0))


def test_default_is_not_single_view() -> None:
    out = Tracker3D(_cfg()).update(
        track_id=1, xyz_obs=np.array([0.0, 0.0, 1.0]), capture_ts=0.0,
        cameras_seeing=("cam_a", "cam_b"), cls="person", max_reproj_error_px=0.3,
    )
    assert out.single_view is False
    assert out.confidence == pytest.approx(1.0)


def test_single_view_emit_flags_and_confidence() -> None:
    out = Tracker3D(_cfg()).update(
        track_id=5, xyz_obs=np.array([2.0, 1.0, 0.0]), capture_ts=0.0,
        cameras_seeing=("cam_a",), cls="person", max_reproj_error_px=0.0,
        single_view=True, confidence=0.5,
    )
    assert out.single_view is True
    assert out.confidence == pytest.approx(0.5)
    assert out.contributing_cameras == ("cam_a",)
    assert out.xyz_m[2] == pytest.approx(0.0, abs=1e-6)


def test_occlusion_continuity_2view_single_2view() -> None:
    """2-view → single-view (Z=0 fallback) → 2-view stays ONE continuous track."""
    tracker = Tracker3D(_cfg())
    a = tracker.update(track_id=9, xyz_obs=np.array([0.0, 0.0, 1.0]), capture_ts=0.0,
                       cameras_seeing=("cam_a", "cam_b"), cls="person", max_reproj_error_px=0.3)
    assert a.single_view is False
    b = tracker.update(track_id=9, xyz_obs=np.array([0.1, 0.0, 0.0]), capture_ts=0.033,
                       cameras_seeing=("cam_a",), cls="person", max_reproj_error_px=0.0,
                       single_view=True, confidence=0.5)
    assert b.single_view is True and b.track_id == 9
    c = tracker.update(track_id=9, xyz_obs=np.array([0.2, 0.0, 1.0]), capture_ts=0.066,
                       cameras_seeing=("cam_a", "cam_b"), cls="person", max_reproj_error_px=0.3)
    assert c.single_view is False and c.track_id == 9
    assert tracker.active_track_ids == (9,)   # one track throughout the occlusion
