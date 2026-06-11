"""``InternalTrack`` — Kalman driving + lifecycle state transitions."""

from __future__ import annotations

import pytest

from backbone.homography.track import InternalTrack, TrackConfig, TrackState


def _track(xy: tuple[float, float] = (1.0, 2.0), cfg: TrackConfig | None = None) -> InternalTrack:
    return InternalTrack.create(
        track_id=1,
        cls_label="person",
        xy_m=xy,
        confidence=0.9,
        cameras_seeing=("cam_a",),
        capture_ts=0.0,
        cfg=cfg,
    )


# ----- Kalman -----


def test_predict_advances_state_with_velocity() -> None:
    track = _track()
    # Inject a known velocity directly.
    track.kf.x[2] = 1.0  # vX
    track.kf.x[3] = 0.5  # vY
    track.predict(dt=0.1)
    x, y = track.xy()
    assert x == pytest.approx(1.1, abs=1e-9)
    assert y == pytest.approx(2.05, abs=1e-9)


def test_predict_with_zero_dt_preserves_state() -> None:
    track = _track()
    track.predict(dt=0.0)
    assert track.xy() == pytest.approx((1.0, 2.0), abs=1e-9)


def test_update_with_reduces_position_uncertainty() -> None:
    track = _track()
    cov_before = track.position_covariance()
    track.update_with(
        xy_m=(1.01, 2.0), cls_label="person", confidence=0.9,
        cameras_seeing=("cam_a", "cam_b"), capture_ts=0.033,
    )
    cov_after = track.position_covariance()
    # Trace decreases — the Kalman update is informative.
    assert cov_after.trace() < cov_before.trace()


def test_class_history_grows_with_updates() -> None:
    track = _track()
    assert list(track.class_history) == ["person"]
    track.update_with(
        xy_m=(1.0, 2.0), cls_label="forklift", confidence=0.6,
        cameras_seeing=("cam_a",), capture_ts=0.1,
    )
    assert list(track.class_history) == ["person", "forklift"]


def test_class_history_respects_window_size() -> None:
    cfg = TrackConfig(class_history_window=3)
    track = _track(cfg=cfg)
    for label in ["forklift", "pallet", "person", "person"]:
        track.update_with(
            xy_m=(1.0, 2.0), cls_label=label, confidence=0.9,
            cameras_seeing=("cam_a",), capture_ts=0.1,
        )
    # Window is 3 — first label ("person", from create) is dropped, then "forklift" etc.
    assert len(track.class_history) == 3
    assert list(track.class_history) == ["pallet", "person", "person"]


# ----- Lifecycle -----


def test_new_track_starts_in_new_state() -> None:
    track = _track()
    assert track.state == TrackState.NEW
    assert track.is_active
    assert not track.is_publishable


def test_new_track_confirms_after_min_hits() -> None:
    cfg = TrackConfig(min_hits_to_confirm=3)
    track = _track(cfg=cfg)
    # 1st hit happens at construction; need 2 more.
    track.update_with(
        xy_m=(1.0, 2.0), cls_label="person", confidence=0.9,
        cameras_seeing=("cam_a",), capture_ts=0.1,
    )
    assert track.state == TrackState.NEW
    track.update_with(
        xy_m=(1.0, 2.0), cls_label="person", confidence=0.9,
        cameras_seeing=("cam_a",), capture_ts=0.2,
    )
    assert track.state == TrackState.TRACKED
    assert track.is_publishable


def test_new_track_dies_on_first_miss() -> None:
    """Flicker suppression: a single-frame detection doesn't become a tracked object."""
    track = _track()
    track.mark_missed()
    assert track.state == TrackState.REMOVED


def test_tracked_track_transitions_to_lost_then_removed() -> None:
    cfg = TrackConfig(min_hits_to_confirm=1, max_lost_frames=3)
    track = _track(cfg=cfg)
    # Initial state is NEW with min_hits_to_confirm=1 → still NEW. Update once to confirm.
    track.update_with(
        xy_m=(1.0, 2.0), cls_label="person", confidence=0.9,
        cameras_seeing=("cam_a",), capture_ts=0.1,
    )
    assert track.state == TrackState.TRACKED
    # Miss once → LOST.
    track.mark_missed()
    assert track.state == TrackState.LOST
    # Miss two more → still LOST.
    track.mark_missed()
    track.mark_missed()
    assert track.state == TrackState.LOST
    # Miss past the limit → REMOVED.
    track.mark_missed()
    assert track.state == TrackState.REMOVED


def test_lost_track_recovers_to_tracked_on_update() -> None:
    cfg = TrackConfig(min_hits_to_confirm=1, max_lost_frames=10)
    track = _track(cfg=cfg)
    track.update_with(
        xy_m=(1.0, 2.0), cls_label="person", confidence=0.9,
        cameras_seeing=("cam_a",), capture_ts=0.1,
    )
    track.mark_missed()
    assert track.state == TrackState.LOST
    # Comes back!
    track.update_with(
        xy_m=(1.05, 2.0), cls_label="person", confidence=0.9,
        cameras_seeing=("cam_a",), capture_ts=0.2,
    )
    assert track.state == TrackState.TRACKED
