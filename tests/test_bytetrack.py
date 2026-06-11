"""``ByteTrackMeters`` — multi-frame scenarios in metric space."""

from __future__ import annotations

import pytest

from backbone.core.interfaces import tracker_registry
from backbone.homography.bytetrack import ByteTrackMeters
from backbone.homography.track import TrackConfig


def _obs(cls: str, xy: tuple[float, float], conf: float, cams: tuple[str, ...] = ("cam_a",)) -> tuple:
    return (cls, xy, conf, cams)


# ----- registration -----


def test_plugin_registered_under_bytetrack() -> None:
    import backbone.homography  # noqa: F401

    assert "bytetrack" in tracker_registry


# ----- basic tracking -----


def test_single_object_gets_stable_track_id() -> None:
    tr = ByteTrackMeters(
        track_config=TrackConfig(min_hits_to_confirm=1, max_lost_frames=30),
    )
    ts = 0.0
    track_ids: set[int] = set()
    for i in range(10):
        ts = i * 0.033
        out = tr.update(ts, [_obs("person", (1.0 + i * 0.01, 2.0), 0.9)])
        for t in out:
            track_ids.add(t.track_id)
    assert len(track_ids) == 1


def test_track_id_recovers_after_brief_miss() -> None:
    """Object disappears for 2 frames within max_lost_frames → same track_id on return."""
    tr = ByteTrackMeters(
        track_config=TrackConfig(min_hits_to_confirm=1, max_lost_frames=10),
    )
    out1 = tr.update(0.0, [_obs("person", (1.0, 2.0), 0.9)])
    out2 = tr.update(0.033, [_obs("person", (1.0, 2.0), 0.9)])
    assert out1[0].track_id == out2[0].track_id
    first_id = out1[0].track_id

    # Miss 2 frames.
    tr.update(0.066, [])
    tr.update(0.099, [])

    # Reappears.
    out_back = tr.update(0.132, [_obs("person", (1.0, 2.0), 0.9)])
    assert len(out_back) == 1
    assert out_back[0].track_id == first_id


def test_track_removed_after_max_lost_frames() -> None:
    """Object missing past the lost budget → next reappearance gets a NEW id."""
    tr = ByteTrackMeters(
        track_config=TrackConfig(min_hits_to_confirm=1, max_lost_frames=3),
    )
    out_first = tr.update(0.0, [_obs("person", (1.0, 2.0), 0.9)])
    first_id = out_first[0].track_id
    for i in range(1, 6):
        tr.update(i * 0.033, [])
    out_back = tr.update(7 * 0.033, [_obs("person", (1.0, 2.0), 0.9)])
    # New track — old one was removed.
    assert len(out_back) == 1
    assert out_back[0].track_id != first_id


def test_two_objects_keep_separate_ids() -> None:
    tr = ByteTrackMeters(
        track_config=TrackConfig(min_hits_to_confirm=1),
    )
    out = tr.update(0.0, [
        _obs("person", (0.0, 0.0), 0.9),
        _obs("person", (5.0, 5.0), 0.9),
    ])
    assert len(out) == 2
    ids = {t.track_id for t in out}
    assert len(ids) == 2


def test_spurious_one_frame_detection_is_suppressed() -> None:
    """Flicker suppression: NEW track dies on first miss, never publishes."""
    tr = ByteTrackMeters(
        track_config=TrackConfig(min_hits_to_confirm=3),
    )
    # First frame: spurious detection appears.
    tr.update(0.0, [_obs("person", (1.0, 1.0), 0.9)])
    # Next frame: nothing.
    out = tr.update(0.033, [])
    # And another empty frame.
    out2 = tr.update(0.066, [])
    assert out == []
    assert out2 == []


def test_low_conf_only_observations_do_not_spawn_tracks() -> None:
    """Below conf_high (0.5) → never starts a new track on its own."""
    tr = ByteTrackMeters(
        conf_high=0.5,
        conf_low=0.1,
        track_config=TrackConfig(min_hits_to_confirm=1),
    )
    out = tr.update(0.0, [_obs("person", (1.0, 1.0), 0.3)])
    assert out == []


def test_low_conf_extends_existing_track() -> None:
    """Once a track exists (from high-conf), low-conf hits can extend it."""
    tr = ByteTrackMeters(
        conf_high=0.5,
        conf_low=0.1,
        track_config=TrackConfig(min_hits_to_confirm=1),
    )
    out0 = tr.update(0.0, [_obs("person", (1.0, 1.0), 0.9)])
    out1 = tr.update(0.033, [_obs("person", (1.02, 1.0), 0.3)])  # low-conf
    assert out1
    assert out0[0].track_id == out1[0].track_id


def test_below_conf_low_dropped_entirely() -> None:
    """Sub-low-confidence detections are discarded; no match, no spawn."""
    tr = ByteTrackMeters(
        conf_high=0.5,
        conf_low=0.1,
        track_config=TrackConfig(min_hits_to_confirm=1),
    )
    out = tr.update(0.0, [_obs("person", (1.0, 1.0), 0.05)])
    assert out == []


def test_reject_invalid_conf_order() -> None:
    with pytest.raises(ValueError, match="conf_low"):
        ByteTrackMeters(conf_high=0.1, conf_low=0.5)


# ----- crossing scenario -----


def test_crossing_objects_preserve_identity() -> None:
    """Two people walking towards each other but not through each other.

    cam_a sees them on parallel tracks separated by 2 m. Their identities
    should remain consistent through the closest-approach frame.
    """
    tr = ByteTrackMeters(
        track_config=TrackConfig(min_hits_to_confirm=1),
    )
    # Person A: moves +X. Person B: moves -X. Their Y coordinates differ by 2 m.
    ids_by_frame: list[set[int]] = []
    for i in range(20):
        ts = i * 0.033
        # Lines stay 2 m apart so they never collide in the metric space.
        xa = i * 0.05         # A walks right
        xb = 5.0 - i * 0.05   # B walks left
        out = tr.update(ts, [
            _obs("person", (xa, 0.0), 0.9),
            _obs("person", (xb, 2.0), 0.9),
        ])
        ids_by_frame.append({t.track_id for t in out})
    # Same two ids in every frame.
    assert len({frozenset(s) for s in ids_by_frame}) == 1
    assert len(ids_by_frame[-1]) == 2
