"""``TemporalStabilizer`` — majority-vote class + flicker suppression."""

from __future__ import annotations

from backbone.homography.bytetrack import ByteTrackMeters
from backbone.homography.temporal_stabilizer import TemporalStabilizer
from backbone.homography.track import TrackConfig


def _obs(cls: str, xy: tuple[float, float], conf: float = 0.9) -> tuple:
    return (cls, xy, conf, ("cam_a",))


def _run_classes(class_sequence: list[str]) -> str:
    """Feed a fixed sequence of class labels at the same position; return final voted cls."""
    tr = ByteTrackMeters(
        track_config=TrackConfig(min_hits_to_confirm=1, class_history_window=5),
    )
    stab = TemporalStabilizer(tr)
    out: list = []
    for i, cls in enumerate(class_sequence):
        out = stab.stabilize(tr.update(i * 0.033, [_obs(cls, (1.0, 1.0))]))
    return out[0].cls


def test_unanimous_history_emits_that_class() -> None:
    assert _run_classes(["person"] * 5) == "person"


def test_majority_wins_over_minority() -> None:
    assert _run_classes(["person", "person", "person", "person", "forklift"]) == "person"


def test_recency_breaks_ties() -> None:
    """2 person + 2 forklift over 4 frames → the most recent class wins the tie."""
    assert _run_classes(["forklift", "person", "forklift", "person"]) == "person"


def test_sliding_window_forgets_oldest() -> None:
    """Window=5 → after 5 'forklift' frames, only forklift remains."""
    assert _run_classes(
        ["person", "person", "forklift", "forklift", "forklift", "forklift", "forklift"]
    ) == "forklift"


def test_min_frames_confirmed_suppresses_short_tracks() -> None:
    """A track with only 2 history entries is suppressed when min_frames_confirmed=5."""
    tr = ByteTrackMeters(
        track_config=TrackConfig(min_hits_to_confirm=1, class_history_window=10),
    )
    stab = TemporalStabilizer(tr, min_frames_confirmed=5)
    # Two frames only.
    raw1 = tr.update(0.0, [_obs("person", (1.0, 1.0))])
    out1 = stab.stabilize(raw1)
    raw2 = tr.update(0.033, [_obs("person", (1.0, 1.0))])
    out2 = stab.stabilize(raw2)
    assert out1 == []
    assert out2 == []
    # Build up to the threshold.
    for i in range(2, 6):
        raw = tr.update(i * 0.033, [_obs("person", (1.0, 1.0))])
        out = stab.stabilize(raw)
    # 5th frame onwards → published.
    assert len(out) == 1
    assert out[0].cls == "person"


def test_min_frames_confirmed_default_is_permissive() -> None:
    """Default min_frames_confirmed=1 → every confirmed track gets published."""
    tr = ByteTrackMeters(
        track_config=TrackConfig(min_hits_to_confirm=1),
    )
    stab = TemporalStabilizer(tr)
    raw = tr.update(0.0, [_obs("person", (1.0, 1.0))])
    out = stab.stabilize(raw)
    assert len(out) == 1
