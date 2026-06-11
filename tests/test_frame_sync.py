"""``FrameSynchronizer`` — pairing algorithm against synthetic timestamp streams."""

from __future__ import annotations

import numpy as np
import pytest

from backbone.core.types import Frame
from backbone.ingestion.frame_sync import FrameSynchronizer


def _frame(camera_id: str, capture_ts: float, idx: int = 0) -> Frame:
    img = np.zeros((1, 1, 3), dtype=np.uint8)
    return Frame(camera_id=camera_id, capture_ts=capture_ts, frame_idx=idx, image=img)


def test_single_camera_is_allowed() -> None:
    """Mode 1 (1 camera) is a valid configuration."""
    sync = FrameSynchronizer(camera_ids=["cam_a"])
    assert sync.camera_ids == ("cam_a",)


def test_rejects_zero_cameras() -> None:
    with pytest.raises(ValueError, match=">=1 camera"):
        FrameSynchronizer(camera_ids=[])


def test_rejects_duplicate_camera_ids() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        FrameSynchronizer(camera_ids=["cam_a", "cam_a"])


def test_aligned_frames_emit_pair() -> None:
    sync = FrameSynchronizer(camera_ids=["cam_a", "cam_b"], max_skew_ms=33.0)
    assert sync.submit(_frame("cam_a", 1.000)) is None  # waiting for cam_b
    pair = sync.submit(_frame("cam_b", 1.010))
    assert pair is not None
    assert set(pair.frames) == {"cam_a", "cam_b"}
    assert pair.capture_ts == pytest.approx(1.005, abs=1e-6)
    assert pair.frame_idx == 1


def test_skew_exceeds_tolerance_does_not_emit() -> None:
    """When skew exceeds tolerance and we haven't waited long enough for solo emit,
    no pair is produced — the synchronizer continues waiting."""
    sync = FrameSynchronizer(
        camera_ids=["cam_a", "cam_b"],
        max_skew_ms=33.0,
        degraded_emit_after_ms=500.0,   # generous, so the test below stays strict
    )
    sync.submit(_frame("cam_a", 1.000))
    pair = sync.submit(_frame("cam_b", 1.100))  # 100 ms apart — too much skew, not solo yet
    assert pair is None


def test_late_arriving_match_is_still_emitted() -> None:
    """A frame arriving after its partner — within tolerance — still pairs."""
    sync = FrameSynchronizer(camera_ids=["cam_a", "cam_b"], max_skew_ms=33.0)
    assert sync.submit(_frame("cam_b", 1.005)) is None
    pair = sync.submit(_frame("cam_a", 1.020))
    assert pair is not None


def test_matched_frames_are_consumed_no_duplicate_pair() -> None:
    sync = FrameSynchronizer(camera_ids=["cam_a", "cam_b"], max_skew_ms=33.0)
    sync.submit(_frame("cam_a", 1.000, idx=0))
    pair1 = sync.submit(_frame("cam_b", 1.010, idx=0))
    assert pair1 is not None
    # No new frames yet → no second pair.
    assert sync.pairs_emitted == 1
    assert sync.buffer_depths == {"cam_a": 0, "cam_b": 0}


def test_multiple_pairs_emitted_over_time() -> None:
    sync = FrameSynchronizer(camera_ids=["cam_a", "cam_b"], max_skew_ms=33.0)
    pairs = []
    # Two ~30 FPS streams, nearly aligned.
    for i in range(5):
        ts = 1.0 + i * 0.033
        sync.submit(_frame("cam_a", ts, i))
        p = sync.submit(_frame("cam_b", ts + 0.005, i))
        if p is not None:
            pairs.append(p)
    assert len(pairs) == 5
    assert [p.frame_idx for p in pairs] == [1, 2, 3, 4, 5]


def test_stale_frames_evicted_by_age() -> None:
    """Frames older than max_age_ms ago are dropped — they cannot pair anymore."""
    sync = FrameSynchronizer(
        camera_ids=["cam_a", "cam_b"], max_skew_ms=33.0, max_age_ms=100.0
    )
    sync.submit(_frame("cam_a", 1.000))  # buffered
    # cam_b arrives much later — beyond the skew window, but more importantly,
    # the eviction (driven by cam_b's ts) drops cam_a's stale frame.
    assert sync.submit(_frame("cam_b", 1.500)) is None
    # cam_a now arrives fresh → must pair with the freshly-buffered cam_b.
    pair = sync.submit(_frame("cam_a", 1.510))
    assert pair is not None


def test_out_of_order_frames_in_same_camera_still_pair() -> None:
    """Out-of-order arrivals from the same camera should still produce a pair.

    RTSP over TCP rarely delivers out-of-order frames in practice, so the
    algorithm doesn't sort: ``buf[-1]`` is the most recently *submitted*
    frame, not the latest timestamp. We only verify that *some* valid
    pair is emitted within tolerance — picking the optimal cam_a frame
    would require timestamp-sorted insertion and isn't worth the cost.
    """
    sync = FrameSynchronizer(camera_ids=["cam_a", "cam_b"], max_skew_ms=33.0)
    sync.submit(_frame("cam_a", 1.020, idx=1))
    sync.submit(_frame("cam_a", 1.000, idx=0))  # out of order arrival
    pair = sync.submit(_frame("cam_b", 1.025))
    assert pair is not None
    cam_a_ts = pair.frames["cam_a"].capture_ts
    assert cam_a_ts in (1.000, 1.020)
    assert abs(cam_a_ts - pair.frames["cam_b"].capture_ts) <= 0.033


def test_unknown_camera_id_ignored() -> None:
    sync = FrameSynchronizer(camera_ids=["cam_a", "cam_b"])
    assert sync.submit(_frame("cam_ghost", 1.0)) is None
    assert sync.buffer_depths == {"cam_a": 0, "cam_b": 0}


# ---------- single-camera (Mode 1) ----------


def test_single_camera_emits_solo_after_degraded_threshold() -> None:
    """Mode 1: every frame becomes a solo FramePair after the timeout."""
    sync = FrameSynchronizer(
        camera_ids=["cam_a"], max_skew_ms=33.0, degraded_emit_after_ms=100.0,
    )
    # First frame: not yet aged enough → no emit.
    assert sync.submit(_frame("cam_a", 1.000)) is None
    # Submit a much later frame; the OLDER buffered frame (1.000) has now
    # waited >= 100 ms relative to `latest_capture_ts`, so it emits solo.
    pair = sync.submit(_frame("cam_a", 1.200))
    assert pair is not None
    assert set(pair.frames) == {"cam_a"}
    assert pair.frames["cam_a"].capture_ts == 1.000


def test_solo_emission_consumes_one_frame_at_a_time() -> None:
    sync = FrameSynchronizer(
        camera_ids=["cam_a"], degraded_emit_after_ms=50.0,
    )
    for i in range(3):
        sync.submit(_frame("cam_a", 1.000 + i * 0.020))
    # buffer now has frames at 1.000, 1.020, 1.040; latest_capture_ts = 1.040
    # 1.000 has aged 40 ms — not enough yet.
    assert sync.buffer_depths == {"cam_a": 3}
    # Push latest forward by 100 ms.
    pair = sync.submit(_frame("cam_a", 1.140))
    assert pair is not None
    assert pair.frames["cam_a"].capture_ts == 1.000
    # The remaining frames keep draining one per submit as they age past the gate.
    pair2 = sync.submit(_frame("cam_a", 1.180))
    assert pair2 is not None
    assert pair2.frames["cam_a"].capture_ts == 1.020


# ---------- dual-cam runtime degradation ----------


def test_dual_cam_solo_emits_when_partner_stops_feeding() -> None:
    """Mode 2 with cam_b silent: surviving cam_a emits solo after the threshold."""
    sync = FrameSynchronizer(
        camera_ids=["cam_a", "cam_b"],
        max_skew_ms=33.0,
        degraded_emit_after_ms=80.0,
    )
    # Both alive: paired.
    sync.submit(_frame("cam_a", 1.000))
    pair = sync.submit(_frame("cam_b", 1.010))
    assert pair is not None
    assert set(pair.frames) == {"cam_a", "cam_b"}

    # cam_b stops. cam_a keeps feeding.
    assert sync.submit(_frame("cam_a", 1.033)) is None    # too young to solo
    assert sync.submit(_frame("cam_a", 1.066)) is None    # 1.033 still young
    pair2 = sync.submit(_frame("cam_a", 1.150))           # 1.033 aged 117 ms → solo
    assert pair2 is not None
    assert set(pair2.frames) == {"cam_a"}


def test_dual_cam_recovery_resumes_pairing_after_solo_phase() -> None:
    """When the partner comes back online, strict pairing resumes."""
    sync = FrameSynchronizer(
        camera_ids=["cam_a", "cam_b"],
        max_skew_ms=33.0,
        degraded_emit_after_ms=80.0,
    )
    sync.submit(_frame("cam_a", 1.000))
    sync.submit(_frame("cam_b", 1.010))     # paired
    sync.submit(_frame("cam_a", 1.033))     # waits
    sync.submit(_frame("cam_a", 1.150))     # solo emit of 1.033
    # cam_b comes back; new aligned pair.
    sync.submit(_frame("cam_a", 1.183))
    pair = sync.submit(_frame("cam_b", 1.193))
    assert pair is not None
    assert set(pair.frames) == {"cam_a", "cam_b"}
