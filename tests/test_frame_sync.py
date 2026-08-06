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


def test_single_camera_emits_solo_immediately() -> None:
    """Mode 1: a single-camera config is permanently degraded — every frame
    emits AT ONCE with its own capture_ts (no 100 ms partner-wait tax, no
    stale backlog)."""
    sync = FrameSynchronizer(
        camera_ids=["cam_a"], max_skew_ms=33.0, degraded_emit_after_ms=100.0,
    )
    pair = sync.submit(_frame("cam_a", 1.000))
    assert pair is not None
    assert set(pair.frames) == {"cam_a"}
    assert pair.frames["cam_a"].capture_ts == 1.000
    pair2 = sync.submit(_frame("cam_a", 1.033))
    assert pair2 is not None and pair2.frames["cam_a"].capture_ts == 1.033


def test_solo_emission_is_latest_only_never_a_backlog() -> None:
    """LATEST-FRAME-ONLY: single-cam emits every frame immediately, so the
    buffer never grows and no submit ever re-serves an older frame."""
    sync = FrameSynchronizer(
        camera_ids=["cam_a"], degraded_emit_after_ms=50.0,
    )
    for i in range(3):
        pair = sync.submit(_frame("cam_a", 1.000 + i * 0.020))
        assert pair is not None
        assert pair.frames["cam_a"].capture_ts == 1.000 + i * 0.020
    assert sync.buffer_depths == {"cam_a": 0}


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
    pair2 = sync.submit(_frame("cam_a", 1.150))           # 1.033 aged 117 ms → degraded
    assert pair2 is not None
    assert set(pair2.frames) == {"cam_a"}
    # LATEST-only entry: the NEWEST frame (1.150) emits and the older backlog
    # (1.033/1.066) is discarded — never drained stale-first.
    assert pair2.frames["cam_a"].capture_ts == 1.150
    assert sync.buffer_depths["cam_a"] == 0
    # While degraded, subsequent frames emit immediately (full input fps).
    pair3 = sync.submit(_frame("cam_a", 1.183))
    assert pair3 is not None and pair3.frames["cam_a"].capture_ts == 1.183


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
    solo = sync.submit(_frame("cam_a", 1.150))   # degraded entry: newest emits
    assert solo is not None and solo.frames["cam_a"].capture_ts == 1.150
    # While degraded, cam_a emits immediately.
    solo2 = sync.submit(_frame("cam_a", 1.183))
    assert solo2 is not None and set(solo2.frames) == {"cam_a"}
    # cam_b comes back; the next close cam_a frame pairs strictly again.
    assert sync.submit(_frame("cam_b", 1.193)) is None    # buffered, waits for cam_a
    pair = sync.submit(_frame("cam_a", 1.200))
    assert pair is not None
    assert set(pair.frames) == {"cam_a", "cam_b"}
    # Recovery cleared the degraded flag: a lone young cam_a frame BUFFERS
    # again (waits for its partner) instead of emitting solo at once.
    assert sync.submit(_frame("cam_a", 1.233)) is None


def test_capture_ts_monotonic_across_degrade_and_recovery() -> None:
    """Latest-only emission must keep the pair capture_ts monotonic through a
    degrade → recover cycle (downstream trackers assume a forward clock)."""
    sync = FrameSynchronizer(
        camera_ids=["cam_a", "cam_b"], max_skew_ms=33.0, degraded_emit_after_ms=80.0,
    )
    emitted: list[float] = []

    def push(cid, ts):
        pair = sync.submit(_frame(cid, ts))
        if pair is not None:
            emitted.append(pair.capture_ts)

    push("cam_a", 1.000)
    push("cam_b", 1.010)          # aligned
    push("cam_a", 1.040)          # cam_b dies
    push("cam_a", 1.080)
    push("cam_a", 1.150)          # degraded entry → newest solo
    push("cam_a", 1.183)          # immediate solo
    push("cam_b", 1.193)          # partner back, buffers
    push("cam_a", 1.200)          # aligned again
    push("cam_a", 1.233)          # buffers (flag cleared)
    assert emitted == sorted(emitted), f"capture_ts went backwards: {emitted}"
    assert len(emitted) >= 4


def test_both_cameras_degraded_recover_to_aligned_pairing() -> None:
    """Regression: once BOTH cameras were sticky-degraded (e.g. a startup
    stall), solo emission cleared the buffers on every submit, so the aligned
    path never saw two heads at once and pairing never resumed — permanent
    solo mode (live 2026-08-06: cameras_seeing never reached 2, so no
    Track3D despite both cameras detecting). A periodic probe must
    un-degrade one camera so alignment can re-form."""
    sync = FrameSynchronizer(
        camera_ids=["cam_a", "cam_b"], max_skew_ms=33.0, degraded_emit_after_ms=80.0,
    )
    # Degrade cam_a (partner never shows), then cam_b the same way.
    sync.submit(_frame("cam_a", 1.000))
    solo_a = sync.submit(_frame("cam_a", 1.100))
    assert solo_a is not None and set(solo_a.frames) == {"cam_a"}
    sync.submit(_frame("cam_b", 1.150))
    solo_b = sync.submit(_frame("cam_b", 1.250))
    assert solo_b is not None and set(solo_b.frames) == {"cam_b"}
    # Both degraded. Feed perfectly aligned frames: pairing must resume.
    fused = False
    ts = 3.0
    for _ in range(50):
        for pair in (sync.submit(_frame("cam_a", ts)),
                     sync.submit(_frame("cam_b", ts + 0.005))):
            if pair is not None and set(pair.frames) == {"cam_a", "cam_b"}:
                fused = True
        ts += 0.070
    assert fused, "both-degraded state never recovered to aligned pairing"
