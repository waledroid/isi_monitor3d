"""``_detect_iter`` — the big-cam annotation iterator.

Pins the STRICT rule: the cam view never runs a full-frame object detector —
pose is the only model that runs on the full frame; object detections come
exclusively from the zone-worker snapshot (empty when no zones).

Also tests the pose carry-forward time gate: the pose model must run at most
``display_fps`` times/second while video frames are yielded at the (higher)
source rate (Camera FPS).
"""

from __future__ import annotations

import numpy as np

from monitor_web.api import routes_video
from monitor_web.config import Settings


def _frame() -> np.ndarray:
    return np.zeros((48, 64, 3), dtype=np.uint8)


def _patch_overlay_helpers(monkeypatch, calls: dict) -> None:
    """Stub everything _detect_iter touches except the logic under test."""
    monkeypatch.setattr(routes_video, "distances_enabled", lambda cfg: False)
    monkeypatch.setattr(routes_video, "distance_line_style", lambda cfg: {})
    monkeypatch.setattr(routes_video, "occupancy_enabled", lambda cfg: False)
    monkeypatch.setattr(routes_video, "nodes_enabled", lambda cfg: True)
    monkeypatch.setattr(routes_video, "masks_enabled", lambda cfg: True)
    monkeypatch.setattr(routes_video, "boxes_enabled", lambda cfg: True)
    monkeypatch.setattr(routes_video, "person_pallet_max_m", lambda cfg: 6.0)
    monkeypatch.setattr(routes_video, "get_pose_detector",
                        lambda cfg: calls.setdefault("pose_fetched", True) and None)

    def fake_annotate(image, detector, **kwargs):
        calls["detector_arg"] = detector
        calls["detections"] = kwargs.get("detections")
        return image

    monkeypatch.setattr(routes_video, "annotate_frame", fake_annotate)


def test_no_zones_never_builds_full_frame_detector(monkeypatch) -> None:
    """With zero zone patches the iterator must not even *look up* the full-frame
    detector — pose only, zone detections empty."""
    calls: dict = {}
    _patch_overlay_helpers(monkeypatch, calls)
    monkeypatch.setattr(
        routes_video, "get_detector",
        lambda cfg: (_ for _ in ()).throw(AssertionError("full-frame detector must not be built")),
    )
    cfg = Settings()
    out = list(routes_video._detect_iter(
        iter([_frame()]), cfg, "cam_a", is_running=lambda: True, get_zone_dets=None))
    assert len(out) == 1
    assert calls["detector_arg"] is None       # annotate never gets a detector
    assert calls["detections"] == []           # no zones → no object detections
    assert calls["pose_fetched"] is True       # pose is still fetched


def test_zone_snapshot_detections_are_rendered(monkeypatch) -> None:
    calls: dict = {}
    _patch_overlay_helpers(monkeypatch, calls)
    sentinel = [object(), object()]
    cfg = Settings()
    out = list(routes_video._detect_iter(
        iter([_frame()]), cfg, "cam_a",
        is_running=lambda: True, get_zone_dets=lambda: sentinel))
    assert len(out) == 1
    assert calls["detector_arg"] is None
    assert calls["detections"] is sentinel


def test_backbone_stopped_yields_raw_frame(monkeypatch) -> None:
    calls: dict = {}
    _patch_overlay_helpers(monkeypatch, calls)
    frame = _frame()
    out = list(routes_video._detect_iter(
        iter([frame]), Settings(), "cam_a", is_running=lambda: False))
    assert len(out) == 1
    assert out[0] is frame
    assert "detector_arg" not in calls         # annotate_frame never called


def test_stop_releases_pose_ref_in_suspended_generator(monkeypatch) -> None:
    """After STOP the suspended generator must NOT pin the pose engine: its
    locals from the last running iteration would otherwise keep the CUDA
    session alive after reset_detector(), leaking VRAM until the panel closes."""
    calls: dict = {}
    _patch_overlay_helpers(monkeypatch, calls)
    sentinel_pose = object()
    monkeypatch.setattr(routes_video, "get_pose_detector", lambda cfg: sentinel_pose)
    running = {"on": True}
    gen = routes_video._detect_iter(
        iter([_frame(), _frame(), _frame()]), Settings(), "cam_a",
        is_running=lambda: running["on"], get_zone_dets=lambda: [object()])
    next(gen)                                   # running frame → pose fetched
    assert gen.gi_frame.f_locals.get("pose") is sentinel_pose
    running["on"] = False
    next(gen)                                   # stopped frame → refs dropped
    assert gen.gi_frame.f_locals.get("pose") is None
    assert gen.gi_frame.f_locals.get("dets") is None
    gen.close()


# ---- pose carry-forward time gate ----


def _patch_for_carry_forward(monkeypatch, pose_call_log: list) -> None:
    """Stubs for the carry-forward tests. Records each call to the pose detector
    factory so we can assert how many times it ran the (expensive) inference."""
    monkeypatch.setattr(routes_video, "distances_enabled", lambda cfg: False)
    monkeypatch.setattr(routes_video, "distance_line_style", lambda cfg: {})
    monkeypatch.setattr(routes_video, "occupancy_enabled", lambda cfg: False)
    monkeypatch.setattr(routes_video, "nodes_enabled", lambda cfg: True)
    monkeypatch.setattr(routes_video, "masks_enabled", lambda cfg: True)
    monkeypatch.setattr(routes_video, "boxes_enabled", lambda cfg: True)
    monkeypatch.setattr(routes_video, "person_pallet_max_m", lambda cfg: 6.0)

    def counting_pose(cfg):
        pose_call_log.append(len(pose_call_log))   # append a counter, not monotonic
        return object()   # new sentinel each call — distinct objects prove carry-forward

    monkeypatch.setattr(routes_video, "get_pose_detector", counting_pose)
    monkeypatch.setattr(routes_video, "annotate_frame",
                        lambda image, detector, **kw: image)


def test_pose_carry_forward_helper_runs_less_often_than_frames(monkeypatch) -> None:
    """The time-gate: given N frames in rapid succession and display_fps < N,
    get_pose_detector must be called fewer times than frames are yielded.

    We drive the iterator with 10 frames and set display_fps to 2 so the gate
    period is 0.5 s. Since all frames arrive within a few ms (no real sleep),
    the gate should fire exactly ONCE (on the first frame), then carry the last
    skeleton forward for the remaining 9 frames.
    """
    pose_calls: list = []
    _patch_for_carry_forward(monkeypatch, pose_calls)
    # display_fps(cfg) reads the UI-settings YAML; stub it to a fixed low value.
    monkeypatch.setattr(routes_video, "display_fps", lambda cfg: 2.0)

    frames_in = [_frame() for _ in range(10)]
    frames_out = list(routes_video._detect_iter(
        iter(frames_in), Settings(), "cam_a",
        is_running=lambda: True, get_zone_dets=lambda: []))

    # All 10 frames must be yielded (video stays fluid).
    assert len(frames_out) == 10
    # But the pose model ran only ONCE (the gate prevents re-inference within 0.5 s).
    assert len(pose_calls) == 1, (
        f"Expected 1 pose call for 10 rapid frames at 2 fps gate, got {len(pose_calls)}")


def test_pose_carry_forward_reruns_after_interval(monkeypatch) -> None:
    """After the gate interval elapses, the pose model is called again.

    We fake time.monotonic to advance manually so the test is instant (no real sleep).
    Three frames: t=0 (first, fires), t=0.05 (within gate, carry), t=1.0 (due again, fires).
    With display_fps=2 the gate period is 0.5 s, so frame 3 at t=1.0 should trigger again.
    """
    pose_calls: list = []
    _patch_for_carry_forward(monkeypatch, pose_calls)
    monkeypatch.setattr(routes_video, "display_fps", lambda cfg: 2.0)

    # We intercept the `import time as _time` inside _detect_iter by patching the
    # `time` module referenced in routes_video. The local `import time as _time`
    # at function entry uses the module object already in sys.modules, so patching
    # routes_video's module-level `time` (via monkeypatch.setattr on the module) is
    # the right seam. However since _detect_iter does `import time as _time` inside
    # itself, we need to patch `time.monotonic` in the global `time` module.
    fake_times = [0.0, 0.05, 1.0]
    call_idx = [0]

    import time as real_time
    original_monotonic = real_time.monotonic

    def fake_monotonic():
        if call_idx[0] < len(fake_times):
            t = fake_times[call_idx[0]]
            call_idx[0] += 1
            return t
        return original_monotonic()

    monkeypatch.setattr(real_time, "monotonic", fake_monotonic)

    frames_out = list(routes_video._detect_iter(
        iter([_frame(), _frame(), _frame()]), Settings(), "cam_a",
        is_running=lambda: True, get_zone_dets=lambda: []))

    assert len(frames_out) == 3
    # Frame 1 (t=0.0): gate fires (0 >= 0.0). Frame 2 (t=0.05): 0.05 < 0.5, carry.
    # Frame 3 (t=1.0): 1.0 >= 0.5, gate fires again.
    assert len(pose_calls) == 2, (
        f"Expected 2 pose calls (frame1 + frame3), got {len(pose_calls)}")
