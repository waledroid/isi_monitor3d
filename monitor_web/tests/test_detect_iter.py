"""``_detect_iter`` — the big-cam annotation iterator.

Pins the STRICT rule: the cam view never runs a full-frame object detector —
pose is the only model that runs on the full frame; object detections come
exclusively from the zone-worker snapshot (empty when no zones).

Also pins that the pose model INHERITS the Camera FPS: it runs on EVERY cam-view
frame (the frames already arrive at the source ``capture_fps`` = Camera FPS), and
is independent of the zones-only ``display_fps`` ("Zones FPS") preference.
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


# ---- pose inherits the Camera FPS (runs every cam-view frame) ----


def _patch_for_pose_rate(monkeypatch, pose_call_log: list) -> None:
    """Stubs for the pose-rate tests. Records each call to the pose detector
    factory so we can assert it runs once per cam-view frame."""
    monkeypatch.setattr(routes_video, "distances_enabled", lambda cfg: False)
    monkeypatch.setattr(routes_video, "distance_line_style", lambda cfg: {})
    monkeypatch.setattr(routes_video, "occupancy_enabled", lambda cfg: False)
    monkeypatch.setattr(routes_video, "nodes_enabled", lambda cfg: True)
    monkeypatch.setattr(routes_video, "masks_enabled", lambda cfg: True)
    monkeypatch.setattr(routes_video, "boxes_enabled", lambda cfg: True)
    monkeypatch.setattr(routes_video, "person_pallet_max_m", lambda cfg: 6.0)

    def counting_pose(cfg):
        pose_call_log.append(len(pose_call_log))
        return object()

    monkeypatch.setattr(routes_video, "get_pose_detector", counting_pose)
    monkeypatch.setattr(routes_video, "annotate_frame",
                        lambda image, detector, **kw: image)


def test_pose_runs_every_frame_inherits_camera_fps(monkeypatch) -> None:
    """Pose inherits the Camera FPS: the cam-view frames arrive at the source
    capture_fps, and pose runs on EVERY one of them — N frames ⇒ N pose calls.
    No time-gate / carry-forward throttle on the cam view."""
    pose_calls: list = []
    _patch_for_pose_rate(monkeypatch, pose_calls)

    frames_in = [_frame() for _ in range(10)]
    frames_out = list(routes_video._detect_iter(
        iter(frames_in), Settings(), "cam_a",
        is_running=lambda: True, get_zone_dets=lambda: []))

    assert len(frames_out) == 10           # all frames yielded (fluid)
    assert len(pose_calls) == 10, (
        f"Expected 10 pose calls (one per frame), got {len(pose_calls)}")


def test_pose_rate_independent_of_display_fps(monkeypatch) -> None:
    """The zones-only ``display_fps`` ("Zones FPS") must NOT throttle cam-view
    pose: even with a tiny display_fps, pose still runs once per frame."""
    pose_calls: list = []
    _patch_for_pose_rate(monkeypatch, pose_calls)
    # A low Zones-FPS must have zero effect on the cam-view pose rate.
    monkeypatch.setattr(routes_video, "display_fps", lambda cfg: 1.0)

    frames_out = list(routes_video._detect_iter(
        iter([_frame() for _ in range(6)]), Settings(), "cam_a",
        is_running=lambda: True, get_zone_dets=lambda: []))

    assert len(frames_out) == 6
    assert len(pose_calls) == 6, (
        f"display_fps must not gate pose; expected 6 calls, got {len(pose_calls)}")


def test_pose_runs_every_frame_for_cam_b(monkeypatch) -> None:
    """Both cam views get per-frame pose — the detect path runs per visible
    camera, so cam_b behaves identically to cam_a (no cam_a-only special case)."""
    pose_calls: list = []
    _patch_for_pose_rate(monkeypatch, pose_calls)

    frames_out = list(routes_video._detect_iter(
        iter([_frame() for _ in range(5)]), Settings(), "cam_b",
        is_running=lambda: True, get_zone_dets=lambda: []))

    assert len(frames_out) == 5
    assert len(pose_calls) == 5
