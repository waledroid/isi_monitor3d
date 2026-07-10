"""``_detect_iter`` — the big-cam annotation iterator.

Pins the STRICT rule: the cam view never runs a full-frame object detector —
pose is the only model that runs on the full frame; object detections come
exclusively from the zone-worker snapshot (empty when no zones).

Also pins the ASYNC pose contract: the per-camera pose runner is fetched and
fed on EVERY cam-view frame (the video rate is never chained to the model's
latency — inference happens in the runner's background worker), independent of
the zones-only ``display_fps`` ("Zones FPS") preference.
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
    monkeypatch.setattr(routes_video, "nodes_enabled", lambda cfg: True)
    monkeypatch.setattr(routes_video, "masks_enabled", lambda cfg: True)
    monkeypatch.setattr(routes_video, "boxes_enabled", lambda cfg: True)
    monkeypatch.setattr(routes_video, "person_pallet_max_m", lambda cfg: 6.0)
    monkeypatch.setattr(routes_video, "get_async_pose",
                        lambda cfg, cam: calls.setdefault("pose_fetched", True) and None)

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
    # routes_video no longer even IMPORTS get_detector (one perception — the
    # warp path's inference was removed too), so the guarantee is structural;
    # the patch stays as a tripwire in case the import ever returns.
    monkeypatch.setattr(
        routes_video, "get_detector",
        lambda cfg: (_ for _ in ()).throw(AssertionError("full-frame detector must not be built")),
        raising=False,
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
    monkeypatch.setattr(routes_video, "get_async_pose", lambda cfg, cam: sentinel_pose)
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


# ---- pose runner is fed every cam-view frame (async, never blocks) ----


def _patch_for_pose_rate(monkeypatch, pose_call_log: list) -> None:
    """Stubs for the pose-rate tests. Records each fetch of the async pose
    runner so we can assert it is fed once per cam-view frame."""
    monkeypatch.setattr(routes_video, "distances_enabled", lambda cfg: False)
    monkeypatch.setattr(routes_video, "distance_line_style", lambda cfg: {})
    monkeypatch.setattr(routes_video, "nodes_enabled", lambda cfg: True)
    monkeypatch.setattr(routes_video, "masks_enabled", lambda cfg: True)
    monkeypatch.setattr(routes_video, "boxes_enabled", lambda cfg: True)
    monkeypatch.setattr(routes_video, "person_pallet_max_m", lambda cfg: 6.0)

    def counting_pose(cfg, cam):
        pose_call_log.append(len(pose_call_log))
        return object()

    monkeypatch.setattr(routes_video, "get_async_pose", counting_pose)
    monkeypatch.setattr(routes_video, "annotate_frame",
                        lambda image, detector, **kw: image)


def test_pose_runs_every_frame_inherits_camera_fps(monkeypatch) -> None:
    """The async pose runner is fetched and fed on EVERY cam-view frame —
    N frames ⇒ N runner submissions (the worker itself paces the actual
    inference; the video loop never waits)."""
    pose_calls: list = []
    _patch_for_pose_rate(monkeypatch, pose_calls)

    frames_in = [_frame() for _ in range(10)]
    frames_out = list(routes_video._detect_iter(
        iter(frames_in), Settings(), "cam_a",
        is_running=lambda: True, get_zone_dets=lambda: []))

    assert len(frames_out) == 10           # all frames yielded (fluid)
    assert len(pose_calls) == 10, (
        f"Expected 10 pose calls (one per frame), got {len(pose_calls)}")



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


def test_overlay_failure_yields_raw_frame_and_keeps_streaming(monkeypatch):
    """An overlay crash (e.g. cv2 on a read-only frame) must NOT kill the pump:
    the panel shows the raw frame and the stream continues."""
    import numpy as np

    from monitor_web.api import routes_video

    calls: dict = {}
    _patch_overlay_helpers(monkeypatch, calls)

    def _boom(*a, **k):
        raise cv2_error("dst marked as output argument … readonly")

    class cv2_error(Exception):
        pass

    monkeypatch.setattr(routes_video, "annotate_frame", _boom)
    frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(3)]
    out = list(routes_video._detect_iter(iter(frames), cfg=None, camera_id="cam_a",
                                         is_running=lambda: True,
                                         get_zone_dets=lambda: []))
    assert len(out) == 3, "stream must survive overlay failures"
    assert all(o is f for o, f in zip(out, frames, strict=False)), "raw frames passed through"
