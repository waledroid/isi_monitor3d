"""``_detect_iter`` — the big-cam annotation iterator.

Pins the STRICT rule: the cam view never runs a full-frame object detector —
object detections come exclusively from the zone-worker snapshot (empty when
no zones), and skeletons come from the wire (``wire_pose``, the producer's
person observations) with ZERO dashboard inference. The CPU branch has no
in-dashboard pose engine at all.
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

    def fake_annotate(image, detector, **kwargs):
        calls["detector_arg"] = detector
        calls["detections"] = kwargs.get("detections")
        calls["pose"] = kwargs.get("pose_detector")
        return image

    monkeypatch.setattr(routes_video, "annotate_frame", fake_annotate)


def test_no_zones_never_builds_full_frame_detector(monkeypatch) -> None:
    """With zero zone patches the iterator must not even *look up* the full-frame
    detector — wire pose only, zone detections empty."""
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
    wire = object()                            # stand-in WirePoseSource
    cfg = Settings()
    out = list(routes_video._detect_iter(
        iter([_frame()]), cfg, "cam_a", is_running=lambda: True,
        get_zone_dets=None, wire_pose=wire))
    assert len(out) == 1
    assert calls["detector_arg"] is None       # annotate never gets a detector
    assert calls["detections"] == []           # no zones → no object detections
    assert calls["pose"] is wire               # skeletons from the wire, zero inference


def test_zone_snapshot_detections_are_rendered(monkeypatch) -> None:
    calls: dict = {}
    _patch_overlay_helpers(monkeypatch, calls)
    sentinel = [object(), object()]
    cfg = Settings()
    out = list(routes_video._detect_iter(
        iter([_frame()]), cfg, "cam_a",
        is_running=lambda: True, get_zone_dets=lambda _img=None: sentinel))
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
                                         get_zone_dets=lambda _img=None: []))
    assert len(out) == 3, "stream must survive overlay failures"
    assert all(o is f for o, f in zip(out, frames, strict=False)), "raw frames passed through"
