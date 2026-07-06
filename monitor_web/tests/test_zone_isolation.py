"""Per-zone isolation guards — VRAM admission, circuit breaker.

One zone's failing/refused detector must never affect the other zones (no shared
CUDA crash, no silent empty results).
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from backbone.core.types import Detection

import monitor_web.engines as overlay  # canonical home of session lifecycle
from monitor_web.detection_overlay import ZoneModelUnavailable
from monitor_web.zone_worker import ZONE_RETRY_COOLDOWN_S, ZoneDetectionWorker


class _CfgStub:
    backbone_config_path = __import__("pathlib").Path("/nonexistent/backbone.yaml")
    ui_settings_path = "/nonexistent/ui.yaml"


def _det(bbox=(20.0, 20.0, 30.0, 30.0)):
    return Detection(camera_id="cam_a", capture_ts=time.time(), cls="palette",
                     confidence=0.9, bbox_xyxy=bbox,
                     foot_uv=((bbox[0] + bbox[2]) / 2.0, bbox[3]))


def _patch(zone_id, x0, y0, x1, y1, **extra):
    p = {
        "id": zone_id, "name": zone_id, "camera": "cam_a",
        "polygon": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        "rect": [x0, y0, x1, y1], "frame_wh": [320, 240],
        "infer_size": 320, "confidence": None,
    }
    p.update(extra)
    return p


class CountingDetector:
    def __init__(self, dets):
        self._dets = dets
        self.calls = 0

    def detect(self, pair):
        self.calls += 1
        cam = next(iter(pair.frames))
        return {cam: list(self._dets)}


def _worker(patches, factory):
    w = ZoneDetectionWorker("cam_a", {"name": "rtsp", "url": "rtsp://x"}, _CfgStub(),
                            lambda: True, detector_factory=factory,
                            hub_factory=lambda: None)
    w.set_patches(patches)
    return w


FRAME = np.zeros((240, 320, 3), dtype=np.uint8)


# ---- circuit breaker ----------------------------------------------------------

def test_failing_zone_is_isolated_others_keep_running():
    """Zone with a refused model → status no_vram + no dets; healthy zone unaffected."""
    good = CountingDetector([_det()])

    def factory(model, cfg, size):
        if model == "/bad.onnx":
            raise ZoneModelUnavailable("no_vram", "refused")
        return good

    patches = [_patch("zbad", 0, 0, 100, 100, model="/bad.onnx"),
               _patch("zgood", 150, 0, 300, 200)]
    w = _worker(patches, factory)
    w._detect_all_zones(FRAME, patches)
    snap = w.snapshot()
    assert snap["status"]["zbad"] == "no_vram"
    assert snap["zones"]["zbad"] == []
    assert snap["status"]["zgood"] == "ok"
    assert len(snap["zones"]["zgood"]) == 1
    assert good.calls == 1


def test_breaker_cooldown_skips_retry_then_reopens():
    """A failed zone is NOT retried within the cooldown; it is after."""
    attempts = {"n": 0}

    def factory(model, cfg, size):
        attempts["n"] += 1
        raise RuntimeError("boom")

    patches = [_patch("z1", 0, 0, 100, 100)]
    w = _worker(patches, factory)
    w._detect_all_zones(FRAME, patches)
    assert attempts["n"] == 1
    assert w.snapshot()["status"]["z1"] == "error"
    w._detect_all_zones(FRAME, patches)          # within cooldown → no new attempt
    assert attempts["n"] == 1
    assert w.snapshot()["status"]["z1"] == "error"
    # Expire the breaker → retried on the next pass.
    blocked_until, reason = w._zone_breaker["z1"]
    w._zone_breaker["z1"] = (blocked_until - ZONE_RETRY_COOLDOWN_S - 1.0, reason)
    w._detect_all_zones(FRAME, patches)
    assert attempts["n"] == 2


def test_set_patches_clears_breaker():
    def factory(model, cfg, size):
        raise RuntimeError("boom")

    patches = [_patch("z1", 0, 0, 100, 100)]
    w = _worker(patches, factory)
    w._detect_all_zones(FRAME, patches)
    assert w._zone_breaker
    w.set_patches(patches)                       # config save → fresh chance
    assert not w._zone_breaker


# ---- snapshot freshness (anti-blink) --------------------------------------------

def test_slow_pass_extends_snapshot_validity():
    """A slow detect pass publishes a proportionally longer validity window, so
    consumers don't see 'stale' (blinking boxes) between publishes."""
    import monitor_web.zone_worker as zw

    class SlowDetector(CountingDetector):
        def detect(self, pair):
            time.sleep(0.6)                       # a heavy pass (> the 2.5x floor/2)
            return super().detect(pair)

    patches = [_patch("z1", 0, 0, 100, 100)]
    w = _worker(patches, lambda m, c, s: SlowDetector([_det()]))
    w._detect_all_zones(FRAME, patches)
    snap = w.snapshot()
    assert snap["valid_s"] >= 2.5 * 0.6 - 0.1
    # frame_ts older than the 1.0 s floor but within valid_s → still served.
    w._snapshot = {**snap, "frame_ts": time.time() - zw.SNAPSHOT_MAX_AGE_S - 0.2}
    assert w.zone_dets("z1")
    # ...but beyond valid_s → stale, no ghosts.
    w._snapshot = {**snap, "frame_ts": time.time() - snap["valid_s"] - 0.2}
    assert w.zone_dets("z1") == []


def test_stalled_camera_keeps_snapshot_alive():
    """Camera delivering no NEW frame (RTSP jitter / low fps) must not expire the
    snapshot — the panels still show that same held frame, so its detections stay
    correct. This was the blinking-overlay bug."""
    frame = FRAME.copy()

    class StalledStream:
        def latest_real_frame(self):
            return frame                          # same object forever

    class Hub:
        def acquire(self, *a):
            return StalledStream()

        def release(self, s):
            pass

    patches = [_patch("z1", 0, 0, 100, 100)]
    w = ZoneDetectionWorker("cam_a", {"name": "rtsp", "url": "rtsp://x"}, _CfgStub(),
                            lambda: True,
                            detector_factory=lambda m, c, s: CountingDetector([_det()]),
                            hub_factory=lambda: Hub())
    w.set_patches(patches)
    w.start()
    try:
        deadline = time.time() + 3.0
        while time.time() < deadline and not w.zone_dets("z1"):
            time.sleep(0.05)
        assert w.zone_dets("z1"), "worker never published"
        time.sleep(1.3)                           # > SNAPSHOT_MAX_AGE_S with no new frame
        assert w.zone_dets("z1"), "stalled camera must not blink the boxes off"
    finally:
        w.stop()


# ---- VRAM admission in get_zone_detector ---------------------------------------

def test_get_zone_detector_refuses_when_vram_low(tmp_path, monkeypatch):
    model = tmp_path / "fake.onnx"
    model.write_bytes(b"not a real model")
    monkeypatch.setattr(overlay, "_ZONE_DETECTORS", {})   # isolate the global cache
    monkeypatch.setattr(overlay, "_gpu_free_mb", lambda: 500.0)
    # Reference the exception class THROUGH the module: test_detector_selection
    # reloads detection_overlay mid-suite, which rebinds the class object — a
    # class imported at this file's import time would no longer match.
    with pytest.raises(overlay.ZoneModelUnavailable) as exc:
        overlay.get_zone_detector(str(model), _CfgStub(), 320)
    assert exc.value.reason == "no_vram"
    assert not overlay._ZONE_DETECTORS            # nothing cached on refusal
