"""Motion gate — static scenes skip inference, cached detections keep flowing."""

from __future__ import annotations

import numpy as np

from backbone.core.types import Detection
from isistream.core import IsistreamCore
from isistream.motion_gate import MotionGate


def _gate(refresh_s=100.0):
    # One zone crop covering the middle of a 1280x720 frame (calibration = frame).
    return MotionGate({"cam_a": [("Z", (400, 200, 800, 600))]},
                      {"cam_a": (1280, 720)}, refresh_s=refresh_s)


def _frame(fill=0):
    return np.full((720, 1280, 3), fill, dtype=np.uint8)


def test_static_zone_skips_after_first_inference():
    g = _gate()
    assert g.objects_due("cam_a", _frame(), 0.0) is True    # first ⇒ infer
    assert g.objects_due("cam_a", _frame(), 1.0) is False   # unchanged ⇒ skip
    assert g.obj_skips == 1


def test_zone_motion_wakes_the_detector():
    g = _gate()
    f = _frame()
    assert g.objects_due("cam_a", f, 0.0) is True
    moved = f.copy()
    moved[300:500, 500:700] = 255                           # inside the zone crop
    assert g.objects_due("cam_a", moved, 1.0) is True


def test_motion_outside_zone_does_not_wake_objects_but_wakes_pose():
    g = _gate()
    f = _frame()
    g.objects_due("cam_a", f, 0.0)
    g.pose_due("cam_a", f, 0.0)
    moved = f.copy()
    moved[0:150, 0:300] = 255                               # OUTSIDE the zone crop
    assert g.objects_due("cam_a", moved, 1.0) is False      # zone crops unchanged
    assert g.pose_due("cam_a", moved, 1.0) is True          # full frame changed


def test_forced_refresh_reinfers():
    g = _gate(refresh_s=0.5)
    f = _frame()
    assert g.objects_due("cam_a", f, 0.0) is True
    assert g.objects_due("cam_a", f, 0.2) is False
    assert g.objects_due("cam_a", f, 0.6) is True           # refresh window elapsed


class _CountingDetector:
    def __init__(self):
        self.calls = 0

    def detect(self, pair):
        self.calls += 1
        return {cid: [Detection(
            camera_id=cid, capture_ts=f.capture_ts, cls="palette",
            confidence=0.8, bbox_xyxy=(500.0, 300.0, 700.0, 500.0),
            foot_uv=(600.0, 500.0))] for cid, f in pair.frames.items()}


def test_gated_core_reemits_cached_dets_with_fresh_ts():
    det = _CountingDetector()
    core = IsistreamCore(
        camera_ids=["cam_a"],
        frame_provider=None, object_detector=det, pose_detector=None,
        ingest_addr=("127.0.0.1", 1), motion_gate=_gate())
    core._frames = lambda cid: None      # tick() reads frames itself; we drive manually

    captured = []
    core._sock = type("S", (), {"sendto": lambda self, p, a: captured.append(p)})()

    img = _frame()
    ts = [100.0]

    def provider(cid):
        return img, ts[0]

    core._frames = provider
    core.tick()                          # infers (first sight)
    ts[0] += 0.07
    core.tick()                          # static ⇒ gated, cached re-emit
    ts[0] += 0.07
    core.tick()
    assert det.calls == 1, "static scene must not re-infer"

    import json
    msgs = [json.loads(p) for p in captured]
    assert len(msgs) == 3
    # Same detection every time (the cache), but each under a FRESH capture_ts.
    assert all(m["dets"][0]["bbox_xyxy"] == msgs[0]["dets"][0]["bbox_xyxy"] for m in msgs)
    assert msgs[0]["ts"] < msgs[1]["ts"] < msgs[2]["ts"]
    assert [m["seq"] for m in msgs] == [0, 1, 2]
