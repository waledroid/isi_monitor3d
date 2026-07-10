"""``perception.IsistreamCore`` — the Direction-1 producer, hermetic.

Fake frame provider + stub detectors + a real loopback DetectionIngest on
the receiving end: the full producer→engine wire path without cameras, CUDA,
or FastAPI.
"""

from __future__ import annotations

import time

import numpy as np

from backbone.core.types import Detection
from backbone.ingestion.points_in import DetectionIngest
from isistream.core import IsistreamCore


class _StubDetector:
    """Emits one palette per camera at a fixed spot, with a crop-local mask."""

    def detect(self, pair):
        out = {}
        for cam_id, frame in pair.frames.items():
            mask = np.zeros((60, 80), dtype=bool)
            mask[10:50, 10:70] = True
            out[cam_id] = [Detection(
                camera_id=cam_id, capture_ts=frame.capture_ts, cls="palette",
                confidence=0.8, bbox_xyxy=(100.0, 100.0, 180.0, 160.0),
                foot_uv=(140.0, 160.0), mask=mask, mask_offset_xy=(100, 100))]
        return out


class _FrameFeed:
    """Scripted (image, ts) per camera; ts advances only when told to."""

    def __init__(self, camera_ids):
        self.ts = dict.fromkeys(camera_ids, 100.0)
        self.img = np.zeros((720, 1280, 3), dtype=np.uint8)

    def advance(self, dt=0.05):
        for cid in self.ts:
            self.ts[cid] += dt

    def __call__(self, camera_id):
        return self.img, self.ts[camera_id]


def _drain(got, n, timeout=2.0):
    deadline = time.time() + timeout
    while len(got) < n and time.time() < deadline:
        time.sleep(0.02)
    return got


def test_core_emits_per_camera_sets_with_masks_and_seq() -> None:
    got = []
    ing = DetectionIngest(["cam_a", "cam_b"], port=0, on_set=got.append)
    ing.start()
    try:
        feed = _FrameFeed(["cam_a", "cam_b"])
        core = IsistreamCore(
            camera_ids=["cam_a", "cam_b"], frame_provider=feed,
            object_detector=_StubDetector(), pose_detector=None,
            ingest_addr=ing.address, fingerprint="fp1")
        core.tick()
        _drain(got, 2)
        assert sorted(ds.camera_id for ds in got) == ["cam_a", "cam_b"]
        ds = got[0]
        assert ds.frame_wh == (1280, 720)
        d = ds.detections[0]
        assert d.cls == "palette" and d.mask is not None
        # The polygon → rasterized mask round-trip preserves the crop origin.
        assert d.mask_offset_xy is not None and d.mask_offset_xy[0] >= 100

        # Same frames again (no advance): STALE ⇒ silence, seq unchanged.
        core.tick()
        time.sleep(0.2)
        assert len(got) == 2

        # Fresh frames ⇒ next seq.
        feed.advance()
        core.tick()
        _drain(got, 4)
        assert len(got) == 4
        assert {ds.frame_idx for ds in got} == {0, 1}   # seq 0 then 1 per camera
    finally:
        ing.stop()


def test_core_explicit_empty_heartbeat_without_detector() -> None:
    got = []
    ing = DetectionIngest(["cam_a"], port=0, on_set=got.append)
    ing.start()
    try:
        feed = _FrameFeed(["cam_a"])
        core = IsistreamCore(
            camera_ids=["cam_a"], frame_provider=feed,
            object_detector=None, pose_detector=None,   # pose-only system, no pose either
            ingest_addr=ing.address)
        core.tick()
        _drain(got, 1)
        assert len(got) == 1 and got[0].detections == []
    finally:
        ing.stop()


def test_core_pose_every_n_amortizes() -> None:
    calls = []

    class _PoseStub:
        def detect(self, pair):
            calls.append(pair.frame_idx)
            return {cid: [Detection(
                camera_id=cid, capture_ts=f.capture_ts, cls="person",
                confidence=0.9, bbox_xyxy=(0.0, 0.0, 50.0, 150.0),
                foot_uv=(25.0, 150.0),
                keypoints_uv=np.zeros((17, 3)))] for cid, f in pair.frames.items()}

    got = []
    ing = DetectionIngest(["cam_a"], port=0, on_set=got.append)
    ing.start()
    try:
        feed = _FrameFeed(["cam_a"])
        core = IsistreamCore(
            camera_ids=["cam_a"], frame_provider=feed,
            object_detector=None, pose_detector=_PoseStub(),
            ingest_addr=ing.address, pose_every_n=2)
        for _ in range(4):
            feed.advance()
            core.tick()
        _drain(got, 4)
        assert len(calls) == 2               # pose ran on ticks 2 and 4 only
        # Cached-emission semantics: once pose has run, every subsequent set
        # carries the LAST person result (off-pose ticks re-emit the cache) —
        # downstream sees a continuous stream, so only tick 1 is person-less.
        with_person = [ds for ds in got if ds.detections]
        assert len(with_person) == 3
        assert with_person[0].detections[0].keypoints_uv is not None
    finally:
        ing.stop()


def test_core_run_loop_paces_and_stops() -> None:
    got = []
    ing = DetectionIngest(["cam_a"], port=0, on_set=got.append)
    ing.start()
    try:
        feed = _FrameFeed(["cam_a"])

        class _AdvancingFeed:
            def __call__(self, camera_id):
                feed.advance(0.001)
                return feed(camera_id)

        core = IsistreamCore(
            camera_ids=["cam_a"], frame_provider=_AdvancingFeed(),
            object_detector=None, pose_detector=None,
            ingest_addr=ing.address, perception_fps=50.0)
        core.start()
        _drain(got, 3)
        core.stop()
        assert not core.running
        assert len(got) >= 3
        assert core.sets_sent["cam_a"] >= 3
    finally:
        ing.stop()
