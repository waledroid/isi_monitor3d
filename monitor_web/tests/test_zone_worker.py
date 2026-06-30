"""ZoneDetectionWorker / ZoneWorkerManager — the background zone-detection driver.

These tests use a fake camera hub (synthetic frames) and a fake detector factory,
so no camera, model, or GPU is needed. They pin the properties the revamp exists
for: ONE coherent snapshot per frame (single timestamp, atomic swap), all
configured zones detected (not just the 2 panelled ones), idle-when-stopped,
cross-zone overlap resolution, and a detection-free panel renderer.
"""

from __future__ import annotations

import threading
import time

import numpy as np
from backbone.core.types import Detection

from monitor_web.zone_worker import (
    SNAPSHOT_MAX_AGE_S,
    ZoneDetectionWorker,
    ZoneWorkerManager,
    _remap_det,
)

# ---- fakes ------------------------------------------------------------------

class FakeStream:
    """Stands in for camera_hub.CameraStream: hands out a synthetic frame."""

    def __init__(self, frame):
        self._frame = frame

    def latest_real_frame(self):
        return self._frame


class FakeHub:
    """Records acquire/release so tests can assert stream lifecycle."""

    def __init__(self, frame):
        self.frame = frame
        self.acquired = 0
        self.released = 0

    def acquire(self, camera_id, plugin, src_cfg):
        self.acquired += 1
        return FakeStream(self.frame)

    def release(self, stream):
        self.released += 1


def _det(cls="palette", conf=0.9, bbox=(10.0, 10.0, 40.0, 40.0), cam="cam_a"):
    return Detection(camera_id=cam, capture_ts=time.time(), cls=cls,
                     confidence=conf, bbox_xyxy=bbox,
                     foot_uv=((bbox[0] + bbox[2]) / 2.0, bbox[3]))


class FakeDetector:
    """Returns a fixed list of fed-image-coordinate detections for every call."""

    def __init__(self, dets):
        self._dets = dets
        self.calls = 0

    def detect(self, pair):
        self.calls += 1
        cam = next(iter(pair.frames))
        return {cam: list(self._dets)}


class FakeBatchDetector:
    """Echoes one centred detection per frame key in the FramePair. Records the
    number of detect() calls and the frame count of each call so tests can assert
    batched (one call, N frames) vs per-zone (N calls, 1 frame each).

    ``supports_batch`` is configurable. ``raise_on_pixel`` makes detect() raise
    whenever ANY fed frame contains a pixel >= that value — a content marker that
    fires in BOTH the batched call and the failing zone's per-zone fallback call
    (which is keyed by camera_id, not zone_id), so it exercises breaker isolation."""

    def __init__(self, *, supports_batch=True, raise_on_pixel=None):
        self.supports_batch = supports_batch
        self._raise_on_pixel = raise_on_pixel
        self.calls = 0
        self.frames_per_call: list[int] = []
        self.keys_per_call: list[list[str]] = []

    def detect(self, pair):
        self.calls += 1
        keys = list(pair.frames.keys())
        self.frames_per_call.append(len(keys))
        self.keys_per_call.append(keys)
        if self._raise_on_pixel is not None and any(
                int(pair.frames[k].image.max()) >= self._raise_on_pixel for k in keys):
            raise RuntimeError("boom on poison pixel")
        # Echo one det per key, centred in that key's fed image so it survives the
        # polygon clip (crops here are square so the centre is well inside).
        out = {}
        for k in keys:
            fr = pair.frames[k]
            h, w = fr.image.shape[:2]
            cx, cy = w / 2.0, h / 2.0
            out[k] = [_det(bbox=(cx - 5, cy - 5, cx + 5, cy + 5))]
        return out


def _patch(zone_id, x0, y0, x1, y1, *, cam="cam_a", conf=None):
    """A zone patch whose polygon is its rect (drawn at the frame's own size,
    so no stored_wh rescale applies)."""
    return {
        "id": zone_id, "name": zone_id, "camera": cam,
        "polygon": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        "rect": [x0, y0, x1, y1], "frame_wh": [320, 240],
        "infer_size": 320, "confidence": conf,
    }


def _worker(patches, dets_by_call=None, *, frame=None, running=True):
    """A worker wired to fakes. detector_factory returns ONE FakeDetector shared by
    all zones (mirrors the shared-session reality)."""
    frame = frame if frame is not None else np.zeros((240, 320, 3), dtype=np.uint8)
    hub = FakeHub(frame)
    detector = FakeDetector(dets_by_call or [])
    w = ZoneDetectionWorker(
        "cam_a", {"name": "rtsp", "url": "rtsp://x"}, _CfgStub(), lambda: running,
        detector_factory=lambda model, cfg, size: detector,
        hub_factory=lambda: hub,
    )
    w.set_patches(patches)
    return w, hub, detector


def _batch_worker(patches, *, detectors_by_key=None, default_supports_batch=True,
                  frame=None, running=True):
    """A worker whose detector_factory returns ONE detector per (model, infer_size)
    key — mirrors detection_overlay.get_zone_detector's shared-session cache. Tests
    can pre-seed specific detectors per key via ``detectors_by_key``; any other key
    gets a fresh FakeBatchDetector(supports_batch=default_supports_batch).

    Returns (worker, hub, factory) where factory.made maps key→detector built."""
    frame = frame if frame is not None else np.zeros((240, 320, 3), dtype=np.uint8)
    hub = FakeHub(frame)
    seed = dict(detectors_by_key or {})

    class _Factory:
        def __init__(self):
            self.made: dict = {}

        def __call__(self, model, cfg, size):
            key = (model, int(size))
            det = self.made.get(key)
            if det is None:
                det = seed.get(key) or FakeBatchDetector(
                    supports_batch=default_supports_batch)
                self.made[key] = det
            return det

    factory = _Factory()
    w = ZoneDetectionWorker(
        "cam_a", {"name": "rtsp", "url": "rtsp://x"}, _CfgStub(), lambda: running,
        detector_factory=factory, hub_factory=lambda: hub,
    )
    w.set_patches(patches)
    return w, hub, factory


class _CfgStub:
    """Settings stand-in: read_backbone/display_fps tolerate a missing file."""
    backbone_config_path = __import__("pathlib").Path("/nonexistent/backbone.yaml")
    ui_settings_path = "/nonexistent/ui.yaml"


# ---- snapshot coherence -----------------------------------------------------

def test_snapshot_covers_all_zones_with_one_timestamp():
    """All 6 configured zones (not just the 2 panelled ones) detect on the SAME
    frame and land in ONE snapshot with a single frame_ts."""
    patches = [_patch(f"z{i}", 10 * i, 10, 10 * i + 50, 60) for i in range(1, 7)]
    # Detections at the centre of the fed crop (50x50 crops; centre inside polygon).
    w, _hub, _detector = _worker(patches, [_det(bbox=(20.0, 20.0, 30.0, 30.0))])
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    w._detect_all_zones(frame, patches)
    snap = w.snapshot()
    assert set(snap["zones"].keys()) == {f"z{i}" for i in range(1, 7)}
    assert isinstance(snap["frame_ts"], float) and snap["frame_ts"] > 0
    # Atomic swap: a second pass publishes a NEW dict object.
    before = snap
    w._detect_all_zones(frame, patches)
    assert w.snapshot() is not before


def test_zone_dets_goes_stale():
    """A snapshot older than SNAPSHOT_MAX_AGE_S yields [] — no stale ghosts."""
    patches = [_patch("z1", 0, 0, 100, 100)]
    w, _, _ = _worker(patches, [_det(bbox=(40.0, 40.0, 60.0, 60.0))])
    w._detect_all_zones(np.zeros((240, 320, 3), dtype=np.uint8), patches)
    assert w.zone_dets("z1")          # fresh → present
    w._snapshot = {**w.snapshot(), "frame_ts": time.time() - SNAPSHOT_MAX_AGE_S - 0.5}
    assert w.zone_dets("z1") == []
    assert w.all_dets() == []


# ---- idle-when-stopped ------------------------------------------------------

def test_worker_idles_when_backbone_stopped():
    """is_running=False → no detector call, empty snapshot, hub stream released."""
    patches = [_patch("z1", 0, 0, 100, 100)]
    w, hub, det = _worker(patches, [_det()], running=False)
    w.start()
    try:
        deadline = time.time() + 2.0
        while time.time() < deadline and w.snapshot()["frame_ts"] == 0.0:
            time.sleep(0.05)
        snap = w.snapshot()
        assert snap["zones"] == {}            # empty publish while stopped
        assert det.calls == 0                 # never inferred
        assert hub.acquired == 0              # never even acquired the camera
    finally:
        w.stop()


def test_worker_detects_when_running():
    """is_running=True → acquires the camera, detects, publishes zone dets."""
    patches = [_patch("z1", 0, 0, 100, 100)]
    w, hub, det = _worker(patches, [_det(bbox=(40.0, 40.0, 60.0, 60.0))])
    w.start()
    try:
        deadline = time.time() + 3.0
        while time.time() < deadline and not w.zone_dets("z1"):
            time.sleep(0.05)
        assert w.zone_dets("z1"), "worker never published a detection"
        assert det.calls >= 1
        assert hub.acquired == 1
    finally:
        w.stop()
    assert hub.released == 1                  # stream released on stop


def test_loop_pace_reads_display_fps(monkeypatch):
    """The detect loop paces itself at the EDITABLE Zones-FPS (display_fps(cfg)),
    not a hardcoded constant. Patch display_fps → assert the post-detect
    self._stop.wait() interval is 1/display_fps."""
    import monitor_web.zone_worker as zw

    monkeypatch.setattr(zw, "display_fps", lambda cfg: 5.0)   # → 0.2 s loop wait

    patches = [_patch("z1", 0, 0, 100, 100)]
    w, _hub, _factory = _worker(patches, [_det(bbox=(40.0, 40.0, 60.0, 60.0))])

    waits: list[float] = []
    real_wait = w._stop.wait

    def spy_wait(timeout=None):
        waits.append(timeout)
        # Stop after we observe the post-detect pace wait so the test is quick.
        if timeout is not None and abs(timeout - 0.2) < 1e-6:
            w._stop.set()
        return real_wait(0)

    monkeypatch.setattr(w._stop, "wait", spy_wait)
    w._run()

    assert any(timeout is not None and abs(timeout - 0.2) < 1e-6 for timeout in waits), (
        f"loop pace must be 1/display_fps = 0.2 s; observed waits: {waits}")


# ---- cross-zone overlap resolution ------------------------------------------

def test_overlap_resolved_to_deepest_zone():
    """Two zones report the same object (full-frame boxes that _same_object merges);
    the winner is the zone whose polygon contains the box centre the deepest."""
    # zone A x:[0,100]; zone B x:[60,300] — overlap x:[60,100].
    pa = _patch("za", 0, 0, 100, 200)
    pb = _patch("zb", 60, 0, 300, 200)
    w, _, _ = _worker([pa, pb])
    # Same object at full-frame centre (80,100): 40px inside B, 20px from A's edge.
    d1 = _det(conf=0.9, bbox=(70.0, 90.0, 90.0, 110.0))
    d2 = _det(conf=0.8, bbox=(71.0, 91.0, 91.0, 111.0))
    polys = {
        "za": np.array(pa["polygon"], dtype=np.float32),
        "zb": np.array(pb["polygon"], dtype=np.float32),
    }
    resolved = w._resolve_overlaps({"za": [d1], "zb": [d2]}, polys)
    total = sum(len(v) for v in resolved.values())
    assert total == 1, f"object must land in exactly one zone, got {total}"
    assert len(resolved["zb"]) == 1           # deeper inside B → B owns it


def test_distinct_objects_kept_per_zone():
    """Two genuinely different objects stay, one per zone."""
    pa = _patch("za", 0, 0, 100, 200)
    pb = _patch("zb", 150, 0, 300, 200)
    w, _, _ = _worker([pa, pb])
    d1 = _det(bbox=(10.0, 10.0, 50.0, 50.0))
    d2 = _det(bbox=(200.0, 10.0, 240.0, 50.0))
    polys = {
        "za": np.array(pa["polygon"], dtype=np.float32),
        "zb": np.array(pb["polygon"], dtype=np.float32),
    }
    resolved = w._resolve_overlaps({"za": [d1], "zb": [d2]}, polys)
    assert len(resolved["za"]) == 1 and len(resolved["zb"]) == 1


# ---- zone-crop batching ------------------------------------------------------

def _frame():
    return np.zeros((240, 320, 3), dtype=np.uint8)


def test_batchable_same_model_zones_one_detect_call():
    """Case 1: N batchable zones sharing (model, infer_size) → ONE detect() call
    fed N frames, and each zone gets its own remapped detection."""
    patches = [_patch(f"z{i}", 10 + 60 * i, 10, 60 + 60 * i, 80) for i in range(3)]
    w, _hub, factory = _batch_worker(patches, default_supports_batch=True)
    w._detect_all_zones(_frame(), patches)
    det = factory.made[(None, 320)]
    assert det.calls == 1, "batchable group must be a single detect() call"
    assert det.frames_per_call == [3], "the one call must carry all 3 zone frames"
    assert det.keys_per_call[0] == ["z0", "z1", "z2"], "keyed by zone_id"
    snap = w.snapshot()
    # every zone produced exactly one detection, remapped into its own crop
    for i in range(3):
        zid = f"z{i}"
        assert len(snap["zones"][zid]) == 1
        assert snap["status"][zid] == "ok"
        bx = snap["zones"][zid][0].bbox_xyxy
        # det is centred in its crop → its full-frame x must lie inside the rect
        assert 10 + 60 * i <= (bx[0] + bx[2]) / 2.0 <= 60 + 60 * i


def test_non_batchable_runs_per_zone():
    """Case 2: supports_batch=False → N per-zone detect() calls, 1 frame each."""
    patches = [_patch(f"z{i}", 10 + 60 * i, 10, 60 + 60 * i, 80) for i in range(3)]
    w, _hub, factory = _batch_worker(patches, default_supports_batch=False)
    w._detect_all_zones(_frame(), patches)
    det = factory.made[(None, 320)]
    assert det.calls == 3, "non-batchable → one detect() per zone"
    assert det.frames_per_call == [1, 1, 1]
    snap = w.snapshot()
    assert all(snap["status"][f"z{i}"] == "ok" for i in range(3))


def test_batched_raise_falls_back_per_zone_and_breaker_isolates():
    """Case 3: a batched detect() that raises → per-zone fallback; the one broken
    zone hits the breaker ('error') while the others stay 'ok' (isolation)."""
    patches = [_patch("zok1", 0, 0, 60, 60),
               _patch("zbad", 120, 0, 180, 60),
               _patch("zok2", 240, 0, 300, 60)]
    # Poison ONLY the zbad rect in the source frame; the detector raises when it
    # sees a hot pixel. The batched call (carries zbad's crop) raises → per-zone
    # fallback, where only zbad's own per-zone call raises → breaker isolates it.
    frame = _frame()
    frame[0:60, 120:180] = 255
    det = FakeBatchDetector(supports_batch=True, raise_on_pixel=200)
    w, _hub, _factory = _batch_worker(
        patches, detectors_by_key={(None, 320): det})
    w._detect_all_zones(frame, patches)
    snap = w.snapshot()
    assert snap["status"]["zbad"] == "error"
    assert snap["status"]["zok1"] == "ok" and snap["status"]["zok2"] == "ok"
    assert snap["zones"]["zok1"] and snap["zones"]["zok2"]
    assert snap["zones"]["zbad"] == []
    # breaker recorded for the culprit only
    assert "zbad" in w._zone_breaker and "zok1" not in w._zone_breaker
    # call shape: 1 batched (raised) + 3 per-zone fallback
    assert det.calls == 4
    assert det.frames_per_call[0] == 3 and det.frames_per_call[1:] == [1, 1, 1]


def test_mixed_groups_behave_per_group():
    """Case 4: a batchable multi-zone group + a non-batchable (RF-DETR-like) zone +
    a single-zone batchable group each resolve independently."""
    # group A: model=None, size 320, 2 batchable zones → batched
    pa1 = _patch("a1", 0, 0, 60, 60)
    pa2 = _patch("a2", 70, 0, 130, 60)
    # group B: a different (resolvable) model, non-batchable → per-zone. Use an
    # absolute existing file so resolve_model returns a DISTINCT group key.
    rfdetr_path = __file__
    pb = _patch("b1", 140, 0, 200, 60)
    pb["model"] = rfdetr_path
    # group C: model=None but a DIFFERENT infer_size → its own single-zone group
    pc = _patch("c1", 210, 0, 270, 60)
    pc["infer_size"] = 256
    rfdetr = FakeBatchDetector(supports_batch=False)
    w, _hub, factory = _batch_worker(
        patches=[pa1, pa2, pb, pc],
        detectors_by_key={(rfdetr_path, 320): rfdetr})
    w._detect_all_zones(_frame(), [pa1, pa2, pb, pc])
    a = factory.made[(None, 320)]
    c = factory.made[(None, 256)]
    assert a.calls == 1 and a.frames_per_call == [2]    # batched pair
    assert rfdetr.calls == 1 and rfdetr.frames_per_call == [1]   # per-zone
    assert c.calls == 1 and c.frames_per_call == [1]    # single-zone → per-zone
    snap = w.snapshot()
    assert all(snap["status"][z] == "ok" for z in ("a1", "a2", "b1", "c1"))


def test_single_zone_batchable_group_uses_per_zone():
    """Case 5: a batchable detector but only ONE zone in the group → per-zone path
    (no point batching a single frame)."""
    patches = [_patch("solo", 0, 0, 60, 60)]
    w, _hub, factory = _batch_worker(patches, default_supports_batch=True)
    w._detect_all_zones(_frame(), patches)
    det = factory.made[(None, 320)]
    assert det.calls == 1 and det.frames_per_call == [1]
    assert w.snapshot()["status"]["solo"] == "ok"


def test_breaker_blocked_zone_excluded_from_batch():
    """Case 6: a zone whose breaker is open is excluded from grouping entirely —
    the batch only carries the allowed zones."""
    patches = [_patch("z0", 0, 0, 60, 60),
               _patch("zblock", 70, 0, 130, 60),
               _patch("z2", 140, 0, 200, 60)]
    w, _hub, factory = _batch_worker(patches, default_supports_batch=True)
    # pre-open the breaker for zblock far into the future
    w._zone_breaker["zblock"] = (time.monotonic() + 1000.0, "no_vram")
    w._detect_all_zones(_frame(), patches)
    det = factory.made[(None, 320)]
    assert det.calls == 1, "still one batched call for the 2 allowed zones"
    assert det.frames_per_call == [2]
    assert det.keys_per_call[0] == ["z0", "z2"]
    snap = w.snapshot()
    assert snap["status"]["zblock"] == "no_vram" and snap["zones"]["zblock"] == []
    assert snap["status"]["z0"] == "ok" and snap["status"]["z2"] == "ok"


def test_empty_crop_excluded_from_batch():
    """Case 7: a degenerate/zero-size crop is excluded from the batch and published
    as [] (still 'ok'); the rest batch normally."""
    # zone with a zero-width rect → empty crop
    bad = _patch("empty", 50, 50, 50, 120)     # x0 == x1 → zero width
    good1 = _patch("g1", 0, 0, 60, 60)
    good2 = _patch("g2", 70, 0, 130, 60)
    w, _hub, factory = _batch_worker([bad, good1, good2], default_supports_batch=True)
    w._detect_all_zones(_frame(), [bad, good1, good2])
    det = factory.made[(None, 320)]
    assert det.calls == 1
    assert det.frames_per_call == [2], "empty crop excluded from the batch"
    assert det.keys_per_call[0] == ["g1", "g2"]
    snap = w.snapshot()
    assert snap["zones"]["empty"] == [] and snap["status"]["empty"] == "ok"
    assert snap["zones"]["g1"] and snap["zones"]["g2"]


# ---- mask remap quality ------------------------------------------------------

def test_remap_det_mask_inter_linear_threshold():
    """Mask upscaling uses INTER_LINEAR + 0.5 threshold: a half-on mask upscales to
    a clean half-on full-frame mask (no nearest-neighbour blockiness asymmetry)."""
    fed_mask = np.zeros((10, 10), dtype=bool)
    fed_mask[:, 5:] = True                    # right half on
    d = Detection(camera_id="cam_a", capture_ts=0.0, cls="palette", confidence=0.9,
                  bbox_xyxy=(0.0, 0.0, 10.0, 10.0), foot_uv=(5.0, 10.0),
                  mask=fed_mask)
    out = _remap_det(d, x0=100, y0=50, rx=2.0, ry=2.0, iw=320, ih=240, ch=20, cw=20)
    assert out.mask.shape == (240, 320) and out.mask.dtype == bool
    assert not out.mask[:50, :].any() and not out.mask[:, :100].any()  # outside crop empty
    sub = out.mask[50:70, 100:120]
    assert sub[:, 12:].all() and not sub[:, :8].any()  # half-on preserved (soft edge ±2px)
    assert out.bbox_xyxy == (100.0, 50.0, 120.0, 70.0)  # bbox affine remap


# ---- manager topology ---------------------------------------------------------

def _manager_with(tmp_path, patches_yaml, cameras_yaml):
    import yaml as _yaml
    bb = tmp_path / "backbone.yaml"
    bb.write_text(_yaml.safe_dump({"cameras": cameras_yaml}))
    ui = tmp_path / "ui.yaml"
    ui.write_text(_yaml.safe_dump({"zone_patches": {"patches": patches_yaml}}))

    class Cfg:
        backbone_config_path = bb
        ui_settings_path = str(ui)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    hub = FakeHub(frame)
    mgr = ZoneWorkerManager(
        Cfg(), is_running=lambda: False,     # idle workers — topology test only
        detector_factory=lambda model, cfg, size: FakeDetector([]),
        hub_factory=lambda: hub,
    )
    return mgr, Cfg, ui


def test_manager_reload_topology(tmp_path, monkeypatch):
    """Workers track the config: cam_a zones → 1 worker; zones moved to cam_b →
    cam_a worker stopped+joined, cam_b worker running; no zones → none."""
    cams = {"cam_a": {"source": {"name": "rtsp", "url": "rtsp://a"}},
            "cam_b": {"source": {"name": "rtsp", "url": "rtsp://b"}}}
    mgr, _Cfg, ui = _manager_with(tmp_path, [_patch("z1", 0, 0, 50, 50)], cams)
    # load_patches reads the unified dashboard config through dashboard_config —
    # point it at our tmp ui.yaml via the settings stub used by the manager.
    import monitor_web.dashboard_config as dc
    monkeypatch.setattr(dc, "unified_path", lambda cfg: ui)

    mgr.start()
    try:
        assert set(mgr._workers) == {"cam_a"}
        w_a = mgr._workers["cam_a"]

        import yaml as _yaml
        ui.write_text(_yaml.safe_dump(
            {"zone_patches": {"patches": [_patch("z1", 0, 0, 50, 50, cam="cam_b")]}}))
        mgr.reload()
        assert set(mgr._workers) == {"cam_b"}
        assert w_a._thread is None or not w_a._thread.is_alive()   # joined

        ui.write_text(_yaml.safe_dump({"zone_patches": {"patches": []}}))
        mgr.reload()
        assert mgr._workers == {}
    finally:
        mgr.stop()


def test_manager_safe_on_missing_config(tmp_path):
    """Empty/missing config → start() is a no-op (no threads, no exceptions) —
    every TestClient app boots through this path."""
    class Cfg:
        backbone_config_path = tmp_path / "absent.yaml"
        ui_settings_path = str(tmp_path / "absent_ui.yaml")
    mgr = ZoneWorkerManager(Cfg(), is_running=lambda: False)
    mgr.start()
    assert mgr._workers == {}
    assert mgr.zone_dets("anything") == []
    assert mgr.camera_dets("cam_a") == []
    mgr.stop()


# ---- renderer is detection-free ----------------------------------------------

def test_zone_render_iter_never_detects(monkeypatch):
    """The panel renderer must not import/run any detector: with get_zone_detector
    forced to raise, frames still render; stopped → raw crop passthrough."""
    import monitor_web.detection_overlay as overlay
    from monitor_web.api.routes_video import _zone_render_iter

    def _boom(*a, **k):
        raise AssertionError("renderer must not build a detector")
    monkeypatch.setattr(overlay, "get_zone_detector", _boom)

    frames = [np.zeros((240, 320, 3), dtype=np.uint8) for _ in range(2)]
    dets = [_det(bbox=(120.0, 60.0, 160.0, 100.0))]      # full-frame coords

    class Cfg:
        ui_settings_path = "/nonexistent/ui.yaml"
        backbone_config_path = __import__("pathlib").Path("/nonexistent/bb.yaml")

    # Running: draws the worker's dets on the crop — no detector touched.
    out = list(_zone_render_iter(iter(frames), Cfg(), "cam_a",
                                 rect=[100, 50, 300, 200], stored_wh=[320, 240],
                                 infer_size=320, is_running=lambda: True,
                                 get_dets=lambda: dets))
    assert len(out) == 2 and all(f.size for f in out)
    # Stopped: raw crop passthrough (crop of zeros stays zeros).
    out = list(_zone_render_iter(iter(frames), Cfg(), "cam_a",
                                 rect=[100, 50, 300, 200], stored_wh=[320, 240],
                                 infer_size=320, is_running=lambda: False,
                                 get_dets=lambda: dets))
    assert len(out) == 2 and all((f == 0).all() for f in out)


# ---- SAHI slicing ------------------------------------------------------------

class WhiteBoxDetector:
    """Detects the white (non-zero) region in each fed frame and returns it as one
    detection in that frame's pixel coords. Lets a SAHI test place a real white box
    in the source frame and assert which tiles see it + where it remaps to.

    ``supports_batch`` toggles the batched-vs-sequential detect path. Records the
    number of detect() calls so a test can assert N tiles → 1 call (batched) vs N
    calls (sequential)."""

    def __init__(self, *, supports_batch=True, with_mask=False):
        self.supports_batch = supports_batch
        self._with_mask = with_mask
        self.calls = 0
        self.frames_per_call: list[int] = []

    def detect(self, pair):
        self.calls += 1
        keys = list(pair.frames.keys())
        self.frames_per_call.append(len(keys))
        out: dict[str, list] = {}
        for k in keys:
            img = pair.frames[k].image
            gray = img.max(axis=2) if img.ndim == 3 else img
            ys, xs = np.where(gray > 0)
            if xs.size == 0:
                out[k] = []
                continue
            x0, x1 = float(xs.min()), float(xs.max() + 1)
            y0, y1 = float(ys.min()), float(ys.max() + 1)
            mask = None
            if self._with_mask:
                mask = gray > 0
            out[k] = [Detection(camera_id=k, capture_ts=time.time(), cls="palette",
                                confidence=0.9, bbox_xyxy=(x0, y0, x1, y1),
                                foot_uv=((x0 + x1) / 2.0, y1), mask=mask)]
        return out


def _sahi_patch(zone_id, x0, y0, x1, y1, *, rows=2, cols=2, overlap=0.2,
                infer_size=320, cam="cam_a"):
    p = _patch(zone_id, x0, y0, x1, y1, cam=cam)
    p.update({"sahi": True, "sahi_rows": rows, "sahi_cols": cols,
              "sahi_overlap": overlap, "infer_size": infer_size})
    return p


def _sahi_worker(patches, detector, *, frame, running=True):
    hub = FakeHub(frame)
    w = ZoneDetectionWorker(
        "cam_a", {"name": "rtsp", "url": "rtsp://x"}, _CfgStub(), lambda: running,
        detector_factory=lambda model, cfg, size: detector,
        hub_factory=lambda: hub,
    )
    w.set_patches(patches)
    return w, hub


def test_sahi_tiles_cover_crop():
    """A 2x2 0.2-overlap grid over a zone crop produces 4 tiles whose union covers
    every crop pixel, with neighbours overlapping (no gaps)."""
    w, _, _ = _worker([], running=True)
    patch = _sahi_patch("z", 0, 0, 200, 160, rows=2, cols=2, overlap=0.2)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    tiles, zmeta = w._build_zone_tiles(frame, patch, 320, 240)
    assert len(tiles) == 4
    assert (zmeta["cw"], zmeta["ch"]) == (200, 160)
    cover = np.zeros((zmeta["ch"], zmeta["cw"]), dtype=bool)
    for _fed, tm in tiles:
        cover[tm["ty0"]:tm["ty0"] + tm["th"], tm["tx0"]:tm["tx0"] + tm["tw"]] = True
    assert cover.all(), "tiles must cover the whole crop"
    # overlap: tile widths sum to MORE than the crop width (shared bands).
    widths = sum(tm["tw"] for _f, tm in tiles[:2])   # the two top-row tiles
    assert widths > zmeta["cw"], "neighbouring tiles must overlap"


def test_sahi_boundary_box_merges_to_one(frame_size=(240, 320)):
    """A white box straddling a tile boundary is seen by two tiles → after merge
    there is ONE box at the correct source coords (no duplicate)."""
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    # Box centred on the vertical mid-seam of a 0..200 crop (seam near x=100).
    frame[60:100, 90:120] = 255          # source coords; inside crop [0,0,200,160]
    patch = _sahi_patch("z", 0, 0, 200, 160, rows=2, cols=2, overlap=0.2)
    det = WhiteBoxDetector(supports_batch=True)
    w, _ = _sahi_worker([patch], det, frame=frame)
    w._detect_all_zones(frame, [patch])
    snap = w.snapshot()
    dets = snap["zones"]["z"]
    assert len(dets) == 1, "the boundary-straddling box must merge to ONE detection"
    bx = dets[0].bbox_xyxy
    cx, cy = (bx[0] + bx[2]) / 2.0, (bx[1] + bx[3]) / 2.0
    assert 90 <= cx <= 120 and 60 <= cy <= 100, "merged box at the source location"


def test_sahi_mask_stitched():
    """A masked detection from a tile remaps into a full-frame mask aligned with the
    source white box."""
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[60:100, 40:70] = 255           # well inside the top-left tile
    patch = _sahi_patch("z", 0, 0, 200, 160, rows=2, cols=2, overlap=0.2,
                        infer_size=320)
    det = WhiteBoxDetector(supports_batch=True, with_mask=True)
    w, _ = _sahi_worker([patch], det, frame=frame)
    w._detect_all_zones(frame, [patch])
    dets = w.snapshot()["zones"]["z"]
    assert len(dets) == 1 and dets[0].mask is not None
    m = dets[0].mask
    assert m.shape == (240, 320)
    assert m[70:90, 45:65].all(), "mask covers the source box interior"


def test_sahi_detector_agnostic_same_result():
    """Batchable (N tiles → 1 detect call) and non-batchable (N calls) detectors
    yield the SAME merged result for the same crop."""
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[60:100, 40:70] = 255
    patch = _sahi_patch("z", 0, 0, 200, 160, rows=2, cols=2, overlap=0.2)

    det_b = WhiteBoxDetector(supports_batch=True)
    wb, _ = _sahi_worker([patch], det_b, frame=frame)
    wb._detect_all_zones(frame, [patch])
    out_b = wb.snapshot()["zones"]["z"]
    assert det_b.calls == 1 and det_b.frames_per_call == [4]

    det_s = WhiteBoxDetector(supports_batch=False)
    ws, _ = _sahi_worker([patch], det_s, frame=frame)
    ws._detect_all_zones(frame, [patch])
    out_s = ws.snapshot()["zones"]["z"]
    assert det_s.calls == 4 and det_s.frames_per_call == [1, 1, 1, 1]

    assert len(out_b) == len(out_s) == 1
    bb, bs = out_b[0].bbox_xyxy, out_s[0].bbox_xyxy
    for a, b in zip(bb, bs):
        assert abs(a - b) <= 2.0, "batched and sequential paths agree"


def test_sahi_false_parity_with_single_pass():
    """A sahi:false zone produces IDENTICAL output to the current single-pass path:
    the SAHI routing is byte-for-byte transparent when off."""
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[60:100, 40:70] = 255
    base = _patch("z", 0, 0, 200, 160)            # sahi absent ⇒ single pass
    det = WhiteBoxDetector(supports_batch=True)
    w, _ = _sahi_worker([base], det, frame=frame)
    w._detect_all_zones(frame, [base])
    out = w.snapshot()["zones"]["z"]
    assert det.calls == 1 and det.frames_per_call == [1], "single fed crop, no tiles"
    assert len(out) == 1
    bx = out[0].bbox_xyxy
    cx = (bx[0] + bx[2]) / 2.0
    assert 40 <= cx <= 70


def test_sahi_carry_forward_between_passes():
    """SAHI runs every SAHI_PERIOD-th genuine frame; between passes the snapshot is
    re-published with a bumped frame_ts and a valid_s wide enough that the carried
    boxes do not expire."""
    import monitor_web.zone_worker as zw
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[60:100, 40:70] = 255
    patch = _sahi_patch("z", 0, 0, 200, 160)
    det = WhiteBoxDetector(supports_batch=True)

    # Sequence of distinct frame objects so each loop tick sees a "new" frame.
    frames = [frame.copy() for _ in range(zw.SAHI_PERIOD + 1)]
    idx = {"i": 0}

    class SeqStream:
        def latest_real_frame(self):
            i = idx["i"]
            return frames[i] if i < len(frames) else frames[-1]

    class SeqHub:
        def acquire(self, *a):
            return SeqStream()

        def release(self, *a):
            pass

    w = ZoneDetectionWorker(
        "cam_a", {"name": "rtsp"}, _CfgStub(), lambda: True,
        detector_factory=lambda m, c, s: det, hub_factory=SeqHub)
    w.set_patches([patch])

    # Drive the worker loop body manually by stepping frames + the tick logic the
    # same way _run does, but synchronously and deterministically.
    snaps = []
    for i in range(zw.SAHI_PERIOD):
        idx["i"] = i
        cur = frames[i]
        w._sahi_tick += 1
        if (any(p.get("sahi") for p in [patch]) and w._sahi_tick % zw.SAHI_PERIOD != 0
                and w._snapshot.get("zones")):
            w._snapshot = {**w._snapshot, "frame_ts": time.time()}
        else:
            w._detect_all_zones(cur, [patch])
        snaps.append(dict(w._snapshot))

    # Heavy pass runs on the warm-start tick (no cache yet) AND the period tick;
    # the middle tick(s) carry the snapshot forward (no detect).
    assert det.calls == 2, "SAHI detects on warm start + once per period, carries between"
    assert det.frames_per_call == [4, 4], "each heavy pass slices into 4 tiles"
    final = w._snapshot
    assert final["zones"]["z"], "the heavy pass detected the box"
    # The published valid_s covers the skip window so carried boxes don't expire.
    assert final["valid_s"] >= zw.SAHI_PERIOD * (1.0 / 10.0)


# ---- worker thread liveness ---------------------------------------------------

def test_worker_start_stop_clean():
    """start/stop is idempotent and leaves no thread behind."""
    w, _, _ = _worker([_patch("z1", 0, 0, 50, 50)], running=False)
    w.start()
    w.start()    # idempotent
    assert w._thread is not None and w._thread.is_alive()
    name = w._thread.name
    assert name == "zonedet[cam_a]"
    w.stop()
    assert w._thread is None
    assert not any(t.name == name for t in threading.enumerate())
