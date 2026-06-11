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
