"""ZoneDetectionWorker / ZoneWorkerManager — the background zone RENDERER.

Doctrine (CLAUDE.md): isistream is the single source of ingestion + perception;
the dashboard renders, it never infers. These tests pin exactly that: the worker
holds no detector and renders the Backbone's per-camera ``ObservationsMessage``
from the UDP bus into ONE coherent snapshot per frame (single timestamp, atomic
swap), grouping wire detections into the operator's zone polygons, resolving
cross-zone overlaps, keeping persons out of zone cards, and idling when stopped.

A fake camera hub (synthetic frames) + a fake bus stand in — no camera, model, or
GPU is needed.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np
from backbone.comms.schemas import ObservationDet, ObservationsMessage

from monitor_web import zone_worker
from monitor_web.zone_worker import (
    SNAPSHOT_MAX_AGE_S,
    ZoneDetectionWorker,
    ZoneWorkerManager,
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


class _FakeBus:
    """app.state.bus stand-in: one per-camera ObservationsMessage."""

    def __init__(self, msg):
        self._msg = msg

    def snapshot(self):
        return SimpleNamespace(observations_by_camera=(
            {self._msg.camera_id: self._msg} if self._msg else {}))


def _obs_msg(cam="cam_a", dets=None, ts=None, frame_wh=(640, 480)):
    return ObservationsMessage(
        ts=ts if ts is not None else time.time(), camera_id=cam,
        frame_wh=frame_wh, dets=tuple(dets or []))


def _obs_det(cls="palette", conf=0.9, bbox=(40.0, 40.0, 120.0, 120.0), **extra):
    return ObservationDet(cls=cls, confidence=conf, bbox_xyxy=bbox,
                          foot_uv=((bbox[0] + bbox[2]) / 2.0, bbox[3]), **extra)


def _det(cls="palette", conf=0.9, bbox=(10.0, 10.0, 40.0, 40.0), cam="cam_a"):
    """A worker-shaped detection namespace (what _snapshot_from_bus produces)."""
    return SimpleNamespace(camera_id=cam, capture_ts=time.time(), cls=cls,
                           confidence=conf, bbox_xyxy=bbox,
                           foot_uv=((bbox[0] + bbox[2]) / 2.0, bbox[3]))


def _patch(zone_id, x0, y0, x1, y1, *, cam="cam_a", conf=None):
    """A zone patch whose polygon is its rect (drawn at the frame's own size,
    so no stored_wh rescale applies)."""
    return {
        "id": zone_id, "name": zone_id, "camera": cam,
        "polygon": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        "rect": [x0, y0, x1, y1], "frame_wh": [320, 240],
        "confidence": conf,
    }


class _CfgStub:
    """Settings stand-in: read_backbone/display_fps tolerate a missing file."""
    backbone_config_path = __import__("pathlib").Path("/nonexistent/backbone.yaml")
    ui_settings_path = "/nonexistent/ui.yaml"


def _worker(patches, bus=None, *, frame=None, running=True):
    """A worker wired to fakes (no detector — one perception, and it's not here)."""
    frame = frame if frame is not None else np.zeros((240, 320, 3), dtype=np.uint8)
    hub = FakeHub(frame)
    w = ZoneDetectionWorker(
        "cam_a", {"name": "rtsp", "url": "rtsp://x"}, _CfgStub(), lambda: running,
        hub_factory=lambda: hub,
        bus_getter=(lambda: bus) if bus is not None else None)
    w.set_patches(patches)
    return w, hub


# ---- doctrine: no local-inference entry point -------------------------------


def test_worker_module_exposes_no_local_inference_entry_point():
    """The dashboard never infers: the zone worker module holds no detector
    factory, no local-detect pass, and no per-zone-model machinery."""
    for gone in ("_zone_source", "_detect_all_zones", "_detect_zone",
                 "_detect_group_batched", "_detect_group_per_zone",
                 "_postprocess_zone", "_build_zone_crop", "_motion_signature",
                 "_merge_tile_dets", "_enhance_crop", "_remap_det",
                 "ZONE_SOURCE_KEY", "ZONE_SOURCE_DEFAULT", "get_zone_detector",
                 "ZoneModelUnavailable"):
        assert not hasattr(zone_worker, gone), f"{gone} must be gone (no local inference)"
    # The worker never accepts a detector factory.
    import inspect
    assert "detector_factory" not in inspect.signature(
        ZoneDetectionWorker.__init__).parameters
    assert "detector_factory" not in inspect.signature(
        ZoneWorkerManager.__init__).parameters
    # detection_overlay no longer re-exports the removed zone-detector symbols.
    import monitor_web.detection_overlay as overlay
    assert not hasattr(overlay, "get_zone_detector")
    assert not hasattr(overlay, "ZoneModelUnavailable")


# ---- snapshot coherence -----------------------------------------------------


def test_snapshot_covers_all_zones_with_one_timestamp():
    """All configured zones (not just the panelled ones) render on the SAME
    frame and land in ONE snapshot with a single frame_ts."""
    patches = [_patch(f"z{i}", 10 * i, 10, 10 * i + 40, 60) for i in range(1, 7)]
    # One wire detection centred in each zone.
    dets = [_obs_det(bbox=(10 * i + 5, 20, 10 * i + 35, 55)) for i in range(1, 7)]
    w, _hub = _worker(patches, bus=_FakeBus(_obs_msg(dets=dets, frame_wh=(320, 240))))
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    w._snapshot_from_bus(frame, patches)
    snap = w.snapshot()
    assert set(snap["zones"].keys()) == {f"z{i}" for i in range(1, 7)}
    assert isinstance(snap["frame_ts"], float) and snap["frame_ts"] > 0
    # Atomic swap: a second pass publishes a NEW dict object.
    before = snap
    w._snapshot_from_bus(frame, patches)
    assert w.snapshot() is not before


def test_zone_dets_goes_stale():
    """A snapshot older than SNAPSHOT_MAX_AGE_S yields [] — no stale ghosts."""
    patches = [_patch("z1", 0, 0, 160, 120)]
    det = _obs_det(bbox=(40.0, 40.0, 120.0, 120.0))
    w, _ = _worker(patches, bus=_FakeBus(_obs_msg(dets=[det], frame_wh=(320, 240))))
    w._snapshot_from_bus(np.zeros((240, 320, 3), dtype=np.uint8), patches)
    assert w.zone_dets("z1")          # fresh → present
    w._snapshot = {**w.snapshot(), "frame_ts": time.time() - SNAPSHOT_MAX_AGE_S - 0.5}
    assert w.zone_dets("z1") == []
    assert w.all_dets() == []


# ---- idle-when-stopped ------------------------------------------------------


def test_worker_idles_when_backbone_stopped():
    """is_running=False → empty snapshot, hub stream never acquired."""
    patches = [_patch("z1", 0, 0, 100, 100)]
    det = _obs_det(bbox=(20.0, 20.0, 80.0, 80.0))
    w, hub = _worker(patches, bus=_FakeBus(_obs_msg(dets=[det])), running=False)
    w.start()
    try:
        deadline = time.time() + 2.0
        while time.time() < deadline and w.snapshot()["frame_ts"] == 0.0:
            time.sleep(0.05)
        snap = w.snapshot()
        assert snap["zones"] == {}            # empty publish while stopped
        assert hub.acquired == 0              # never even acquired the camera
    finally:
        w.stop()


def test_worker_renders_when_running():
    """is_running=True → acquires the camera, renders the wire dets, publishes."""
    patches = [_patch("z1", 0, 0, 160, 120)]
    det = _obs_det(bbox=(40.0, 40.0, 120.0, 120.0))
    w, hub = _worker(patches, bus=_FakeBus(_obs_msg(dets=[det], frame_wh=(320, 240))))
    w.start()
    try:
        deadline = time.time() + 3.0
        while time.time() < deadline and not w.zone_dets("z1"):
            time.sleep(0.05)
        assert w.zone_dets("z1"), "worker never published a detection"
        assert hub.acquired == 1
    finally:
        w.stop()
    assert hub.released == 1                   # stream released on stop


def test_worker_has_no_operator_rate_limit():
    """Zone FPS limiting is gone: the worker is a pure renderer, paced by the
    camera and the producer's tick, not by an operator setting."""
    from monitor_web import zone_worker

    assert not hasattr(zone_worker, "display_fps")
    assert not hasattr(zone_worker, "DEFAULT_DETECTION_FPS")
    assert zone_worker._RENDER_FLOOR_S < 0.1     # a spin guard, not a cap



def test_overlap_resolved_to_deepest_zone():
    """Two zones report the same object (full-frame boxes that _same_object merges);
    the winner is the zone whose polygon contains the box centre the deepest."""
    pa = _patch("za", 0, 0, 100, 200)
    pb = _patch("zb", 60, 0, 300, 200)
    w, _ = _worker([pa, pb])
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
    w, _ = _worker([pa, pb])
    d1 = _det(bbox=(10.0, 10.0, 50.0, 50.0))
    d2 = _det(bbox=(200.0, 10.0, 240.0, 50.0))
    polys = {
        "za": np.array(pa["polygon"], dtype=np.float32),
        "zb": np.array(pb["polygon"], dtype=np.float32),
    }
    resolved = w._resolve_overlaps({"za": [d1], "zb": [d2]}, polys)
    assert len(resolved["za"]) == 1 and len(resolved["zb"]) == 1


# ---- manager topology --------------------------------------------------------


def test_manager_reload_topology(tmp_path, monkeypatch):
    """Workers track the config: cam_a zones → 1 worker; zones moved to cam_b →
    cam_a worker stopped+joined, cam_b worker running; no zones → none."""
    import yaml as _yaml
    bb = tmp_path / "backbone.yaml"
    bb.write_text(_yaml.safe_dump({"cameras": {
        "cam_a": {"source": {"name": "rtsp", "url": "rtsp://a"}},
        "cam_b": {"source": {"name": "rtsp", "url": "rtsp://b"}}}}))
    ui = tmp_path / "ui.yaml"
    ui.write_text(_yaml.safe_dump(
        {"zone_patches": {"patches": [_patch("z1", 0, 0, 50, 50)]}}))

    class Cfg:
        backbone_config_path = bb
        ui_settings_path = str(ui)

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    mgr = ZoneWorkerManager(
        Cfg(), is_running=lambda: False, hub_factory=lambda: FakeHub(frame))

    import monitor_web.dashboard_config as dc
    monkeypatch.setattr(dc, "unified_path", lambda cfg: ui)

    mgr.start()
    try:
        assert set(mgr._workers) == {"cam_a"}
        w_a = mgr._workers["cam_a"]

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


# ---- worker thread liveness --------------------------------------------------


def test_worker_start_stop_clean():
    """start/stop is idempotent and leaves no thread behind."""
    w, _hub = _worker([_patch("z1", 0, 0, 50, 50)], running=False)
    w.start()
    w.start()    # idempotent
    assert w._thread is not None and w._thread.is_alive()
    name = w._thread.name
    assert name == "zonedet[cam_a]"
    w.stop()
    assert w._thread is None
    assert not any(t.name == name for t in threading.enumerate())


# ---- backbone-sourced snapshots ---------------------------------------------


def test_backbone_source_renders_wire_observations():
    """The worker publishes the Backbone's observations — rescaled to the hub
    frame, grouped into the containing patch, occupancy hints intact — WITHOUT
    calling any detector."""
    patches = [_patch("z1", 0, 0, 160, 120)]
    # Observation in a 640x480 calibration frame; hub frames here are 320x240
    # (half) → the det should land at half coords, inside z1.
    det = ObservationDet(cls="palette", confidence=0.9,
                         bbox_xyxy=(40.0, 40.0, 120.0, 120.0),
                         foot_uv=(80.0, 120.0), occupancy_state="full",
                         occupancy_content="carton", occupancy_confidence=0.7,
                         mask_poly=((40.0, 40.0), (120.0, 40.0), (120.0, 120.0)))
    w, _ = _worker(patches, bus=_FakeBus(_obs_msg(dets=[det])))
    w._snapshot_from_bus(np.zeros((240, 320, 3), np.uint8), patches)
    dets = w.zone_dets("z1")
    assert len(dets) == 1
    d = dets[0]
    assert d.cls == "palette" and d.occupancy_state == "full"
    assert abs(d.bbox_xyxy[0] - 20.0) < 1e-6      # 40 * (320/640)
    assert abs(d.bbox_xyxy[3] - 60.0) < 1e-6      # 120 * (240/480)
    assert d.mask_poly and abs(d.mask_poly[0][0] - 20.0) < 1e-6
    assert w.zone_status("z1") == "ok"


def test_backbone_source_stale_observations_publish_empty():
    patches = [_patch("z1", 0, 0, 160, 120)]
    bus = _FakeBus(_obs_msg(dets=[], ts=time.time() - 10.0))   # stale
    w, _ = _worker(patches, bus=bus)
    w._snapshot_from_bus(np.zeros((240, 320, 3), np.uint8), patches)
    assert w.zone_dets("z1") == []


def test_backbone_source_persons_never_boxed():
    """A person-class observation never lands in a zone card — it rides the
    snapshot's people list (the map) and is drawn by pose on the cam views."""
    patches = [_patch("z1", 0, 0, 160, 120)]
    det = ObservationDet(cls="person", confidence=0.9,
                         bbox_xyxy=(40.0, 40.0, 120.0, 120.0), foot_uv=(80.0, 120.0))
    w, _ = _worker(patches, bus=_FakeBus(_obs_msg(dets=[det])))
    w._snapshot_from_bus(np.zeros((240, 320, 3), np.uint8), patches)
    assert w.zone_dets("z1") == []
    people = w.snapshot()["people"]
    assert len(people) == 1                       # person rides the people list
    assert abs(people[0]["foot_uv"][0] - 40.0) < 1e-6   # 80 * (320/640)


def test_backbone_source_no_bus_publishes_empty_ok():
    """No bus getter (pre-attach / bus absent) → empty zones, still 'ok'."""
    patches = [_patch("z1", 0, 0, 160, 120)]
    w, _ = _worker(patches)                        # bus_getter is None
    w._snapshot_from_bus(np.zeros((240, 320, 3), np.uint8), patches)
    assert w.zone_dets("z1") == []
    assert w.zone_status("z1") == "ok"


def test_no_per_zone_confidence_knob_remains():
    """ONE global model, ONE threshold (Settings > Isistream). The dashboard
    renders every detection the wire carries; it never re-filters per zone."""
    import inspect

    from monitor_web import zone_worker

    src = inspect.getsource(zone_worker)
    assert "conf_floor" not in src



def test_people_bridge_carries_last_seen_across_poseless_ticks():
    """The producer amortizes pose across ticks; a person-less tick within the
    bridge window keeps the last-seen people so the map doesn't blink."""
    patches = [_patch("z1", 0, 0, 160, 120)]
    person = ObservationDet(cls="person", confidence=0.9,
                            bbox_xyxy=(40.0, 40.0, 120.0, 120.0), foot_uv=(80.0, 120.0))
    # First tick carries a person.
    w, _ = _worker(patches, bus=_FakeBus(_obs_msg(dets=[person])))
    w._snapshot_from_bus(np.zeros((240, 320, 3), np.uint8), patches)
    assert len(w.snapshot()["people"]) == 1
    # Next tick: no person on the wire → bridged from the cache.
    w._bus_getter = lambda: _FakeBus(_obs_msg(dets=[]))
    w._snapshot_from_bus(np.zeros((240, 320, 3), np.uint8), patches)
    assert len(w.snapshot()["people"]) == 1, "person bridged across the poseless tick"


# ---- panel renderer (routes_video) is detection-free ------------------------


def test_zone_render_iter_never_detects():
    """The panel renderer draws the worker's dets on a crop — no detector,
    structurally (routes_video no longer imports one)."""
    from monitor_web.api.routes_video import _zone_render_iter

    frames = [np.zeros((240, 320, 3), dtype=np.uint8) for _ in range(2)]
    dets = [_det(bbox=(120.0, 60.0, 160.0, 100.0))]      # full-frame coords

    class Cfg:
        ui_settings_path = "/nonexistent/ui.yaml"
        backbone_config_path = __import__("pathlib").Path("/nonexistent/bb.yaml")

    # Running: draws the worker's dets on the crop.
    out = list(_zone_render_iter(iter(frames), Cfg(), "cam_a",
                                 rect=[100, 50, 300, 200], stored_wh=[320, 240],
                                 is_running=lambda: True, get_dets=lambda: dets))
    assert len(out) == 2 and all(f.size for f in out)
    # Stopped: raw crop passthrough (crop of zeros stays zeros).
    out = list(_zone_render_iter(iter(frames), Cfg(), "cam_a",
                                 rect=[100, 50, 300, 200], stored_wh=[320, 240],
                                 is_running=lambda: False, get_dets=lambda: dets))
    assert len(out) == 2 and all((f == 0).all() for f in out)


def test_to_crop_handles_wire_observation_dets():
    """The panel renderer's crop translation must accept the wire's namespace
    dets (no camera_id/capture_ts, mask as POLYGON) and translate the polygon
    into crop coords."""
    from monitor_web.api.routes_video import _to_crop

    wire_det = SimpleNamespace(
        cls="palette", confidence=0.9, bbox_xyxy=(120.0, 130.0, 220.0, 230.0),
        foot_uv=(170.0, 230.0), mask=None,
        mask_poly=[[120.0, 130.0], [220.0, 130.0], [220.0, 230.0]],
        occupancy_state="full", occupancy_content="carton",
        occupancy_confidence=0.7)
    c = _to_crop(wire_det, 100, 100, 200, 200)
    assert c.bbox_xyxy == (20.0, 30.0, 120.0, 130.0)
    assert c.foot_uv == (70.0, 130.0)
    assert c.mask_poly[0] == [20.0, 30.0]
    assert c.occupancy_state == "full"
