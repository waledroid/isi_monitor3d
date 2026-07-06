"""Zone-scoped detection (`backbone.detection.zone_scope`).

The system is zone-based: the object detector sees only the configured floor
zones' crops. These tests pin the geometry (zone polygon → per-camera crop
box), the detection remap (crop → frame pixels, incl. ingest-downscaled
frames), and the orchestrator wiring (scope default, no-zones ⇒ pose-only).
"""

from __future__ import annotations

import numpy as np
import yaml

from backbone.core.types import Detection, Frame, FramePair
from backbone.detection.zone_scope import (
    ZoneScopedDetector,
    zone_crop_boxes,
)
from backbone.shared.geometry import floor_homography_from_K_R_t
from backbone.shared.zones import ZoneRegistry

K = np.array([[1000.0, 0.0, 500.0], [0.0, 1000.0, 500.0], [0.0, 0.0, 1.0]])
R_LOOK_DOWN = np.diag([1.0, -1.0, -1.0])


class _View:
    def __init__(self, xy=(0.0, 0.0), image_size_wh=(1000, 1000)):
        self.K = K
        self.D = np.zeros(5)
        self.R = R_LOOK_DOWN
        self.t = np.array([xy[0], xy[1], 3.0])
        self.H = floor_homography_from_K_R_t(self.K, self.R, self.t)
        self.image_size_wh = image_size_wh


class _FakeRig:
    def __init__(self, views: dict):
        self._views = views

    @property
    def camera_ids(self):
        return tuple(self._views)

    def __getitem__(self, cam_id):
        return self._views[cam_id]

    def __contains__(self, cam_id):
        return cam_id in self._views


def _zones(polygon, name="Z1") -> ZoneRegistry:
    return ZoneRegistry.from_dict(
        {"zones": [{"name": name, "type": "palette", "polygon": polygon}]})


# ---- zone_crop_boxes -----------------------------------------------------


def test_crop_box_contains_projected_zone() -> None:
    """A floor zone around world (1, 0) — projecting to pixels near (833, 500)
    for the look-down camera at origin — must land inside the crop box, and
    the z=2m lift must extend the box beyond the pure floor footprint."""
    rig = _FakeRig({"cam_a": _View()})
    zones = _zones([[0.8, -0.2], [1.2, -0.2], [1.2, 0.2], [0.8, 0.2]])
    boxes = zone_crop_boxes(rig, zones)
    assert [name for name, _ in boxes["cam_a"]] == ["Z1"]
    x0, y0, x1, y1 = boxes["cam_a"][0][1]
    # Floor footprint: x in [766, 900], y in [433, 566] (u = 1000*X/3 + 500).
    assert x0 <= 766 and x1 >= 900
    assert y0 <= 433 and y1 >= 566
    # Height extent: at z=2 the same X spans u = 1000*X/1 + 500 → up to 1500+,
    # clipped to the frame — the box must be materially larger than the
    # floor-only footprint (not just the 16 px margin).
    assert (x1 - x0) > (900 - 766) + 2 * 16 + 1


def test_invisible_zone_produces_no_box() -> None:
    """A zone far outside the camera's field yields no crop for that camera."""
    rig = _FakeRig({"cam_a": _View()})
    zones = _zones([[100.0, 100.0], [101.0, 100.0], [101.0, 101.0], [100.0, 101.0]])
    boxes = zone_crop_boxes(rig, zones)
    assert boxes["cam_a"] == []


# ---- ZoneScopedDetector remap ---------------------------------------------


class _EchoDetector:
    """Reports one detection filling each crop — remap math becomes exact."""

    def detect(self, pair: FramePair):
        out = {}
        for sid, f in pair.frames.items():
            h, w = f.image.shape[:2]
            out[sid] = [Detection(
                camera_id=sid, capture_ts=f.capture_ts, cls="pallet",
                confidence=0.9, bbox_xyxy=(0.0, 0.0, float(w), float(h)),
                foot_uv=(w / 2.0, float(h)),
                keypoints_uv=np.array([[1.0, 2.0, 0.9]]),
                mask=np.ones((h, w), dtype=bool),
            )]
        return out


def _pair(images: dict) -> FramePair:
    frames = {cid: Frame(camera_id=cid, capture_ts=0.0, frame_idx=0, image=img)
              for cid, img in images.items()}
    return FramePair(capture_ts=0.0, frame_idx=0, frames=frames)


def test_detections_remap_to_frame_pixels() -> None:
    boxes = {"cam_a": [("Z1", (100, 200, 400, 500))]}
    det = ZoneScopedDetector(_EchoDetector(), boxes, {"cam_a": (1000, 1000)})
    out = det.detect(_pair({"cam_a": np.zeros((1000, 1000, 3), np.uint8)}))
    (d,) = out["cam_a"]
    assert d.camera_id == "cam_a"
    assert d.bbox_xyxy == (100.0, 200.0, 400.0, 500.0)   # crop-filling box → zone box
    assert d.foot_uv == (250.0, 500.0)                    # crop bottom-centre + offset
    assert tuple(d.keypoints_uv[0][:2]) == (101.0, 202.0)
    assert d.mask is None                                 # crop-sized masks are dropped


def test_remap_scales_with_ingest_downscale() -> None:
    """Boxes are calibration px (1000²); a half-size ingest frame (500²) crops
    at half coordinates and detections come back in FRAME pixels."""
    boxes = {"cam_a": [("Z1", (100, 200, 400, 500))]}
    det = ZoneScopedDetector(_EchoDetector(), boxes, {"cam_a": (1000, 1000)})
    out = det.detect(_pair({"cam_a": np.zeros((500, 500, 3), np.uint8)}))
    (d,) = out["cam_a"]
    assert d.bbox_xyxy == (50.0, 100.0, 200.0, 250.0)
    assert d.foot_uv == (125.0, 250.0)


def test_camera_without_zone_and_empty_boxes() -> None:
    boxes = {"cam_a": [("Z1", (100, 200, 400, 500))], "cam_b": []}
    det = ZoneScopedDetector(_EchoDetector(), boxes, {"cam_a": (1000, 1000),
                                                      "cam_b": (1000, 1000)})
    out = det.detect(_pair({
        "cam_a": np.zeros((1000, 1000, 3), np.uint8),
        "cam_b": np.zeros((1000, 1000, 3), np.uint8),
    }))
    assert len(out["cam_a"]) == 1 and out["cam_b"] == []


def test_no_crops_at_all_returns_empty_without_inner_call() -> None:
    class _Boom:
        def detect(self, pair):
            raise AssertionError("inner detector must not run with no crops")

    det = ZoneScopedDetector(_Boom(), {"cam_a": []}, {"cam_a": (1000, 1000)})
    out = det.detect(_pair({"cam_a": np.zeros((100, 100, 3), np.uint8)}))
    assert out == {"cam_a": []}


# ---- orchestrator wiring ---------------------------------------------------


def test_orchestrator_no_zones_means_no_object_detector(tmp_path) -> None:
    """Default scope is `zones`: with no zones configured the object detector
    is not even built (pose-only Backbone) and step() still runs."""
    from backbone.runtime import Orchestrator
    from tests.test_orchestrator import (
        CLASS_NAMES,
        _bind_receiver,
        _write_calibration,
        _write_stub_onnx,
    )

    cal = _write_calibration(tmp_path)
    onnx_path = _write_stub_onnx(tmp_path)
    sock, port = _bind_receiver()
    try:
        cfg = tmp_path / "bb.yaml"
        cfg.write_text(yaml.safe_dump({
            "calibration_path": str(cal),
            "cameras": {"cam_a": {"source": {"name": "replay", "frames": []}}},
            "detection": {"plugin": "yolo_onnx", "onnx_path": str(onnx_path),
                          "class_names": CLASS_NAMES,
                          "providers": ["CPUExecutionProvider"]},
            "homography": {"tracker": {"plugin": "bytetrack"},
                           "track_config": {"min_hits_to_confirm": 1}},
            "metadata": {"sinks": [{"plugin": "udp", "host": "127.0.0.1", "port": port}]},
        }))
        orch = Orchestrator(cfg)
        assert orch._detector is None
        img = np.zeros((1000, 1000, 3), np.uint8)
        t2, _ = orch.step(FramePair(capture_ts=0.0, frame_idx=0, frames={
            "cam_a": Frame(camera_id="cam_a", capture_ts=0.0, frame_idx=0, image=img)}))
        assert t2 == []                       # no zones → no object tracks
        orch.publisher.close()
    finally:
        sock.close()


def test_orchestrator_zone_scope_detects_inside_zone(tmp_path) -> None:
    """With a zone configured, scope=zones wraps the detector in crops and the
    pipeline still emits tracks (the stub anchor decodes inside the crop)."""
    from backbone.detection.zone_scope import ZoneScopedDetector as ZSD
    from backbone.runtime import Orchestrator
    from tests.test_orchestrator import (
        CLASS_NAMES,
        _bind_receiver,
        _write_calibration,
        _write_stub_onnx,
    )

    cal = _write_calibration(tmp_path)
    onnx_path = _write_stub_onnx(tmp_path)
    zones_path = tmp_path / "zones.yaml"
    zones_path.write_text(yaml.safe_dump({"zones": [{
        "name": "Z1", "type": "palette",
        "polygon": [[-1.5, -1.5], [1.5, -1.5], [1.5, 1.5], [-1.5, 1.5]],
    }]}))
    sock, port = _bind_receiver()
    try:
        cfg = tmp_path / "bb.yaml"
        cfg.write_text(yaml.safe_dump({
            "calibration_path": str(cal),
            "zones_path": str(zones_path),
            "cameras": {"cam_a": {"source": {"name": "replay", "frames": []}},
                        "cam_b": {"source": {"name": "replay", "frames": []}}},
            "detection": {"plugin": "yolo_onnx", "onnx_path": str(onnx_path),
                          "class_names": CLASS_NAMES,
                          "providers": ["CPUExecutionProvider"]},
            "homography": {"tracker": {"plugin": "bytetrack"},
                           "track_config": {"min_hits_to_confirm": 1}},
            "metadata": {"sinks": [{"plugin": "udp", "host": "127.0.0.1", "port": port}]},
        }))
        orch = Orchestrator(cfg)
        assert isinstance(orch._detector, ZSD)
        img = np.zeros((1000, 1000, 3), np.uint8)
        tracks: list = []
        for i in range(3):
            pair = FramePair(capture_ts=i * 0.033, frame_idx=i, frames={
                cid: Frame(camera_id=cid, capture_ts=i * 0.033, frame_idx=i, image=img)
                for cid in ("cam_a", "cam_b")})
            t2, _ = orch.step(pair)
            tracks = t2 or tracks
        assert tracks, "zone-scoped detection produced no tracks"
        # The zone spans [-1.5, 1.5]²; every track must sit inside it.
        for t in tracks:
            x, y = t.xy_m
            assert -1.6 <= x <= 1.6 and -1.6 <= y <= 1.6
        orch.publisher.close()
    finally:
        sock.close()
