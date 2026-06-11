"""/api/warehouse-map round-trip + /api/warp-snapshot."""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import yaml
from fastapi.testclient import TestClient

from monitor_web.app import create_app
from monitor_web.camera_hub import CameraStream
from monitor_web.config import Settings


def _app(tmp_path: Path):
    bb = tmp_path / "backbone.yaml"
    bb.write_text(yaml.safe_dump({"cameras": {}, "metadata": {"sinks": []}}))
    return create_app(Settings(backbone_config_path=bb, udp_port=0, port=0,
                               warehouse_map_path=tmp_path / "warehouse_map.yaml"))


def test_warehouse_map_empty_then_round_trip(tmp_path):
    with TestClient(_app(tmp_path)) as client:
        assert client.get("/api/warehouse-map").json() == {"elements": [], "outline": None}
        payload = {"elements": [{"id": "rack_a1", "type": "rack", "shape": "rectangle",
                                 "footprint": [[2, 0], [3.5, 0], [3.5, 1], [2, 1]],
                                 "height_m": 2.5, "label": "A1"}], "outline": None}
        assert client.post("/api/warehouse-map", json=payload).status_code == 200
        got = client.get("/api/warehouse-map").json()
        assert got["elements"][0]["id"] == "rack_a1"   # persisted via the merged config + round-trips


def test_warehouse_map_rejects_bad_payload(tmp_path):
    with TestClient(_app(tmp_path)) as client:
        bad = {"elements": [{"id": "x", "type": "ufo",
                             "footprint": [[0, 0], [1, 0], [1, 1]], "height_m": 1}]}
        assert client.post("/api/warehouse-map", json=bad).status_code == 400


def test_warp_snapshot_uncalibrated_404(tmp_path):
    bb = tmp_path / "backbone.yaml"
    bb.write_text(yaml.safe_dump(
        {"cameras": {"cam_a": {"source": {"name": "rtsp", "url": "rtsp://x/y"}}},
         "metadata": {"sinks": []}}))
    app = create_app(Settings(backbone_config_path=bb, udp_port=0, port=0,
                              warehouse_map_path=tmp_path / "warehouse_map.yaml"))
    with TestClient(app) as client:
        # configured but no calibration file present for the mode
        assert client.get("/api/warp-snapshot/cam_a").status_code == 404
        # unknown camera
        assert client.get("/api/warp-snapshot/cam_z").status_code == 404


def test_wait_for_real_frame_skips_placeholders():
    """wait_for_real_frame returns the first NON-placeholder frame, not the
    pump's initial 'connecting…' placeholder (the warp-snapshot bug)."""
    stream = CameraStream("cam_a", "rtsp", {"url": "rtsp://x/y"})
    real = np.ones((4, 4, 3), dtype=np.uint8)

    # Initial state mirrors the pump: a placeholder is already published.
    stream._publish(np.zeros((4, 4, 3), dtype=np.uint8), placeholder=True)

    def producer():
        # A couple more placeholders, then a real decoded frame.
        stream._publish(np.zeros((4, 4, 3), dtype=np.uint8), placeholder=True)
        stream._publish(real, placeholder=False)

    threading.Timer(0.02, producer).start()
    got = stream.wait_for_real_frame(timeout=2.0)
    assert got is not None
    assert np.array_equal(got, real)


def test_wait_for_real_frame_times_out_on_only_placeholders():
    """No real frame within the timeout → None (caller falls back, no 500)."""
    stream = CameraStream("cam_a", "rtsp", {"url": "rtsp://x/y"})
    stream._publish(np.zeros((4, 4, 3), dtype=np.uint8), placeholder=True)
    assert stream.wait_for_real_frame(timeout=0.05) is None


def test_map_twin_degrades_without_calibration(tmp_path):
    """/api/map/twin must answer gracefully (available: false) when no calibration
    exists — the 3D map falls back to having no detection twin, never a 500."""
    with TestClient(_app(tmp_path)) as client:
        res = client.get("/api/map/twin")
        assert res.status_code == 200
        body = res.json()
        assert body["available"] is False
        assert "reason" in body
