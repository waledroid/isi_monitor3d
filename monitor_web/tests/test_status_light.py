"""3-state system status light (RED / AMBER / GREEN) on /api/status.

- GREEN  = Backbone running AND ≥1 camera delivering real frames (strict).
- AMBER  = all preconditions met + Backbone stopped → press START.
- RED    = a precondition missing, or the Backbone crashed.

Hermetic: the camera probe (`grab_real_frame`) is monkeypatched so we never
touch a real camera, and the supervisor state is swapped for a fake.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml
from calibration.schema import CALIBRATION_VERSION
from fastapi.testclient import TestClient

from monitor_web.api import routes_status
from monitor_web.app import create_app
from monitor_web.config import Settings


def _mode1_cal_doc(cam_id: str = "cam_a") -> str:
    H = np.diag([0.01, 0.01, 1.0]).tolist()
    return json.dumps({
        "version": CALIBRATION_VERSION,
        "created_at": "2026-06-08T00:00:00+00:00",
        "floor_anchor_method": "4pt_floor",
        "floor_origin_note": "test",
        "calibration_mode": "single_cam_4pt",
        "cameras": {cam_id: {
            "camera_id": cam_id, "image_size_wh": [1920, 1080],
            "K": np.eye(3).tolist(), "D": [0.0] * 5,
            "R": np.eye(3).tolist(), "t": [0.0, 0.0, 0.0],
            "H": H, "P": np.hstack([np.eye(3), np.zeros((3, 1))]).tolist(),
            "reprojection_rms_px": 0.0,
        }},
    })


def _build_app(tmp_path: Path, *, cameras=None, with_model=True, with_cal=True, with_sink=True):
    bb = tmp_path / "backbone.yaml"
    cams = cameras if cameras is not None else {"cam_a": {"source": {"name": "replay", "path": "x.mp4"}}}
    body: dict = {"cameras": cams}
    if with_model:
        onnx = tmp_path / "model.onnx"
        onnx.write_bytes(b"stub")
        body["detection"] = {"plugin": "yolo_onnx", "onnx_path": str(onnx)}
    if with_sink:
        body["metadata"] = {"sinks": [{"plugin": "udp", "host": "127.0.0.1", "port": 0}]}
    bb.write_text(yaml.safe_dump(body))
    if with_cal:
        (tmp_path / "mode1").mkdir(exist_ok=True) or (tmp_path / "mode1" / "calibration.json").write_text(_mode1_cal_doc("cam_a"))
    return create_app(Settings(backbone_config_path=bb, udp_port=0, port=0))


def _light(app, monkeypatch, *, state="stopped", camera_live=True) -> dict:
    monkeypatch.setattr(
        routes_status, "grab_real_frame",
        lambda *a, **k: (np.zeros((4, 4, 3), np.uint8) if camera_live else None),
    )
    with TestClient(app) as client:
        app.state.supervisor = SimpleNamespace(state=state, pid=None, last_exit_code=None)
        return client.get("/api/status").json()["readiness"]


def test_ready_and_stopped_is_amber(tmp_path, monkeypatch):
    r = _light(_build_app(tmp_path), monkeypatch, state="stopped", camera_live=True)
    assert r["light"] == "amber"
    assert r["ready"] is True


def test_running_with_live_camera_is_green(tmp_path, monkeypatch):
    r = _light(_build_app(tmp_path), monkeypatch, state="running", camera_live=True)
    assert r["light"] == "green"


def test_running_without_live_camera_is_red(tmp_path, monkeypatch):
    # Strict green: process up but no camera frames → red, not green.
    r = _light(_build_app(tmp_path), monkeypatch, state="running", camera_live=False)
    assert r["light"] == "red"
    assert r["checks"]["camera_live"] is False


def test_crashed_is_red(tmp_path, monkeypatch):
    r = _light(_build_app(tmp_path), monkeypatch, state="crashed", camera_live=True)
    assert r["light"] == "red"


def test_missing_calibration_is_red(tmp_path, monkeypatch):
    r = _light(_build_app(tmp_path, with_cal=False), monkeypatch, state="stopped", camera_live=True)
    assert r["light"] == "red"
    assert r["checks"]["calibration_ok"] is False


def test_missing_sink_is_red(tmp_path, monkeypatch):
    r = _light(_build_app(tmp_path, with_sink=False), monkeypatch, state="stopped", camera_live=True)
    assert r["light"] == "red"
    assert r["checks"]["sink_ok"] is False


def test_no_cameras_is_red(tmp_path, monkeypatch):
    r = _light(_build_app(tmp_path, cameras={}), monkeypatch, state="stopped", camera_live=True)
    assert r["light"] == "red"
    assert r["checks"]["config_ok"] is False
    assert r["checks"]["camera_live"] is False
