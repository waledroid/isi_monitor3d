"""``GET /api/config`` + ``POST /api/config`` — read state, atomic-write edits."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

import monitor_web.api.routes_config as routes_config
from monitor_web.app import create_app
from monitor_web.config import Settings

# ---- fixtures ----


@pytest.fixture
def populated_app(tmp_path: Path):
    """An app whose backbone.yaml + zones.yaml already exist on disk."""
    zones_path = tmp_path / "zones.yaml"
    zones_path.write_text(yaml.safe_dump({
        "zones": [{
            "name": "rack_a",
            "type": "storage",
            "kind": "storage",
            "severity": "info",
            "polygon": [[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]],
        }],
    }))
    backbone_yaml = tmp_path / "backbone.yaml"
    backbone_yaml.write_text(yaml.safe_dump({
        "cameras": {
            "cam_a": {"source": {"name": "rtsp", "url": "rtsp://a.example/Streaming",
                                  "latency_ms": 100}},
            "cam_b": {"source": {"name": "rtsp", "url": "rtsp://b.example/Streaming",
                                  "latency_ms": 100}},
        },
        "zones_path": str(zones_path),
    }))
    cfg = Settings(backbone_config_path=backbone_yaml, udp_port=0, port=0)
    return create_app(cfg), backbone_yaml, zones_path


@pytest.fixture
def empty_app(tmp_path: Path):
    """An app with no backbone.yaml on disk — covers the first-time-write path."""
    backbone_yaml = tmp_path / "backbone.yaml"
    cfg = Settings(backbone_config_path=backbone_yaml, udp_port=0, port=0)
    return create_app(cfg), backbone_yaml


# ---- GET /api/config ----


def test_get_config_returns_cameras_and_zones(populated_app) -> None:
    app, _, _ = populated_app
    with TestClient(app) as client:
        res = client.get("/api/config")
        assert res.status_code == 200
        data = res.json()
        assert data["cameras"]["cam_a"]["url"] == "rtsp://a.example/Streaming"
        assert data["cameras"]["cam_b"]["url"] == "rtsp://b.example/Streaming"
        assert len(data["zones"]) == 1
        assert data["zones"][0]["name"] == "rack_a"
        assert data["zones"][0]["kind"] == "storage"
        assert data["max_zones"] == 6
        assert "danger" in data["allowed_kinds"]
        assert "critical" in data["allowed_severities"]


def test_get_config_handles_absent_backbone_yaml(empty_app) -> None:
    app, _ = empty_app
    with TestClient(app) as client:
        res = client.get("/api/config")
        assert res.status_code == 200
        data = res.json()
        assert data["cameras"] == {}
        assert data["zones"] == []


# ---- GET /api/detection/onnx-files ----


def test_onnx_files_lists_trained_exports(populated_app, monkeypatch) -> None:
    """The picker endpoint surfaces the trainer's best.onnx exports, newest first.

    Monkeypatched so the test is hermetic (no dependency on a real trainer dir).
    """
    fake = [
        {"path": "/abs/runs/segment/models/yolo/seg_run/weights/best.onnx",
         "label": "segment/models/yolo/seg_run/weights/best.onnx", "mtime": 2.0},
        {"path": "/abs/runs/detect/models/yolo/det_run/weights/best.onnx",
         "label": "detect/models/yolo/det_run/weights/best.onnx", "mtime": 1.0},
    ]
    monkeypatch.setattr(
        "monitor_web.api.routes_config.list_trained_onnx", lambda: fake
    )
    app, _, _ = populated_app
    with TestClient(app) as client:
        res = client.get("/api/detection/onnx-files")
        assert res.status_code == 200
        files = res.json()["files"]
        # newest first; value is the absolute path, label is the path under runs/
        assert files[0]["label"].startswith("segment/")
        assert files[1]["label"].startswith("detect/")
        assert all(f["path"].startswith("/") and f["path"].endswith(".onnx") for f in files)


def test_onnx_files_empty_when_no_exports(populated_app, monkeypatch) -> None:
    monkeypatch.setattr(
        "monitor_web.api.routes_config.list_trained_onnx", lambda: []
    )
    app, _, _ = populated_app
    with TestClient(app) as client:
        res = client.get("/api/detection/onnx-files")
        assert res.status_code == 200
        assert res.json() == {"files": []}


# ---- POST /api/config ----


def test_post_config_writes_cameras_and_zones(populated_app) -> None:
    app, backbone_yaml, zones_path = populated_app
    payload = {
        "cameras": {
            "cam_a": {"url": "rtsp://NEW-A/Streaming"},
            "cam_b": {"url": "rtsp://NEW-B/Streaming"},
        },
        "zones": [
            {
                "name": "press_north",
                "type": "danger",
                "kind": "danger",
                "severity": "critical",
                "polygon": [[1.0, 0.0], [3.0, 0.0], [3.0, 1.5], [1.0, 1.5]],
            },
        ],
    }
    with TestClient(app) as client:
        res = client.post("/api/config", json=payload)
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["ok"] is True
        assert body["cameras_written"] == 2
        assert body["zones_written"] == 1

    # Re-read disk and verify both files were rewritten correctly.
    bb = yaml.safe_load(backbone_yaml.read_text())
    assert bb["cameras"]["cam_a"]["source"]["url"] == "rtsp://NEW-A/Streaming"
    assert bb["cameras"]["cam_b"]["source"]["url"] == "rtsp://NEW-B/Streaming"
    assert bb["cameras"]["cam_a"]["source"]["latency_ms"] == 100  # preserved

    zn = yaml.safe_load(zones_path.read_text())
    assert zn["zones"][0]["name"] == "press_north"
    assert zn["zones"][0]["kind"] == "danger"
    assert zn["zones"][0]["severity"] == "critical"
    assert zn["zones"][0]["polygon"] == [[1.0, 0.0], [3.0, 0.0], [3.0, 1.5], [1.0, 1.5]]


def test_post_config_without_zones_preserves_zones_yaml(populated_app) -> None:
    """The metric-zone editor was retired, so the dashboard omits `zones`. An
    omitted `zones` must leave zones.yaml exactly as it was (cameras still saved)."""
    app, backbone_yaml, zones_path = populated_app
    before = zones_path.read_text()
    with TestClient(app) as client:
        res = client.post("/api/config", json={"cameras": {"cam_a": {"url": "rtsp://X/s"}}})
        assert res.status_code == 200, res.text
        assert res.json()["zones_written"] == 0
    assert zones_path.read_text() == before          # untouched
    bb = yaml.safe_load(backbone_yaml.read_text())
    assert bb["cameras"]["cam_a"]["source"]["url"] == "rtsp://X/s"   # cameras still written


def test_post_config_rejects_polygon_below_three_points(populated_app) -> None:
    app, _, _ = populated_app
    payload = {
        "cameras": {},
        "zones": [{"name": "bad", "polygon": [[0.0, 0.0], [1.0, 0.0]]}],
    }
    with TestClient(app) as client:
        res = client.post("/api/config", json=payload)
        assert res.status_code == 422


def test_post_config_rejects_unknown_kind(populated_app) -> None:
    app, _, _ = populated_app
    payload = {
        "cameras": {},
        "zones": [{
            "name": "z",
            "kind": "not-a-kind",
            "polygon": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
        }],
    }
    with TestClient(app) as client:
        res = client.post("/api/config", json=payload)
        assert res.status_code == 422


def test_post_config_caps_zones_at_six(populated_app) -> None:
    app, _, _ = populated_app
    polygon = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
    zones = [{"name": f"z{i}", "polygon": polygon} for i in range(7)]
    with TestClient(app) as client:
        res = client.post("/api/config", json={"cameras": {}, "zones": zones})
        assert res.status_code == 422


# ---- S12: V4L2 device vs RTSP url camera config ----


def test_get_config_reports_camera_name_and_device(populated_app) -> None:
    """RTSP cameras report name=rtsp + url; device is null."""
    app, _, _ = populated_app
    with TestClient(app) as client:
        data = client.get("/api/config").json()
        assert data["cameras"]["cam_a"]["name"] == "rtsp"
        assert data["cameras"]["cam_a"]["url"] == "rtsp://a.example/Streaming"
        assert data["cameras"]["cam_a"]["device"] is None


def test_post_config_writes_v4l2_device(populated_app) -> None:
    app, backbone_yaml, _ = populated_app
    payload = {
        "cameras": {"cam_a": {"device": "/dev/video0"}},
        "zones": [],
    }
    with TestClient(app) as client:
        res = client.post("/api/config", json=payload)
        assert res.status_code == 200, res.text

    bb = yaml.safe_load(backbone_yaml.read_text())
    src = bb["cameras"]["cam_a"]["source"]
    assert src["name"] == "v4l2"
    assert src["device"] == "/dev/video0"
    assert "url" not in src              # RTSP-only key dropped


def test_post_config_switches_v4l2_back_to_rtsp(populated_app) -> None:
    """A camera saved as v4l2 then re-saved with a url drops the device key."""
    app, backbone_yaml, _ = populated_app
    with TestClient(app) as client:
        client.post("/api/config", json={
            "cameras": {"cam_a": {"device": "/dev/video1"}}, "zones": [],
        })
        client.post("/api/config", json={
            "cameras": {"cam_a": {"url": "rtsp://back-to-ip/Streaming"}}, "zones": [],
        })
    src = yaml.safe_load(backbone_yaml.read_text())["cameras"]["cam_a"]["source"]
    assert src["name"] == "rtsp"
    assert src["url"] == "rtsp://back-to-ip/Streaming"
    assert "device" not in src


def test_post_config_removes_cleared_cam_b(populated_app) -> None:
    """Clearing Cam 2 (omitting it from the payload) drops it from backbone.yaml
    — Mode 2 → Mode 1. Regression test for the 'cam2 still shows' bug."""
    app, backbone_yaml, _ = populated_app
    # populated_app seeds both cam_a + cam_b.
    assert "cam_b" in yaml.safe_load(backbone_yaml.read_text())["cameras"]
    with TestClient(app) as client:
        res = client.post("/api/config", json={
            "cameras": {"cam_a": {"url": "rtsp://only-a/Streaming"}},
            "zones": [],
        })
        assert res.status_code == 200, res.text
    cams = yaml.safe_load(backbone_yaml.read_text())["cameras"]
    assert "cam_a" in cams
    assert "cam_b" not in cams        # removed → Mode 1


def test_post_config_does_not_touch_unmanaged_cameras(populated_app) -> None:
    """A camera that isn't a dashboard-managed slot (cam_a/cam_b) — e.g. a
    hand-added cam_c — is left intact when the dashboard saves."""
    app, backbone_yaml, _ = populated_app
    # Hand-add a cam_c directly to the YAML.
    data = yaml.safe_load(backbone_yaml.read_text())
    data["cameras"]["cam_c"] = {"source": {"name": "rtsp", "url": "rtsp://hand/added"}}
    backbone_yaml.write_text(yaml.safe_dump(data))
    with TestClient(app) as client:
        client.post("/api/config", json={
            "cameras": {"cam_a": {"url": "rtsp://a/Streaming"}},
            "zones": [],
        })
    cams = yaml.safe_load(backbone_yaml.read_text())["cameras"]
    assert "cam_b" not in cams          # managed slot, cleared
    assert cams["cam_c"]["source"]["url"] == "rtsp://hand/added"   # untouched


def test_post_config_rejects_both_url_and_device(populated_app) -> None:
    app, _, _ = populated_app
    payload = {
        "cameras": {"cam_a": {"url": "rtsp://x/y", "device": "/dev/video0"}},
        "zones": [],
    }
    with TestClient(app) as client:
        res = client.post("/api/config", json=payload)
        assert res.status_code == 422


def test_post_config_rejects_camera_with_neither_url_nor_device(populated_app) -> None:
    app, _, _ = populated_app
    payload = {"cameras": {"cam_a": {}}, "zones": []}
    with TestClient(app) as client:
        res = client.post("/api/config", json=payload)
        assert res.status_code == 422


def test_post_config_first_time_creates_files(empty_app) -> None:
    """No backbone.yaml on disk yet — POST should create both files."""
    app, backbone_yaml = empty_app
    payload = {
        "cameras": {"cam_a": {"url": "rtsp://first/Streaming"}},
        "zones": [{
            "name": "z1",
            "polygon": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
        }],
    }
    with TestClient(app) as client:
        res = client.post("/api/config", json=payload)
        assert res.status_code == 200

    assert backbone_yaml.exists()
    bb = yaml.safe_load(backbone_yaml.read_text())
    assert bb["cameras"]["cam_a"]["source"]["url"] == "rtsp://first/Streaming"
    # zones.yaml was created beside backbone.yaml and registered in it.
    zones_path = Path(bb["zones_path"])
    assert zones_path.exists()
    zn = yaml.safe_load(zones_path.read_text())
    assert zn["zones"][0]["name"] == "z1"


def test_post_config_round_trips_through_get(populated_app) -> None:
    """A GET-after-POST returns the values just written."""
    app, _, _ = populated_app
    payload = {
        "cameras": {"cam_a": {"url": "rtsp://ROUND/Streaming"}},
        "zones": [{
            "name": "round_zone",
            "kind": "danger",
            "severity": "warning",
            "polygon": [[1.0, 1.0], [3.0, 1.0], [3.0, 2.0], [1.0, 2.0]],
        }],
    }
    with TestClient(app) as client:
        client.post("/api/config", json=payload)
        res = client.get("/api/config")
        data = res.json()
        assert data["cameras"]["cam_a"]["url"] == "rtsp://ROUND/Streaming"
        assert data["zones"][0]["name"] == "round_zone"
        assert data["zones"][0]["severity"] == "warning"


# ---- unified UI settings (server-side YAML store) ----


def test_ui_settings_round_trips(tmp_path: Path) -> None:
    backbone_yaml = tmp_path / "backbone.yaml"
    backbone_yaml.write_text(yaml.safe_dump({"cameras": {}}))
    cfg = Settings(
        backbone_config_path=backbone_yaml,
        ui_settings_path=tmp_path / "ui.yaml",
        udp_port=0, port=0,
    )
    app = create_app(cfg)
    with TestClient(app) as client:
        assert client.get("/api/ui-settings").json() == {}          # none yet
        r = client.post("/api/ui-settings", json={"mp4_selected": "video/p/clip.mp4"})
        assert r.status_code == 200 and r.json()["ok"] is True
        assert client.get("/api/ui-settings").json()["mp4_selected"] == "video/p/clip.mp4"
        # merge (not replace)
        client.post("/api/ui-settings", json={"lang": "fr"})
        got = client.get("/api/ui-settings").json()
        assert got["mp4_selected"] == "video/p/clip.mp4" and got["lang"] == "fr"



def _detection_app(tmp_path: Path):
    backbone_yaml = tmp_path / "backbone.yaml"
    backbone_yaml.write_text(yaml.safe_dump({"cameras": {}}))
    cfg = Settings(
        backbone_config_path=backbone_yaml,
        ui_settings_path=tmp_path / "ui.yaml",   # keep the repo's UI file untouched
        udp_port=0, port=0,
    )
    return create_app(cfg), backbone_yaml


def _force_gpu(monkeypatch, gpu: bool) -> None:
    monkeypatch.setattr(routes_config, "gpu_available", lambda: gpu)


def test_gpu_host_writes_onnx(tmp_path: Path, monkeypatch) -> None:
    _force_gpu(monkeypatch, True)
    app, backbone_yaml = _detection_app(tmp_path)
    with TestClient(app) as client:
        res = client.post("/api/config", json={"cameras": {}, "zones": [], "detection": {
            "onnx_path": "./models/best.onnx", "class_names": ["palette_vide"]}})
        assert res.status_code == 200, res.text
    det = yaml.safe_load(backbone_yaml.read_text())["detection"]
    assert det["plugin"] == "yolo_onnx"
    assert det["onnx_path"] == "./models/best.onnx"
    assert "model_xml" not in det and "device" not in det


def test_cpu_host_writes_openvino(tmp_path: Path, monkeypatch) -> None:
    _force_gpu(monkeypatch, False)
    app, backbone_yaml = _detection_app(tmp_path)
    with TestClient(app) as client:
        res = client.post("/api/config", json={"cameras": {}, "zones": [], "detection": {
            "model_xml": "./models/model.xml", "class_names": ["palette_vide"]}})
        assert res.status_code == 200, res.text
    det = yaml.safe_load(backbone_yaml.read_text())["detection"]
    assert det["plugin"] == "yolo_openvino"
    assert det["model_xml"] == "./models/model.xml"
    assert det["device"] == "AUTO"
    assert "onnx_path" not in det


def test_get_detection_backend_reflects_hardware(tmp_path: Path, monkeypatch) -> None:
    app, _ = _detection_app(tmp_path)
    with TestClient(app) as client:
        _force_gpu(monkeypatch, True)
        assert client.get("/api/config").json()["detection"]["backend"] == "yolo_onnx"
        _force_gpu(monkeypatch, False)
        assert client.get("/api/config").json()["detection"]["backend"] == "yolo_openvino"


def test_get_detection_remembers_inactive_path(tmp_path: Path, monkeypatch) -> None:
    """On a GPU host saving ONNX, a previously-entered OpenVINO xml is still shown."""
    _force_gpu(monkeypatch, True)
    app, _ = _detection_app(tmp_path)
    with TestClient(app) as client:
        # modal sends both inputs; only onnx is active on a GPU host
        client.post("/api/config", json={"cameras": {}, "zones": [], "detection": {
            "onnx_path": "./best.onnx", "model_xml": "./remembered.xml",
            "class_names": ["palette_vide"]}})
        det = client.get("/api/config").json()["detection"]
    assert det["backend"] == "yolo_onnx"
    assert det["onnx_path"] == "./best.onnx"
    assert det["model_xml"] == "./remembered.xml"   # inactive path still surfaced


def test_gpu_host_missing_onnx_is_400(tmp_path: Path, monkeypatch) -> None:
    _force_gpu(monkeypatch, True)
    app, _ = _detection_app(tmp_path)
    with TestClient(app) as client:
        res = client.post("/api/config", json={"cameras": {}, "zones": [], "detection": {
            "model_xml": "./only.xml", "class_names": ["palette_vide"]}})
        assert res.status_code == 400   # GPU host needs an onnx_path


def test_detection_requires_at_least_one_path(tmp_path: Path, monkeypatch) -> None:
    _force_gpu(monkeypatch, True)
    app, _ = _detection_app(tmp_path)
    with TestClient(app) as client:
        res = client.post("/api/config", json={"cameras": {}, "zones": [], "detection": {
            "class_names": ["palette_vide"]}})
        assert res.status_code == 422   # DetectionConfig: no path at all


def test_detection_show_nodes_round_trips_via_ui_settings(tmp_path: Path, monkeypatch) -> None:
    """show_nodes is a dashboard-only flag — persisted to UI settings, NOT
    backbone.yaml's detection block. Default True; toggling to False survives."""
    _force_gpu(monkeypatch, True)
    app, backbone_yaml = _detection_app(tmp_path)
    ui_path = tmp_path / "ui.yaml"
    with TestClient(app) as client:
        assert client.get("/api/config").json()["detection"]["show_nodes"] is True
        res = client.post("/api/config", json={"cameras": {}, "zones": [], "detection": {
            "onnx_path": "./best.onnx", "class_names": ["palette_vide"], "show_nodes": False,
        }})
        assert res.status_code == 200, res.text
        assert yaml.safe_load(ui_path.read_text())["show_nodes"] is False
        det_yaml = yaml.safe_load(backbone_yaml.read_text())["detection"]
        assert "show_nodes" not in det_yaml         # not polluting backbone.yaml
        assert client.get("/api/config").json()["detection"]["show_nodes"] is False


def test_detection_show_masks_round_trips_via_ui_settings(tmp_path: Path, monkeypatch) -> None:
    """show_masks (seg-overlay toggle) follows the same pattern as show_nodes —
    UI-only flag, never written to backbone.yaml's detection block."""
    _force_gpu(monkeypatch, True)
    app, backbone_yaml = _detection_app(tmp_path)
    ui_path = tmp_path / "ui.yaml"
    with TestClient(app) as client:
        assert client.get("/api/config").json()["detection"]["show_masks"] is True
        res = client.post("/api/config", json={"cameras": {}, "zones": [], "detection": {
            "onnx_path": "./best.onnx", "class_names": ["palette"], "show_masks": False,
        }})
        assert res.status_code == 200, res.text
        assert yaml.safe_load(ui_path.read_text())["show_masks"] is False
        det_yaml = yaml.safe_load(backbone_yaml.read_text())["detection"]
        assert "show_masks" not in det_yaml
        assert client.get("/api/config").json()["detection"]["show_masks"] is False


# ---- S16: distance-line rules round-trip through /api/config ----


def _link_lines_app(tmp_path: Path):
    backbone_yaml = tmp_path / "backbone.yaml"
    backbone_yaml.write_text(yaml.safe_dump({"cameras": {}}))
    cfg = Settings(
        backbone_config_path=backbone_yaml,
        ui_settings_path=tmp_path / "ui.yaml",
        link_lines_path=tmp_path / "link_lines.yaml",
        udp_port=0, port=0,
    )
    return create_app(cfg), tmp_path / "link_lines.yaml"


def test_get_config_returns_empty_link_lines_when_file_missing(tmp_path: Path) -> None:
    app, _ = _link_lines_app(tmp_path)
    with TestClient(app) as client:
        assert client.get("/api/config").json()["link_lines"] == []


def test_post_config_writes_link_lines_atomically(tmp_path: Path) -> None:
    app, _ll_path = _link_lines_app(tmp_path)
    payload = {
        "cameras": {}, "zones": [],
        "link_lines": [
            {"from": "person", "to": ["palette", "forklift"], "max_distance_m": 5.0},
            {"from": "forklift", "to": ["*"]},
        ],
    }
    with TestClient(app) as client:
        res = client.post("/api/config", json=payload)
        assert res.status_code == 200, res.text
        rules = client.get("/api/config").json()["link_lines"]   # merged-config round-trip
    assert isinstance(rules, list)
    assert rules[0]["from"] == "person"
    assert rules[0]["to"] == ["palette", "forklift"]
    assert rules[0]["max_distance_m"] == 5.0
    assert rules[1]["to"] == ["*"]


def test_link_lines_round_trip_via_get(tmp_path: Path) -> None:
    app, _ = _link_lines_app(tmp_path)
    payload = {"cameras": {}, "zones": [], "link_lines": [
        {"from": "person", "to": ["forklift"], "max_distance_m": 3.0, "color": "#ff0"},
    ]}
    with TestClient(app) as client:
        client.post("/api/config", json=payload)
        rules = client.get("/api/config").json()["link_lines"]
    assert len(rules) == 1
    assert rules[0]["from"] == "person"
    assert rules[0]["to"] == ["forklift"]
    assert rules[0]["max_distance_m"] == 3.0
    assert rules[0]["color"] == "#ff0"


def test_link_lines_empty_list_clears_existing_rules(tmp_path: Path) -> None:
    """Posting an empty link_lines list explicitly clears the saved rules."""
    app, _ = _link_lines_app(tmp_path)
    with TestClient(app) as client:
        # Save one rule first.
        client.post("/api/config", json={"cameras": {}, "zones": [], "link_lines": [
            {"from": "person", "to": ["palette"]}]})
        # Now clear them.
        res = client.post("/api/config", json={"cameras": {}, "zones": [], "link_lines": []})
        assert res.status_code == 200
        assert client.get("/api/config").json()["link_lines"] == []


def test_link_lines_omitted_does_not_touch_file(tmp_path: Path) -> None:
    """A POST without ``link_lines`` leaves the on-disk file alone (no change)."""
    app, ll_path = _link_lines_app(tmp_path)
    ll_path.write_text(yaml.safe_dump({"rules": [
        {"from": "person", "to": ["palette"]}]}))
    with TestClient(app) as client:
        res = client.post("/api/config", json={"cameras": {}, "zones": []})
        assert res.status_code == 200
    # Should still hold the original rule.
    doc = yaml.safe_load(ll_path.read_text())
    assert doc["rules"][0]["from"] == "person"


def test_link_lines_malformed_rule_rejected(tmp_path: Path) -> None:
    app, _ = _link_lines_app(tmp_path)
    payload = {"cameras": {}, "zones": [], "link_lines": [
        {"from": "person", "to": []},   # empty 'to' is invalid
    ]}
    with TestClient(app) as client:
        res = client.post("/api/config", json=payload)
        assert res.status_code == 422


# ---- mode-aware calibration_path (auto-warp per mode) ----


def test_save_sets_mode1_calibration_path(populated_app) -> None:
    """Saving a single-camera config points calibration_path at the Mode-1 file."""
    app, backbone_yaml, _ = populated_app
    with TestClient(app) as client:
        res = client.post("/api/config", json={
            "cameras": {"cam_a": {"url": "rtsp://a/x"}},
            "zones": [],
        })
        assert res.status_code == 200
    data = yaml.safe_load(backbone_yaml.read_text())
    assert data["calibration_path"].endswith("mode1/calibration.json")
    # metadata.sinks injected so the config is launchable.
    assert data["metadata"]["sinks"][0]["plugin"] == "udp"


def test_save_repoints_calibration_path_on_mode_switch(populated_app) -> None:
    """Adding Cam 2 (Mode 2) repoints calibration_path to the Mode-2 file in the
    same save; dropping back to one camera restores the Mode-1 file."""
    app, backbone_yaml, _ = populated_app
    with TestClient(app) as client:
        client.post("/api/config", json={
            "cameras": {"cam_a": {"url": "rtsp://a/x"}, "cam_b": {"url": "rtsp://b/x"}},
            "zones": [],
        })
        assert yaml.safe_load(backbone_yaml.read_text())["calibration_path"].endswith(
            "mode2/calibration.json")
        client.post("/api/config", json={
            "cameras": {"cam_a": {"url": "rtsp://a/x"}},
            "zones": [],
        })
        assert yaml.safe_load(backbone_yaml.read_text())["calibration_path"].endswith(
            "mode1/calibration.json")


# ---- pose model dropdown + persistence ----


def test_pose_onnx_files_lists_exports(populated_app, monkeypatch) -> None:
    fake = [{"path": "/abs/runs/pose/p/weights/best.onnx",
             "label": "trainer/isidet/runs/pose/p/weights/best.onnx", "mtime": 1.0}]
    monkeypatch.setattr("monitor_web.api.routes_config.list_pose_onnx", lambda: fake)
    app, _, _ = populated_app
    with TestClient(app) as client:
        res = client.get("/api/detection/pose-onnx-files")
        assert res.status_code == 200
        files = res.json()["files"]
        assert files[0]["path"].endswith("best.onnx")
        assert "pose" in files[0]["label"]


def test_save_persists_pose_path(populated_app) -> None:
    """A saved pose_onnx_path lands in backbone.yaml's detection block AND the
    UI-settings memory, and round-trips on GET; clearing it removes it."""
    app, backbone_yaml, _ = populated_app
    pose = "/abs/runs/pose/person/weights/best.onnx"
    with TestClient(app) as client:
        res = client.post("/api/config", json={
            "cameras": {"cam_a": {"url": "rtsp://a/x"}},
            "zones": [],
            "detection": {"onnx_path": "/abs/det/best.onnx",
                          "class_names": ["pallet"], "pose_onnx_path": pose},
        })
        assert res.status_code == 200, res.text
        det = yaml.safe_load(backbone_yaml.read_text())["detection"]
        assert det["pose_onnx_path"] == pose
        assert client.get("/api/config").json()["detection"]["pose_onnx_path"] == pose
        # Clear it.
        client.post("/api/config", json={
            "cameras": {"cam_a": {"url": "rtsp://a/x"}},
            "zones": [],
            "detection": {"onnx_path": "/abs/det/best.onnx",
                          "class_names": ["pallet"], "pose_onnx_path": None},
        })
        assert "pose_onnx_path" not in yaml.safe_load(backbone_yaml.read_text())["detection"]


def _tiny_rfdetr_onnx(tmp_path):
    """A minimal ONNX whose outputs are named like RF-DETR (dets/labels/masks),
    so the save's plugin selection + class inference fire without a 130 MB model."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    nodes, outs = [], []
    for name, shape in [("dets", (1, 2, 4)), ("labels", (1, 2, 5)), ("masks", (1, 2, 4, 4))]:
        const = numpy_helper.from_array(np.zeros(shape, np.float32), name=f"c_{name}")
        nodes.append(helper.make_node("Constant", [], [name], value=const))
        outs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, list(shape)))
    graph = helper.make_graph(nodes, "tinyrf", [], outs)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
    path = tmp_path / "tiny_rfdetr.onnx"
    onnx.save(model, str(path))
    return path


def test_rfdetr_model_save_sets_plugin_and_infers_classes(populated_app, tmp_path, monkeypatch):
    """Selecting an RF-DETR model: classes are inferred (no embedded names) and the
    saved plugin becomes rfdetr_onnx_seg with the YOLO-only knobs dropped."""
    app, backbone_yaml, _ = populated_app
    from monitor_web.api import routes_config
    monkeypatch.setattr(routes_config, "_detect_backend", lambda: "yolo_onnx")
    rf = _tiny_rfdetr_onnx(tmp_path)
    with TestClient(app) as client:
        cls = client.get("/api/detection/classes", params={"path": str(rf)}).json()["classes"]
        assert cls == ["palette", "carton", "polybag"]
        res = client.post("/api/config", json={
            "cameras": {}, "zones": [],
            "detection": {"onnx_path": str(rf), "class_names": cls, "confidence_threshold": 0.3},
        })
        assert res.status_code == 200, res.text
    det = yaml.safe_load(backbone_yaml.read_text())["detection"]
    assert det["plugin"] == "rfdetr_onnx_seg"
    assert det["onnx_path"] == str(rf)
    assert det["class_names"] == ["palette", "carton", "polybag"]
    assert "iou_threshold" not in det and "inference_imgsz" not in det


# ---- pose-only payload (the current Settings modal) ----


def test_pose_payload_updates_only_pose_keys(tmp_path: Path) -> None:
    """`pose` splices the pose keys into the detection block WITHOUT touching the
    object-model keys (those are managed per zone now)."""
    backbone_yaml = tmp_path / "backbone.yaml"
    backbone_yaml.write_text(yaml.safe_dump({
        "cameras": {},
        "detection": {"plugin": "yolo_onnx", "onnx_path": "./models/best.onnx",
                      "class_names": ["palette"], "confidence_threshold": 0.4},
    }))
    cfg = Settings(backbone_config_path=backbone_yaml,
                   ui_settings_path=tmp_path / "ui.yaml", udp_port=0, port=0)
    with TestClient(create_app(cfg)) as client:
        res = client.post("/api/config", json={"cameras": {}, "pose": {
            "pose_onnx_path": "/models/yolo11n-pose.onnx",
            "pose_confidence_threshold": 0.4}})
        assert res.status_code == 200, res.text
    det = yaml.safe_load(backbone_yaml.read_text())["detection"]
    assert det["pose_onnx_path"] == "/models/yolo11n-pose.onnx"
    assert det["pose_confidence_threshold"] == 0.4
    # Object-model keys are exactly as they were on disk.
    assert det["onnx_path"] == "./models/best.onnx"
    assert det["plugin"] == "yolo_onnx"
    assert det["class_names"] == ["palette"]
    assert det["confidence_threshold"] == 0.4


def test_pose_payload_empty_path_clears(tmp_path: Path) -> None:
    backbone_yaml = tmp_path / "backbone.yaml"
    backbone_yaml.write_text(yaml.safe_dump({
        "cameras": {},
        "detection": {"onnx_path": "./m.onnx", "pose_onnx_path": "/old-pose.onnx"},
    }))
    cfg = Settings(backbone_config_path=backbone_yaml,
                   ui_settings_path=tmp_path / "ui.yaml", udp_port=0, port=0)
    with TestClient(create_app(cfg)) as client:
        res = client.post("/api/config", json={"cameras": {}, "pose": {"pose_onnx_path": ""}})
        assert res.status_code == 200, res.text
    det = yaml.safe_load(backbone_yaml.read_text())["detection"]
    assert "pose_onnx_path" not in det
    assert det["onnx_path"] == "./m.onnx"


# ---- Communication: MQTT sink + node_id ----


def _comm_populated_app(tmp_path: Path):
    """App with backbone.yaml that already has a udp sink — mirrors populated_app."""
    backbone_yaml = tmp_path / "backbone.yaml"
    backbone_yaml.write_text(yaml.safe_dump({
        "cameras": {
            "cam_a": {"source": {"name": "rtsp", "url": "rtsp://a.example/Streaming"}},
        },
        "metadata": {
            "sinks": [{"plugin": "udp", "host": "127.0.0.1", "port": 50001}],
        },
    }))
    cfg = Settings(backbone_config_path=backbone_yaml, udp_port=0, port=0)
    return create_app(cfg), backbone_yaml


def test_get_config_returns_node_id_and_mqtt_sink(tmp_path: Path) -> None:
    """GET /api/config surfaces node_id and mqtt_sink keys."""
    backbone_yaml = tmp_path / "backbone.yaml"
    backbone_yaml.write_text(yaml.safe_dump({
        "cameras": {},
        "node_id": "wh-node-42",
        "metadata": {
            "sinks": [
                {"plugin": "udp", "host": "127.0.0.1", "port": 50001},
                {"plugin": "mqtt", "host": "broker.local", "port": 1883,
                 "prefix": "isi/wh-node-42"},
            ],
        },
    }))
    cfg = Settings(backbone_config_path=backbone_yaml, udp_port=0, port=0)
    with TestClient(create_app(cfg)) as client:
        res = client.get("/api/config")
        assert res.status_code == 200
        data = res.json()
        assert data["node_id"] == "wh-node-42"
        assert data["mqtt_sink"]["host"] == "broker.local"
        assert data["mqtt_sink"]["port"] == 1883
        assert data["mqtt_sink"]["prefix"] == "isi/wh-node-42"
        assert data["mqtt_sink"]["tls"] is False


def test_post_config_writes_mqtt_sink_preserves_udp_and_cameras(tmp_path: Path) -> None:
    """POST with node_id + mqtt_sink writes them; udp sink + cameras are untouched."""
    app, backbone_yaml = _comm_populated_app(tmp_path)
    payload = {
        "cameras": {"cam_a": {"url": "rtsp://a.example/Streaming"}},
        "node_id": "wh-node-1",
        "mqtt_sink": {
            "host": "mqtt.lan",
            "port": 1883,
            "tls": False,
            "ca_cert": "",
            "username": "",
            "password": "",
            "prefix": "isi/wh-node-1",
        },
    }
    with TestClient(app) as client:
        res = client.post("/api/config", json=payload)
        assert res.status_code == 200, res.text

    bb = yaml.safe_load(backbone_yaml.read_text())
    assert bb["node_id"] == "wh-node-1"
    sinks = bb["metadata"]["sinks"]
    plugins = [s["plugin"] for s in sinks]
    assert "udp" in plugins           # original udp sink preserved
    assert "mqtt" in plugins          # mqtt sink added
    mqtt = next(s for s in sinks if s["plugin"] == "mqtt")
    assert mqtt["host"] == "mqtt.lan"
    assert mqtt["port"] == 1883
    assert mqtt["prefix"] == "isi/wh-node-1"
    # cameras untouched
    assert bb["cameras"]["cam_a"]["source"]["url"] == "rtsp://a.example/Streaming"


def test_post_config_replaces_existing_mqtt_sink_not_duplicate(tmp_path: Path) -> None:
    """A second POST with mqtt_sink replaces the existing entry — exactly one mqtt."""
    backbone_yaml = tmp_path / "backbone.yaml"
    backbone_yaml.write_text(yaml.safe_dump({
        "cameras": {},
        "metadata": {
            "sinks": [
                {"plugin": "udp", "host": "127.0.0.1", "port": 50001},
                {"plugin": "mqtt", "host": "old-broker", "port": 1883, "prefix": "old/"},
            ],
        },
    }))
    cfg = Settings(backbone_config_path=backbone_yaml, udp_port=0, port=0)
    with TestClient(create_app(cfg)) as client:
        res = client.post("/api/config", json={
            "cameras": {},
            "mqtt_sink": {
                "host": "new-broker", "port": 8883, "tls": True,
                "ca_cert": "/etc/ssl/ca.pem", "username": "user",
                "password": "pass", "prefix": "isi/node2",
            },
        })
        assert res.status_code == 200, res.text

    bb = yaml.safe_load(backbone_yaml.read_text())
    sinks = bb["metadata"]["sinks"]
    mqtt_sinks = [s for s in sinks if s["plugin"] == "mqtt"]
    assert len(mqtt_sinks) == 1          # exactly one — not duplicated
    m = mqtt_sinks[0]
    assert m["host"] == "new-broker"
    assert m["port"] == 8883
    assert m["tls"] is True
    assert m["ca_cert"] == "/etc/ssl/ca.pem"
    assert m["username"] == "user"
    assert m["password"] == "pass"
    assert m["prefix"] == "isi/node2"
    udp_sinks = [s for s in sinks if s["plugin"] == "udp"]
    assert len(udp_sinks) == 1           # udp preserved


def test_post_config_without_mqtt_sink_leaves_sinks_untouched(tmp_path: Path) -> None:
    """POST without mqtt_sink leaves metadata.sinks exactly as it was on disk."""
    backbone_yaml = tmp_path / "backbone.yaml"
    original_sinks = [
        {"plugin": "udp", "host": "127.0.0.1", "port": 50001},
        {"plugin": "mqtt", "host": "existing-broker", "port": 1883, "prefix": "isi/node"},
    ]
    backbone_yaml.write_text(yaml.safe_dump({
        "cameras": {},
        "metadata": {"sinks": original_sinks},
    }))
    cfg = Settings(backbone_config_path=backbone_yaml, udp_port=0, port=0)
    with TestClient(create_app(cfg)) as client:
        res = client.post("/api/config", json={
            "cameras": {"cam_a": {"url": "rtsp://a/s"}},
        })
        assert res.status_code == 200, res.text

    bb = yaml.safe_load(backbone_yaml.read_text())
    sinks = bb["metadata"]["sinks"]
    # _ensure_launchable won't overwrite a non-empty sinks list, so we still
    # have both the original udp + mqtt entries.
    assert any(s["plugin"] == "mqtt" and s["host"] == "existing-broker" for s in sinks)
    assert any(s["plugin"] == "udp" for s in sinks)


# ---- camera_fps (decoupled FPS: video rate vs detection rate) ----


def _camera_fps_app(tmp_path: Path):
    """App with both cam_a and cam_b already in backbone.yaml."""
    backbone_yaml = tmp_path / "backbone.yaml"
    backbone_yaml.write_text(yaml.safe_dump({
        "cameras": {
            "cam_a": {"source": {"name": "rtsp", "url": "rtsp://a.example/Streaming",
                                  "capture_fps": 12}},
            "cam_b": {"source": {"name": "rtsp", "url": "rtsp://b.example/Streaming",
                                  "capture_fps": 12}},
        },
    }))
    cfg = Settings(backbone_config_path=backbone_yaml, udp_port=0, port=0)
    return create_app(cfg), backbone_yaml


def test_get_config_returns_camera_fps(tmp_path: Path) -> None:
    """GET /api/config exposes camera_fps (read from source.capture_fps, default 20)."""
    app, _ = _camera_fps_app(tmp_path)
    with TestClient(app) as client:
        data = client.get("/api/config").json()
    assert data["camera_fps"] == 12   # from backbone.yaml


def test_get_config_camera_fps_defaults_to_20_when_absent(tmp_path: Path) -> None:
    """When backbone.yaml has no capture_fps, GET returns camera_fps=20."""
    backbone_yaml = tmp_path / "backbone.yaml"
    backbone_yaml.write_text(yaml.safe_dump({
        "cameras": {
            "cam_a": {"source": {"name": "rtsp", "url": "rtsp://a.example/Streaming"}},
        },
    }))
    cfg = Settings(backbone_config_path=backbone_yaml, udp_port=0, port=0)
    with TestClient(create_app(cfg)) as client:
        data = client.get("/api/config").json()
    assert data["camera_fps"] == 20


def test_post_camera_fps_writes_to_both_cameras(tmp_path: Path) -> None:
    """POST camera_fps=20 writes source.capture_fps=20 to BOTH cam_a and cam_b."""
    app, backbone_yaml = _camera_fps_app(tmp_path)
    with TestClient(app) as client:
        res = client.post("/api/config", json={
            "cameras": {
                "cam_a": {"url": "rtsp://a.example/Streaming"},
                "cam_b": {"url": "rtsp://b.example/Streaming"},
            },
            "camera_fps": 20,
        })
        assert res.status_code == 200, res.text

    bb = yaml.safe_load(backbone_yaml.read_text())
    assert bb["cameras"]["cam_a"]["source"]["capture_fps"] == 20
    assert bb["cameras"]["cam_b"]["source"]["capture_fps"] == 20


def test_post_camera_fps_preserves_url_and_other_source_keys(tmp_path: Path) -> None:
    """camera_fps write must not clobber the url or latency_ms keys."""
    backbone_yaml = tmp_path / "backbone.yaml"
    backbone_yaml.write_text(yaml.safe_dump({
        "cameras": {
            "cam_a": {"source": {"name": "rtsp", "url": "rtsp://keep/Streaming",
                                  "latency_ms": 100, "capture_fps": 12}},
        },
    }))
    cfg = Settings(backbone_config_path=backbone_yaml, udp_port=0, port=0)
    with TestClient(create_app(cfg)) as client:
        res = client.post("/api/config", json={
            "cameras": {"cam_a": {"url": "rtsp://keep/Streaming"}},
            "camera_fps": 20,
        })
        assert res.status_code == 200, res.text

    src = yaml.safe_load(backbone_yaml.read_text())["cameras"]["cam_a"]["source"]
    assert src["url"] == "rtsp://keep/Streaming"
    assert src["latency_ms"] == 100
    assert src["capture_fps"] == 20


def test_post_camera_fps_round_trips_via_get(tmp_path: Path) -> None:
    """POST camera_fps=15 then GET returns camera_fps=15."""
    app, _ = _camera_fps_app(tmp_path)
    with TestClient(app) as client:
        client.post("/api/config", json={
            "cameras": {"cam_a": {"url": "rtsp://a.example/Streaming"}},
            "camera_fps": 15,
        })
        data = client.get("/api/config").json()
    assert data["camera_fps"] == 15


def test_post_camera_fps_omitted_leaves_existing_value(tmp_path: Path) -> None:
    """Omitting camera_fps from the POST leaves source.capture_fps untouched."""
    app, backbone_yaml = _camera_fps_app(tmp_path)
    with TestClient(app) as client:
        res = client.post("/api/config", json={
            "cameras": {"cam_a": {"url": "rtsp://a.example/Streaming"}},
        })
        assert res.status_code == 200, res.text

    src = yaml.safe_load(backbone_yaml.read_text())["cameras"]["cam_a"]["source"]
    assert src["capture_fps"] == 12   # unchanged from fixture


def test_post_camera_fps_clamped_to_1_30(tmp_path: Path) -> None:
    """Out-of-range camera_fps values are clamped (not rejected)."""
    app, backbone_yaml = _camera_fps_app(tmp_path)
    with TestClient(app) as client:
        # 0 → clamped to 1
        res = client.post("/api/config", json={
            "cameras": {"cam_a": {"url": "rtsp://a.example/Streaming"}},
            "camera_fps": 0,
        })
        assert res.status_code == 200, res.text
        src = yaml.safe_load(backbone_yaml.read_text())["cameras"]["cam_a"]["source"]
        assert src["capture_fps"] == 1

        # 99 → clamped to 30
        client.post("/api/config", json={
            "cameras": {"cam_a": {"url": "rtsp://a.example/Streaming"}},
            "camera_fps": 99,
        })
        src = yaml.safe_load(backbone_yaml.read_text())["cameras"]["cam_a"]["source"]
        assert src["capture_fps"] == 30


def test_isistream_save_writes_global_knobs_and_a_real_plugin(tmp_path, monkeypatch):
    """Settings ▸ Isistream: one model + global size/conf/SAHI/ENH for ALL zones.

    The plugin must be a REGISTERED implementation name derived from the
    model's outputs — never the model path (that crash-looped the producer).
    """
    import yaml as _yaml

    import backbone.detection  # noqa: F401 — fires the @register decorators
    from backbone.core.interfaces import detector_registry

    app, backbone_yaml = _detection_app(tmp_path)
    _force_gpu(monkeypatch, True)
    monkeypatch.setattr(routes_config, "_onnx_output_names",
                        lambda p: ["output0", "output1"])       # YOLO seg head

    with TestClient(app) as client:
        res = client.post("/api/config", json={
            "cameras": {"cam_a": {"url": "rtsp://x"}},
            "pose": {
                "pose_enabled": True, "pose_onnx_path": "", "pose_confidence_threshold": 0.3,
                "onnx_path": "/models/best.onnx",
                "zone_imgsz": 512, "confidence_threshold": 0.2,
                "sahi_enabled": True, "sahi_tile": 320, "sahi_overlap": 0.35,
                "enhance_enabled": True, "enhance_gamma": 1.2,
            },
        })
        assert res.status_code == 200, res.text

    det = _yaml.safe_load(backbone_yaml.read_text())["detection"]
    assert det["onnx_path"] == "/models/best.onnx"
    assert det["plugin"] in detector_registry.names(), det["plugin"]
    assert det["zone_imgsz"] == 512
    assert det["confidence_threshold"] == 0.2
    assert det["sahi"] == {"enabled": True, "tile": 320, "overlap": 0.35}
    assert det["enhance"] == {"enabled": True, "gamma": 1.2}


def test_isistream_save_clamps_out_of_range_knobs(tmp_path, monkeypatch):
    import yaml as _yaml

    app, backbone_yaml = _detection_app(tmp_path)
    _force_gpu(monkeypatch, True)
    with TestClient(app) as client:
        res = client.post("/api/config", json={
            "cameras": {"cam_a": {"url": "rtsp://x"}},
            "pose": {"pose_onnx_path": "", "zone_imgsz": 99999,
                     "confidence_threshold": 5.0, "sahi_overlap": 3.0,
                     "enhance_gamma": 99.0},
        })
        assert res.status_code == 200, res.text
    det = _yaml.safe_load(backbone_yaml.read_text())["detection"]
    assert det["zone_imgsz"] == 1280
    assert det["confidence_threshold"] == 1.0
    assert det["sahi"]["overlap"] == 0.9
    assert det["enhance"]["gamma"] == 3.0
