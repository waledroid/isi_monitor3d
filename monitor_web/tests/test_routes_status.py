"""HTTP routes — status, zones, control endpoints via FastAPI TestClient."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from monitor_web.app import create_app
from monitor_web.config import Settings


@pytest.fixture
def app_with_settings(tmp_path: Path):
    """Build an app with config + zones written to a tmp_path."""
    zones_path = tmp_path / "zones.yaml"
    zones_path.write_text(yaml.safe_dump({
        "zones": [{
            "name": "rack_a",
            "type": "storage",
            "polygon": [[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]],
        }],
    }))
    backbone_yaml = tmp_path / "backbone.yaml"
    backbone_yaml.write_text(yaml.safe_dump({
        "calibration_path": str(tmp_path / "missing-cal.json"),
        "cameras": {"cam_a": {"source": {"name": "replay", "frames": []}}},
        "detection": {"plugin": "yolo_onnx", "onnx_path": "x.onnx", "class_names": ["person"]},
        "metadata": {"sinks": [{"plugin": "udp", "host": "127.0.0.1", "port": 0}]},
        "zones_path": str(zones_path),
    }))
    cfg = Settings(
        backbone_config_path=backbone_yaml,
        # OS-assigned port via 0 — bus_subscriber binds and the kernel picks.
        udp_port=0,
        port=0,
    )
    return create_app(cfg)


def test_status_endpoint_returns_expected_shape(app_with_settings) -> None:
    with TestClient(app_with_settings) as client:
        res = client.get("/api/status")
        assert res.status_code == 200
        data = res.json()
        assert "backbone" in data
        assert "udp" in data
        assert "tracks" in data
        assert data["backbone"]["state"] == "stopped"
        assert data["udp"]["fresh"] is False
        assert data["tracks"]["active_2d"] == 0


def test_zones_endpoint_returns_loaded_polygons(app_with_settings) -> None:
    with TestClient(app_with_settings) as client:
        res = client.get("/api/zones")
        assert res.status_code == 200
        data = res.json()
        assert len(data["zones"]) == 1
        z = data["zones"][0]
        assert z["name"] == "rack_a"
        assert z["type"] == "storage"
        assert z["kind"] == "palette"          # default category (kind omitted in YAML)
        assert z["severity"] == "info"         # default severity
        assert z["polygon"] == [[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]]


def test_zones_endpoint_exposes_kind_and_severity(tmp_path: Path) -> None:
    """S10: /api/zones must surface the Type-2 danger fields."""
    zones_path = tmp_path / "zones.yaml"
    zones_path.write_text(yaml.safe_dump({
        "zones": [{
            "name": "press_north",
            "type": "danger",
            "kind": "danger",
            "severity": "critical",
            "polygon": [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]],
        }],
    }))
    backbone_yaml = tmp_path / "backbone.yaml"
    backbone_yaml.write_text(yaml.safe_dump({"zones_path": str(zones_path)}))
    cfg = Settings(backbone_config_path=backbone_yaml, udp_port=0, port=0)
    app = create_app(cfg)
    with TestClient(app) as client:
        res = client.get("/api/zones")
        assert res.status_code == 200
        z = res.json()["zones"][0]
        assert z["kind"] == "danger"
        assert z["severity"] == "critical"


def test_danger_zones_object_endpoint_returns_class_radii(tmp_path: Path) -> None:
    """S10: /api/danger-zones-object exposes per-class proximity radii."""
    cfg_path = tmp_path / "danger_zones_object.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "classes": {
            "custom-robot": {"green_m": 3.0, "yellow_m": 1.5, "red_m": 0.5, "alpha": 0.2},
        },
    }))
    cfg = Settings(
        backbone_config_path=tmp_path / "no-backbone.yaml",
        danger_zones_object_path=cfg_path,
        udp_port=0, port=0,
    )
    app = create_app(cfg)
    with TestClient(app) as client:
        res = client.get("/api/danger-zones-object")
        assert res.status_code == 200
        data = res.json()
        assert "custom-robot" in data["classes"]
        assert data["classes"]["custom-robot"]["red_m"] == 0.5


def test_danger_zones_object_endpoint_handles_missing_file(tmp_path: Path) -> None:
    """Absent config file => empty classes map (no error)."""
    cfg = Settings(
        backbone_config_path=tmp_path / "no-backbone.yaml",
        danger_zones_object_path=tmp_path / "absent.yaml",
        udp_port=0, port=0,
    )
    app = create_app(cfg)
    with TestClient(app) as client:
        res = client.get("/api/danger-zones-object")
        assert res.status_code == 200
        assert res.json() == {"classes": {}}


def test_zones_endpoint_handles_missing_zones_path(tmp_path: Path) -> None:
    """If backbone.yaml has no zones_path, return empty list."""
    backbone_yaml = tmp_path / "backbone.yaml"
    backbone_yaml.write_text(yaml.safe_dump({
        "cameras": {},
        "metadata": {"sinks": []},
    }))
    cfg = Settings(backbone_config_path=backbone_yaml, udp_port=0, port=0)
    app = create_app(cfg)
    with TestClient(app) as client:
        res = client.get("/api/zones")
        assert res.status_code == 200
        assert res.json() == {"zones": []}


def test_control_state_endpoint_initially_stopped(app_with_settings) -> None:
    with TestClient(app_with_settings) as client:
        res = client.get("/api/control/state")
        assert res.status_code == 200
        assert res.json()["state"] == "stopped"


def test_control_stop_when_not_running_is_idempotent(app_with_settings) -> None:
    """STOP answers instantly ("stopping") and tears down in the background;
    on a not-running system the teardown is a no-op that settles to
    "stopped" within moments. Repeated STOPs never error."""
    import time as _time

    with TestClient(app_with_settings) as client:
        res = client.post("/api/control/stop")
        assert res.status_code == 200
        assert res.json()["state"] in ("stopping", "stopped")
        deadline = _time.time() + 5.0
        while _time.time() < deadline:
            state = client.get("/api/control/state").json()["state"]
            if state == "stopped":
                break
            _time.sleep(0.05)
        assert state == "stopped"
        # Idempotent: a second STOP on a stopped system is still fine.
        assert client.post("/api/control/stop").status_code == 200


# ---- S16: /api/link-lines ----


def test_link_lines_endpoint_handles_missing_file(tmp_path: Path) -> None:
    """Absent link_lines.yaml -> empty rules list (no error)."""
    cfg = Settings(
        backbone_config_path=tmp_path / "no-backbone.yaml",
        link_lines_path=tmp_path / "absent.yaml",
        udp_port=0, port=0,
    )
    app = create_app(cfg)
    with TestClient(app) as client:
        res = client.get("/api/link-lines")
        assert res.status_code == 200
        assert res.json() == {"rules": []}


def test_link_lines_endpoint_returns_loaded_rules(tmp_path: Path) -> None:
    cfg_path = tmp_path / "link_lines.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "rules": [
            {"from": "person", "to": ["palette", "forklift"], "max_distance_m": 5.0},
            {"from": "forklift", "to": ["*"]},
        ],
    }))
    cfg = Settings(
        backbone_config_path=tmp_path / "no-backbone.yaml",
        link_lines_path=cfg_path,
        udp_port=0, port=0,
    )
    app = create_app(cfg)
    with TestClient(app) as client:
        res = client.get("/api/link-lines")
        assert res.status_code == 200
        data = res.json()
        assert len(data["rules"]) == 2
        assert data["rules"][0]["from"] == "person"
        assert data["rules"][0]["max_distance_m"] == 5.0
        assert data["rules"][1]["to"] == ["*"]


def test_link_lines_endpoint_500_on_malformed_yaml(tmp_path: Path) -> None:
    cfg_path = tmp_path / "link_lines.yaml"
    cfg_path.write_text("rules: not-a-list\n")
    cfg = Settings(
        backbone_config_path=tmp_path / "no-backbone.yaml",
        link_lines_path=cfg_path,
        udp_port=0, port=0,
    )
    app = create_app(cfg)
    with TestClient(app) as client:
        res = client.get("/api/link-lines")
        assert res.status_code == 500
        assert res.json()["rules"] == []


# ---- /api/zones/state — local-bus zone contents for the comms-panel cards ----


def test_zones_state_empty_when_no_bus_traffic(app_with_settings) -> None:
    from fastapi.testclient import TestClient

    with TestClient(app_with_settings) as client:
        res = client.get("/api/zones/state")
    assert res.status_code == 200
    data = res.json()
    assert data["fresh"] is False
    assert data["states"] == {}


def test_zones_state_serves_bus_zone_state(app_with_settings) -> None:
    import time as _time

    from backbone.comms.schemas import ZoneObject, ZoneStateMessage
    from fastapi.testclient import TestClient

    with TestClient(app_with_settings) as client:
        bus = client.app.state.bus
        msg = ZoneStateMessage(
            ts=_time.time(), zone="dock",
            objects=(ZoneObject(track_id=7, cls="palette", confidence=0.9,
                                xy_m=(1.0, 1.0), occupancy_state="full"),),
            count=1,
        )
        # Inject directly (same path _handle_payload uses under its lock).
        with bus._lock:
            bus._state.zone_state_by_zone["dock"] = msg
            bus._state.last_envelope_ts = _time.time()
        res = client.get("/api/zones/state")

    data = res.json()
    assert data["fresh"] is True
    dock = data["states"]["dock"]
    assert dock["count"] == 1
    assert dock["objects"][0]["cls"] == "palette"
    assert dock["objects"][0]["occupancy_state"] == "full"
    # No PalletStateManager verdict on the message → explicit null (the JS
    # falls back to its objects heuristic).
    assert dock["decision"] is None


def test_zones_state_forwards_decision(app_with_settings) -> None:
    """The PalletStateManager verdict riding ZoneStateMessage.decision must be
    forwarded verbatim so the comms-panel cards render the enum, not the
    objects heuristic."""
    import time as _time

    from backbone.comms.schemas import ZoneDecisionModel, ZoneStateMessage
    from fastapi.testclient import TestClient

    with TestClient(app_with_settings) as client:
        bus = client.app.state.bus
        msg = ZoneStateMessage(
            ts=_time.time(), zone="dock", objects=(), count=0,
            decision=ZoneDecisionModel(
                palette_state="palette_loaded", content=("carton",),
                counts={"palette": 1, "carton": 1},
            ),
        )
        with bus._lock:
            bus._state.zone_state_by_zone["dock"] = msg
            bus._state.last_envelope_ts = _time.time()
        res = client.get("/api/zones/state")

    dock = res.json()["states"]["dock"]
    assert dock["decision"]["palette_state"] == "palette_loaded"
    assert dock["decision"]["content"] == ["carton"]
    assert dock["decision"]["counts"] == {"palette": 1, "carton": 1}
