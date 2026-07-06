"""Tests for GET /api/gateway/nodes.

All four cases from the spec:
  (a) gateway_url unset → {configured: false, nodes: []}
  (b) gateway_url set, urlopen monkeypatched to return JSON → configured:true + nodes
  (c) urlopen raises URLError → configured:true + error key, status 200 (no 500)
  (d) token set → request carried Authorization: Bearer header
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from monitor_web.app import create_app
from monitor_web.config import Settings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NODES_PAYLOAD = [
    {
        "node_id": "node-warehouse-a",
        "area": "Zone A",
        "status": "alive",
        "last_seen": 1_700_000_000.0,
        "mode": "dual_cam_homography_triangulation",
        "cameras": ["cam_a", "cam_b"],
        "latency_ms": 87.3,
        "fps": 14.2,
    },
    {
        "node_id": "node-warehouse-b",
        "area": "Zone B",
        "status": "stale",
        "last_seen": 1_699_999_900.0,
        "mode": "single_cam_homography",
        "cameras": ["cam_a"],
        "latency_ms": None,
        "fps": None,
    },
]

_GATEWAY_RESPONSE = {"nodes": _NODES_PAYLOAD, "count": 2}


def _make_app(tmp_path: Path, **settings_kwargs) -> object:
    """Create a minimal test app with an ephemeral backbone.yaml."""
    backbone_yaml = tmp_path / "backbone.yaml"
    backbone_yaml.write_text(yaml.safe_dump({
        "cameras": {"cam_a": {"source": {"name": "replay", "frames": []}}},
        "detection": {"plugin": "yolo_onnx", "onnx_path": "x.onnx", "class_names": ["person"]},
        "metadata": {"sinks": [{"plugin": "udp", "host": "127.0.0.1", "port": 0}]},
    }))
    cfg = Settings(
        backbone_config_path=backbone_yaml,
        udp_port=0,
        port=0,
        **settings_kwargs,
    )
    return create_app(cfg)


# ---------------------------------------------------------------------------
# Fake HTTP response helper
# ---------------------------------------------------------------------------

class _FakeResponse:
    """Minimal file-like object returned by the monkeypatched urlopen."""

    def __init__(self, payload: object) -> None:
        self._data = json.dumps(payload).encode()

    def read(self) -> bytes:
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


# ---------------------------------------------------------------------------
# (a) gateway_url unset → configured: false
# ---------------------------------------------------------------------------

def test_gateway_nodes_not_configured(tmp_path: Path) -> None:
    """When gateway_url is not set the endpoint returns {configured:false, nodes:[]}."""
    app = _make_app(tmp_path)  # no gateway_url → defaults to None
    with TestClient(app) as client:
        res = client.get("/api/gateway/nodes")
    assert res.status_code == 200
    data = res.json()
    assert data["configured"] is False
    assert data["nodes"] == []


# ---------------------------------------------------------------------------
# (b) gateway_url set, urlopen returns nodes
# ---------------------------------------------------------------------------

def test_gateway_nodes_returns_node_list(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With gateway_url set and a successful response the nodes list is forwarded."""
    captured: list = []

    def _fake_urlopen(req, timeout=None):
        captured.append(req)
        return _FakeResponse(_GATEWAY_RESPONSE)

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

    app = _make_app(tmp_path, gateway_url="http://gateway-host:8080")
    with TestClient(app) as client:
        res = client.get("/api/gateway/nodes")

    assert res.status_code == 200
    data = res.json()
    assert data["configured"] is True
    assert "error" not in data
    assert data["gateway_url"] == "http://gateway-host:8080"
    assert len(data["nodes"]) == 2
    assert data["nodes"][0]["node_id"] == "node-warehouse-a"
    assert data["nodes"][1]["status"] == "stale"


# ---------------------------------------------------------------------------
# (c) urlopen raises → configured:true + error key, still HTTP 200
# ---------------------------------------------------------------------------

def test_gateway_nodes_error_never_500(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A network error surfaces as {configured:true, error:..., nodes:[]} (not HTTP 500)."""
    def _failing_urlopen(req, timeout=None):
        raise urllib.error.URLError("Connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", _failing_urlopen)

    app = _make_app(tmp_path, gateway_url="http://dead-gateway:8080")
    with TestClient(app) as client:
        res = client.get("/api/gateway/nodes")

    assert res.status_code == 200
    data = res.json()
    assert data["configured"] is True
    assert "error" in data
    assert data["nodes"] == []
    # Verify the error message is a non-empty string, not a crash traceback.
    assert isinstance(data["error"], str)
    assert len(data["error"]) > 0


# ---------------------------------------------------------------------------
# (d) token set → Authorization: Bearer header sent
# ---------------------------------------------------------------------------

def test_gateway_nodes_sends_bearer_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When gateway_token is configured the outbound request carries the Bearer header."""
    captured_requests: list = []

    def _recording_urlopen(req, timeout=None):
        captured_requests.append(req)
        return _FakeResponse({"nodes": [], "count": 0})

    monkeypatch.setattr(urllib.request, "urlopen", _recording_urlopen)

    app = _make_app(
        tmp_path,
        gateway_url="http://gateway-host:8080",
        gateway_token="super-secret-token",
    )
    with TestClient(app) as client:
        client.get("/api/gateway/nodes")

    assert len(captured_requests) == 1
    req = captured_requests[0]
    auth_header = req.get_header("Authorization")
    assert auth_header == "Bearer super-secret-token"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_gateway_nodes_timeout_is_passed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The configured timeout_s is forwarded to urlopen."""
    timeouts_seen: list = []

    def _recording_urlopen(req, timeout=None):
        timeouts_seen.append(timeout)
        return _FakeResponse({"nodes": []})

    monkeypatch.setattr(urllib.request, "urlopen", _recording_urlopen)

    app = _make_app(tmp_path, gateway_url="http://gw:8080", gateway_timeout_s=1.5)
    with TestClient(app) as client:
        client.get("/api/gateway/nodes")

    assert timeouts_seen == [1.5]


def test_gateway_nodes_bare_list_response(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Gateway returning a bare JSON list (not wrapped in 'nodes' key) is handled."""
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda req, timeout=None: _FakeResponse(_NODES_PAYLOAD),  # bare list
    )
    app = _make_app(tmp_path, gateway_url="http://gw:8080")
    with TestClient(app) as client:
        res = client.get("/api/gateway/nodes")

    data = res.json()
    assert data["configured"] is True
    assert len(data["nodes"]) == 2


def test_gateway_nodes_http_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An HTTP 403 from the gateway surfaces as error, not 500."""
    def _http_error(req, timeout=None):
        raise urllib.error.HTTPError(
            url="http://gw:8080/nodes",
            code=403,
            msg="Forbidden",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr(urllib.request, "urlopen", _http_error)

    app = _make_app(tmp_path, gateway_url="http://gw:8080")
    with TestClient(app) as client:
        res = client.get("/api/gateway/nodes")

    assert res.status_code == 200
    data = res.json()
    assert data["configured"] is True
    assert "403" in data["error"]
    assert data["nodes"] == []


# ---------------------------------------------------------------------------
# UI-settings resolution order
# ---------------------------------------------------------------------------

def _ui_settings_path_from_env() -> Path:
    """Return the ui-settings path that the conftest autouse fixture installed."""
    import os
    return Path(os.environ["MONITOR_WEB_UI_SETTINGS_PATH"])


def test_gateway_nodes_uses_ui_settings_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When ui-settings has gateway_url, /api/gateway/nodes uses it (not the env value)."""
    import yaml as _yaml

    # Write gateway_url into the ui-settings file BEFORE building the app so
    # cfg.ui_settings_path (set by the conftest autouse fixture) already contains it.
    ui_path = _ui_settings_path_from_env()
    ui_path.write_text(_yaml.safe_dump({"gateway_url": "http://ui-gateway:8080"}))

    captured: list = []

    def _fake_urlopen(req, timeout=None):
        captured.append(req.full_url)
        return _FakeResponse(_GATEWAY_RESPONSE)

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

    # App configured with env gateway_url pointing to a different host.
    app = _make_app(tmp_path, gateway_url="http://env-gateway:9999")

    with TestClient(app) as client:
        res = client.get("/api/gateway/nodes")

    assert res.status_code == 200
    data = res.json()
    assert data["configured"] is True
    # The URL actually hit must be the ui-settings one, not the env one.
    assert len(captured) == 1
    assert "ui-gateway:8080" in captured[0]
    assert "env-gateway" not in captured[0]
    assert data["gateway_url"] == "http://ui-gateway:8080"


def test_gateway_nodes_falls_back_to_env_when_ui_settings_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When ui-settings has no gateway_url, the env/Settings value is used as fallback."""
    import yaml as _yaml

    # ui-settings exists but has no gateway_url key.
    ui_path = _ui_settings_path_from_env()
    ui_path.write_text(_yaml.safe_dump({"mp4_selected": "foo.mp4"}))

    captured: list = []

    def _fake_urlopen(req, timeout=None):
        captured.append(req.full_url)
        return _FakeResponse({"nodes": [], "count": 0})

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

    app = _make_app(tmp_path, gateway_url="http://fallback-gateway:8080")

    with TestClient(app) as client:
        res = client.get("/api/gateway/nodes")

    assert res.status_code == 200
    data = res.json()
    assert data["configured"] is True
    assert len(captured) == 1
    assert "fallback-gateway:8080" in captured[0]


def test_gateway_nodes_ui_settings_token_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When ui-settings has gateway_token, it is sent as the Bearer header."""
    import yaml as _yaml

    ui_path = _ui_settings_path_from_env()
    ui_path.write_text(_yaml.safe_dump({
        "gateway_url": "http://ui-gateway:8080",
        "gateway_token": "ui-secret",
    }))

    captured_reqs: list = []

    def _fake_urlopen(req, timeout=None):
        captured_reqs.append(req)
        return _FakeResponse({"nodes": []})

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

    app = _make_app(tmp_path)  # no env gateway_url / token

    with TestClient(app) as client:
        client.get("/api/gateway/nodes")

    assert len(captured_reqs) == 1
    assert captured_reqs[0].get_header("Authorization") == "Bearer ui-secret"


# ---------------------------------------------------------------------------
# ui-settings persistence: POST gateway_url then GET returns it
# ---------------------------------------------------------------------------

def test_ui_settings_persist_gateway_url(tmp_path: Path) -> None:
    """POST gateway_url to /api/ui-settings then GET returns it."""
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        post_res = client.post(
            "/api/ui-settings",
            json={"gateway_url": "http://stored-gateway:8080", "gateway_token": "tok"},
        )
        assert post_res.status_code == 200

        get_res = client.get("/api/ui-settings")
        assert get_res.status_code == 200
        data = get_res.json()
        assert data.get("gateway_url") == "http://stored-gateway:8080"
        assert data.get("gateway_token") == "tok"


# ---------------------------------------------------------------------------
# /api/gateway/zones — same contract, forwards the enriched zone list
# ---------------------------------------------------------------------------

_GATEWAY_ZONES_RESPONSE = {
    "zones": [
        {"node_id": "zone_a", "name": "dock", "kind": "palette",
         "objects": [{"track_id": 1, "cls": "palette", "confidence": 0.9,
                      "xy_m": [1.0, 1.0], "occupancy_state": "empty"}],
         "count": 1, "state_ts": 123.0},
        {"node_id": "zone_a", "name": "cold", "kind": "danger",
         "objects": None, "count": None, "state_ts": None},
    ],
    "count": 2,
}


def test_gateway_zones_not_configured(tmp_path: Path) -> None:
    app = _make_app(tmp_path)
    with TestClient(app) as client:
        res = client.get("/api/gateway/zones")
    assert res.status_code == 200
    assert res.json() == {"configured": False, "zones": []}


def test_gateway_zones_forwards_enriched_list(tmp_path: Path,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list = []

    def _fake_urlopen(req, timeout=None):
        captured.append(req)
        return _FakeResponse(_GATEWAY_ZONES_RESPONSE)

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    app = _make_app(tmp_path, gateway_url="http://gateway-host:8080",
                    gateway_token="sekret")
    with TestClient(app) as client:
        res = client.get("/api/gateway/zones")

    data = res.json()
    assert data["configured"] is True
    assert len(data["zones"]) == 2
    assert data["zones"][0]["objects"][0]["occupancy_state"] == "empty"
    assert data["zones"][1]["objects"] is None   # no state yet — card stays dim
    assert captured[0].full_url == "http://gateway-host:8080/zones"
    assert captured[0].get_header("Authorization") == "Bearer sekret"


def test_gateway_zones_error_never_500(tmp_path: Path,
                                       monkeypatch: pytest.MonkeyPatch) -> None:
    def _failing_urlopen(req, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", _failing_urlopen)
    app = _make_app(tmp_path, gateway_url="http://gateway-host:8080")
    with TestClient(app) as client:
        res = client.get("/api/gateway/zones")
    assert res.status_code == 200
    data = res.json()
    assert data["configured"] is True
    assert "error" in data
    assert data["zones"] == []
