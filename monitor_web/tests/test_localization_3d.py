"""3D localization — Settings-owned triangulation subscriptions (subscriptions.yaml).

``GET /api/config`` exposes the currently subscribed classes; ``POST
/api/config`` with ``localization_3d`` regenerates the file wholesale
(UI-owned; person is a dead triangulation path and never offered/written).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from monitor_web.app import create_app
from monitor_web.config import Settings

# ---- fixtures ----


def _subscription_rules() -> list[dict]:
    """An on-disk file with a dead person rule + two live object rules."""
    return [
        {
            "name": "person_3d",
            "module": "securite",
            "match": {"cls": "person", "cameras_seeing_min": 2},
            "request": "xyz",
            "rate_hz": 10.0,
        },
        {
            "name": "palette_3d",
            "module": "palettes",
            "match": {"cls": "palette", "cameras_seeing_min": 2},
            "request": "xyz",
            "rate_hz": 5.0,
        },
        {
            "name": "carton_3d",
            "module": "palettes",
            "match": {"cls": "carton", "cameras_seeing_min": 2},
            "request": "xyz",
            "rate_hz": 5.0,
        },
    ]


@pytest.fixture
def subs_app(tmp_path: Path):
    """An app whose backbone.yaml points at an existing subscriptions.yaml."""
    subs_path = tmp_path / "subscriptions.yaml"
    subs_path.write_text(yaml.safe_dump(_subscription_rules(), sort_keys=False))
    backbone_yaml = tmp_path / "backbone.yaml"
    backbone_yaml.write_text(yaml.safe_dump({
        "cameras": {
            "cam_a": {"source": {"name": "rtsp", "url": "rtsp://a.example/Streaming"}},
            "cam_b": {"source": {"name": "rtsp", "url": "rtsp://b.example/Streaming"}},
        },
        "subscriptions_path": str(subs_path),
    }))
    cfg = Settings(backbone_config_path=backbone_yaml, udp_port=0, port=0)
    return create_app(cfg), backbone_yaml, subs_path


@pytest.fixture
def no_subs_path_app(tmp_path: Path):
    """An app whose backbone.yaml has NO subscriptions_path — the self-heal path."""
    backbone_yaml = tmp_path / "backbone.yaml"
    backbone_yaml.write_text(yaml.safe_dump({
        "cameras": {
            "cam_a": {"source": {"name": "rtsp", "url": "rtsp://a.example/Streaming"}},
        },
    }))
    cfg = Settings(backbone_config_path=backbone_yaml, udp_port=0, port=0)
    return create_app(cfg), backbone_yaml


# ---- GET /api/config ----


def test_get_config_exposes_localization_3d_person_hidden(subs_app) -> None:
    """The selection mirrors the on-disk xyz rules; the dead person rule is hidden."""
    app, _, _ = subs_app
    with TestClient(app) as client:
        res = client.get("/api/config")
        assert res.status_code == 200
        assert res.json()["localization_3d"] == ["palette", "carton"]


def test_get_config_missing_subscriptions_file_is_empty(subs_app) -> None:
    app, _, subs_path = subs_app
    subs_path.unlink()
    with TestClient(app) as client:
        res = client.get("/api/config")
        assert res.status_code == 200
        assert res.json()["localization_3d"] == []


# ---- POST /api/config ----


def test_post_writes_exact_rule_doc(subs_app) -> None:
    """Two checked classes → exactly two deterministic rules, file order preserved."""
    app, _, subs_path = subs_app
    with TestClient(app) as client:
        res = client.post("/api/config", json={
            "cameras": {"cam_a": {"url": "rtsp://a.example/Streaming"}},
            "localization_3d": ["palette", "carton"],
        })
        assert res.status_code == 200, res.text
    assert yaml.safe_load(subs_path.read_text()) == [
        {
            "name": "palette_3d",
            "module": "palettes",
            "match": {"cls": "palette", "cameras_seeing_min": 2},
            "request": "xyz",
            "rate_hz": 5.0,
        },
        {
            "name": "carton_3d",
            "module": "palettes",
            "match": {"cls": "carton", "cameras_seeing_min": 2},
            "request": "xyz",
            "rate_hz": 5.0,
        },
    ]


def test_post_without_key_leaves_file_untouched(subs_app) -> None:
    """An omitted ``localization_3d`` must never rewrite subscriptions.yaml."""
    app, _, subs_path = subs_app
    before = subs_path.read_bytes()
    with TestClient(app) as client:
        res = client.post("/api/config", json={
            "cameras": {"cam_a": {"url": "rtsp://a.example/Streaming"}},
        })
        assert res.status_code == 200, res.text
    assert subs_path.read_bytes() == before


def test_post_empty_list_clears_file(subs_app) -> None:
    app, _, subs_path = subs_app
    with TestClient(app) as client:
        res = client.post("/api/config", json={
            "cameras": {"cam_a": {"url": "rtsp://a.example/Streaming"}},
            "localization_3d": [],
        })
        assert res.status_code == 200, res.text
    assert yaml.safe_load(subs_path.read_text()) == []


def test_post_drops_person(subs_app) -> None:
    """person never reaches triangulation — the writer silently drops it."""
    app, _, subs_path = subs_app
    with TestClient(app) as client:
        res = client.post("/api/config", json={
            "cameras": {"cam_a": {"url": "rtsp://a.example/Streaming"}},
            "localization_3d": ["person", "palette"],
        })
        assert res.status_code == 200, res.text
    rules = yaml.safe_load(subs_path.read_text())
    assert [r["match"]["cls"] for r in rules] == ["palette"]


def test_post_default_path_self_heal(no_subs_path_app) -> None:
    """No subscriptions_path in backbone.yaml → default beside it AND record it."""
    app, backbone_yaml = no_subs_path_app
    with TestClient(app) as client:
        res = client.post("/api/config", json={
            "cameras": {"cam_a": {"url": "rtsp://a.example/Streaming"}},
            "localization_3d": ["palette"],
        })
        assert res.status_code == 200, res.text
    expected = backbone_yaml.parent / "mode2" / "subscriptions.yaml"
    bb = yaml.safe_load(backbone_yaml.read_text())
    assert bb["subscriptions_path"] == str(expected)
    rules = yaml.safe_load(expected.read_text())
    assert [r["match"]["cls"] for r in rules] == ["palette"]


def test_written_file_loads_via_subscription_manager(subs_app) -> None:
    """Guard against DSL drift: the Backbone's own loader must accept the file."""
    from backbone.triangulation.subscription_manager import SubscriptionManager

    app, _, subs_path = subs_app
    with TestClient(app) as client:
        res = client.post("/api/config", json={
            "cameras": {"cam_a": {"url": "rtsp://a.example/Streaming"}},
            "localization_3d": ["palette", "carton"],
        })
        assert res.status_code == 200, res.text
    mgr = SubscriptionManager.load(subs_path)
    assert [r.match.cls for r in mgr.rules] == ["palette", "carton"]
    assert all(r.request == "xyz" for r in mgr.rules)
    assert all(r.match.cameras_seeing_min == 2 for r in mgr.rules)


def test_post_rejects_invalid_class_name(subs_app) -> None:
    """Class names are file-name material (``<cls>_3d``) — path junk → 422."""
    app, _, subs_path = subs_app
    before = subs_path.read_bytes()
    with TestClient(app) as client:
        res = client.post("/api/config", json={
            "cameras": {"cam_a": {"url": "rtsp://a.example/Streaming"}},
            "localization_3d": ["../evil"],
        })
        assert res.status_code == 422
    assert subs_path.read_bytes() == before
