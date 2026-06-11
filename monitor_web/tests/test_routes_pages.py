"""Dashboard page renders with expected MUI markers + segmented toggle + status."""

from __future__ import annotations

from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from monitor_web.app import create_app
from monitor_web.config import Settings


def _minimal_app(tmp_path: Path):
    backbone_yaml = tmp_path / "backbone.yaml"
    backbone_yaml.write_text(yaml.safe_dump({
        "cameras": {},
        "metadata": {"sinks": []},
    }))
    return create_app(Settings(backbone_config_path=backbone_yaml, udp_port=0, port=0))


def test_dashboard_page_renders(tmp_path) -> None:
    app = _minimal_app(tmp_path)
    with TestClient(app) as client:
        res = client.get("/")
        assert res.status_code == 200
        html = res.text
        # Segmented toggle + START/STOP buttons (custom tab implementation).
        assert "custom-tabs" in html
        assert "btn-start" in html
        assert "btn-stop" in html
        # The 3-way toggle is wired.
        assert 'data-view="map"' in html
        assert 'data-view="cam_a"' in html
        assert 'data-view="cam_b"' in html
        # Status indicator + lang toggle.
        assert "status-dot" in html
        assert 'data-lang="en"' in html
        assert 'data-lang="fr"' in html
        # Zone-empty placeholders (S9 v1 — before zone manager).
        assert "zone-empty" in html


def test_logs_partial_renders(tmp_path) -> None:
    app = _minimal_app(tmp_path)
    with TestClient(app) as client:
        res = client.get("/api/logs")
        assert res.status_code == 200
        assert "log-line" in res.text or "logs-empty" in res.text
