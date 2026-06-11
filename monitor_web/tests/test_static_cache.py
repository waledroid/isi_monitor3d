"""Static assets are served with a revalidate (no-cache) policy so edits to
CSS/JS are always picked up without a browser hard-refresh."""

from __future__ import annotations

from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from monitor_web.app import create_app
from monitor_web.config import Settings


def _app(tmp_path: Path):
    bb = tmp_path / "backbone.yaml"
    bb.write_text(yaml.safe_dump({"cameras": {}, "metadata": {"sinks": []}}))
    return create_app(Settings(backbone_config_path=bb, udp_port=0, port=0))


def test_static_assets_revalidate(tmp_path) -> None:
    with TestClient(_app(tmp_path)) as client:
        for path in ("/static/js/big_panel.js", "/static/css/dashboard.css"):
            res = client.get(path)
            assert res.status_code == 200, path
            assert res.headers.get("cache-control") == "no-cache", path


def test_non_static_paths_unaffected(tmp_path) -> None:
    """The middleware only touches /static — page/API responses keep their own
    caching (i.e. it doesn't blanket every response)."""
    with TestClient(_app(tmp_path)) as client:
        res = client.get("/api/calibrate/status")
        assert res.status_code == 200
        assert res.headers.get("cache-control") != "no-cache"
