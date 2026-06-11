"""``GET /api/cameras/available`` + V4L2 device discovery helper."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from monitor_web.api.routes_cameras import discover_v4l2_devices
from monitor_web.app import create_app
from monitor_web.config import Settings


def _make_sysfs(tmp_path: Path, entries: dict[str, str | None]) -> Path:
    """Build a fake /sys/class/video4linux tree. ``entries`` maps videoN -> name
    (or None to omit the name file)."""
    root = tmp_path / "video4linux"
    root.mkdir()
    for dev, name in entries.items():
        d = root / dev
        d.mkdir()
        if name is not None:
            (d / "name").write_text(name + "\n")
    return root


def test_discover_returns_empty_when_sysfs_absent(tmp_path: Path) -> None:
    assert discover_v4l2_devices(tmp_path / "nope") == []


def test_discover_lists_devices_sorted_by_index(tmp_path: Path) -> None:
    root = _make_sysfs(tmp_path, {"video2": "Cam Two", "video0": "Cam Zero"})
    devices = discover_v4l2_devices(root)
    assert devices == [
        {"path": "/dev/video0", "name": "Cam Zero"},
        {"path": "/dev/video2", "name": "Cam Two"},
    ]


def test_discover_falls_back_to_node_name_when_name_file_missing(tmp_path: Path) -> None:
    root = _make_sysfs(tmp_path, {"video0": None})
    devices = discover_v4l2_devices(root)
    assert devices == [{"path": "/dev/video0", "name": "video0"}]


def test_discover_ignores_non_video_entries(tmp_path: Path) -> None:
    root = _make_sysfs(tmp_path, {"video0": "Cam", "v4l-subdev0": "subdev"})
    devices = discover_v4l2_devices(root)
    assert devices == [{"path": "/dev/video0", "name": "Cam"}]


def _app(tmp_path: Path):
    backbone_yaml = tmp_path / "backbone.yaml"
    backbone_yaml.write_text(yaml.safe_dump({"cameras": {}, "metadata": {"sinks": []}}))
    return create_app(Settings(backbone_config_path=backbone_yaml, udp_port=0, port=0))


def test_endpoint_returns_shape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Force discovery to a fake tree with one device.
    root = _make_sysfs(tmp_path, {"video0": "USB Webcam"})
    import monitor_web.api.routes_cameras as mod
    monkeypatch.setattr(mod, "_SYSFS_V4L2_ROOT", root)

    app = _app(tmp_path)
    with TestClient(app) as client:
        res = client.get("/api/cameras/available")
        assert res.status_code == 200
        data = res.json()
        assert data["devices"] == [{"path": "/dev/video0", "name": "USB Webcam"}]
        assert "rtsp" in data["plugins"]
        assert "v4l2" in data["plugins"]


def test_endpoint_empty_when_no_devices(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import monitor_web.api.routes_cameras as mod
    monkeypatch.setattr(mod, "_SYSFS_V4L2_ROOT", tmp_path / "absent")
    app = _app(tmp_path)
    with TestClient(app) as client:
        res = client.get("/api/cameras/available")
        assert res.status_code == 200
        assert res.json()["devices"] == []
