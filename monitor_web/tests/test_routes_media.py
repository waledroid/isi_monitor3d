"""Hidden MP4 dev viewer routes — unlock, media listing, path safety."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from monitor_web.app import create_app
from monitor_web.config import Settings


@pytest.fixture
def app_with_media(tmp_path: Path):
    media = tmp_path / "media"
    media.mkdir()
    (media / "clip_a.mp4").write_bytes(b"\x00\x00")      # contents irrelevant for these tests
    (media / "clip_b.mp4").write_bytes(b"\x00\x00")
    (media / "notes.txt").write_text("ignore me")
    backbone_yaml = tmp_path / "backbone.yaml"
    backbone_yaml.write_text(yaml.safe_dump({"cameras": {}, "metadata": {"sinks": []}}))
    cfg = Settings(
        backbone_config_path=backbone_yaml,
        media_dir=media,
        mp4_unlock_password="s3cret",
        udp_port=0, port=0,
    )
    return create_app(cfg), media


def test_unlock_rejects_wrong_password(app_with_media) -> None:
    app, _ = app_with_media
    with TestClient(app) as client:
        res = client.post("/api/unlock", json={"password": "nope"})
        assert res.status_code == 200
        assert res.json() == {"ok": False}


def test_unlock_accepts_right_password(app_with_media) -> None:
    app, _ = app_with_media
    with TestClient(app) as client:
        res = client.post("/api/unlock", json={"password": "s3cret"})
        assert res.json() == {"ok": True}


def test_unlock_rejects_empty_password(app_with_media) -> None:
    app, _ = app_with_media
    with TestClient(app) as client:
        assert client.post("/api/unlock", json={"password": ""}).json() == {"ok": False}


def test_media_list_returns_only_mp4(app_with_media) -> None:
    app, _ = app_with_media
    with TestClient(app) as client:
        res = client.get("/api/media/mp4")
        assert res.status_code == 200
        assert res.json()["files"] == ["clip_a.mp4", "clip_b.mp4"]   # sorted, .txt excluded


def test_media_list_empty_when_dir_missing(tmp_path: Path) -> None:
    backbone_yaml = tmp_path / "backbone.yaml"
    backbone_yaml.write_text(yaml.safe_dump({"cameras": {}}))
    cfg = Settings(backbone_config_path=backbone_yaml,
                   media_dir=tmp_path / "absent", udp_port=0, port=0)
    app = create_app(cfg)
    with TestClient(app) as client:
        assert client.get("/api/media/mp4").json() == {"files": []}


def test_stream_rejects_non_mp4(app_with_media) -> None:
    app, _ = app_with_media
    with TestClient(app) as client:
        res = client.get("/stream/mp4/notes.txt")
        assert res.status_code == 400


def test_stream_rejects_path_traversal(app_with_media) -> None:
    app, _ = app_with_media
    with TestClient(app) as client:
        # encoded ../ should not escape media_dir
        res = client.get("/stream/mp4/..%2f..%2fetc%2fpasswd")
        assert res.status_code in (400, 404)


def test_stream_404_for_missing_mp4(app_with_media) -> None:
    app, _ = app_with_media
    with TestClient(app) as client:
        res = client.get("/stream/mp4/missing.mp4")
        assert res.status_code == 404
