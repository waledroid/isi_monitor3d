"""Studio routes via TestClient (hermetic — no camera/Multical)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from isical.app import create_app
from isical.config import Settings


def _client():
    return TestClient(create_app(Settings()))


def test_pages_and_project_crud():
    with _client() as c:
        assert c.get("/").status_code == 200
        assert c.get("/api/projects").json() == {"projects": []}
        body = {"name": "rig", "cam_a": {"type": "rtsp", "url": "rtsp://x/a"},
                "cam_b": {"type": "rtsp", "url": "rtsp://x/b"}}
        assert c.post("/api/projects", json=body).json()["ok"] is True
        assert c.post("/api/projects", json=body).status_code == 409          # dup
        names = [p["name"] for p in c.get("/api/projects").json()["projects"]]
        assert names == ["rig"]
        assert c.get("/p/rig").status_code == 200
        assert c.get("/p/rig/capture/intrinsic").status_code == 200
        assert c.get("/p/nope").status_code == 404


def test_create_requires_cam_a():
    with _client() as c:
        r = c.post("/api/projects", json={"name": "x", "cam_a": {"type": "rtsp", "url": ""}})
        assert r.status_code == 422


def test_status_and_capture_status():
    with _client() as c:
        c.post("/api/projects", json={"name": "rig", "cam_a": {"type": "rtsp", "url": "rtsp://x/a"}})
        st = c.get("/api/p/rig/status").json()
        assert st["cameras"] == ["cam_a"] and st["intrinsic_done"] is False
        cap = c.get("/api/p/rig/capture/status").json()
        assert cap["active"] is False


def test_extrinsic_capture_needs_two_cameras():
    with _client() as c:
        c.post("/api/projects", json={"name": "solo", "cam_a": {"type": "rtsp", "url": "rtsp://x/a"}})
        r = c.post("/api/p/solo/capture/extrinsic/start")
        assert r.status_code == 422                       # extrinsic needs both cams


def test_run_unknown_phase_404():
    with _client() as c:
        c.post("/api/projects", json={"name": "rig", "cam_a": {"type": "rtsp", "url": "rtsp://x/a"}})
        assert c.post("/api/p/rig/run/bogus").status_code == 404
