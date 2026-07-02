"""Targetless capture ROUTES — self-contained start/stop/capture-pair/shots.

Hermetic: a stub CaptureManager is injected so no camera is opened. Proves the
targetless routes drive TargetlessSession (not the board session) and expose the
targetless shot gallery, and that board capture routes/behaviour are unchanged.
"""

from __future__ import annotations

import cv2
import numpy as np
from fastapi.testclient import TestClient

from isical.app import create_app
from isical.config import Settings


def _client():
    return TestClient(create_app(Settings()))


def _mk_rig(c, name="rig"):
    c.post("/api/projects", json={
        "name": name,
        "cam_a": {"type": "rtsp", "url": "rtsp://x/a"},
        "cam_b": {"type": "rtsp", "url": "rtsp://x/b"}})


class _FakeTargetless:
    def __init__(self):
        self.pair_count = 0

    def status(self):
        return {"phase": "targetless", "pair_count": self.pair_count,
                "cameras": {"cam_a": {"count": self.pair_count, "status": "live",
                                      "texture": {"features": 300, "blur_var": 90.0, "ok": True}},
                            "cam_b": {"count": self.pair_count, "status": "live",
                                      "texture": {"features": 250, "blur_var": 80.0, "ok": True}}}}

    def capture_pair(self):
        self.pair_count += 1
        return {"captured": True, "pair_count": self.pair_count, "files": {}}


class _FakeManager:
    def __init__(self):
        self._tl = None

    def start_targetless(self, project, d, cfg, **kw):
        self._tl = _FakeTargetless()
        return self._tl.status()

    def stop_targetless(self):
        self._tl = None

    def targetless(self, project):
        return self._tl

    # board-side no-ops used by other routes if hit
    def active(self, project, phase=None):
        return None


def test_targetless_capture_lifecycle_and_pair():
    with _client() as c:
        _mk_rig(c)
        c.app.state.capture = _FakeManager()
        r = c.post("/api/p/rig/targetless/capture/start")
        assert r.status_code == 200 and r.json()["ok"] is True
        st = c.get("/api/p/rig/targetless/capture/status").json()
        assert st["active"] is True and st["phase"] == "targetless"
        assert "texture" in st["cameras"]["cam_a"]
        p = c.post("/api/p/rig/targetless/capture-pair").json()
        assert p["captured"] is True and p["pair_count"] == 1
        assert c.post("/api/p/rig/targetless/capture/stop").json()["ok"] is True


def test_targetless_status_inactive_when_not_started():
    with _client() as c:
        _mk_rig(c)
        c.app.state.capture = _FakeManager()
        assert c.get("/api/p/rig/targetless/capture/status").json() == {"active": False}


def test_targetless_capture_pair_without_session_409():
    with _client() as c:
        _mk_rig(c)
        c.app.state.capture = _FakeManager()
        assert c.post("/api/p/rig/targetless/capture-pair").status_code == 409


def test_targetless_shots_lists_captured_pairs(tmp_path):
    with _client() as c:
        _mk_rig(c)
        d = c.app.state.settings.data_dir / "rig"
        for cid in ("cam_a", "cam_b"):
            td = d / "targetless" / cid
            td.mkdir(parents=True, exist_ok=True)
            for i in range(2):
                cv2.imwrite(str(td / f"{cid}_{i:03d}.jpg"),
                            np.zeros((16, 16, 3), np.uint8))
        r = c.get("/api/p/rig/targetless-shots").json()
        assert r["pair_count"] == 2
        assert len(r["cameras"]["cam_a"]["files"]) == 2
        # the image server serves them
        f = r["cameras"]["cam_a"]["files"][0]
        assert c.get(f"/targetless-shot/rig/cam_a/{f}").status_code == 200


def test_targetless_shot_image_path_guarded():
    with _client() as c:
        _mk_rig(c)
        assert c.get("/targetless-shot/rig/cam_a/..%2f..%2fcalib.yaml").status_code == 404


def test_board_shots_route_unchanged():
    """The board shots endpoint still only accepts intrinsic|extrinsic."""
    with _client() as c:
        _mk_rig(c)
        assert c.get("/api/p/rig/shots/targetless/cam_a").status_code == 404
