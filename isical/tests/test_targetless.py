"""isical targetless-extrinsics wiring (hermetic — no camera / ONNX / Multical).

Covers the Stage-3 backend surface the frontend drives: the method selector routes
correctly, scale-references persist + validate, the extrinsic job dispatches to the
targetless runner when selected, the report/stage-image endpoints behave, and the
runner surfaces a clear ONNX-weights-missing error on the (hermetic) no-weights path.
"""

from __future__ import annotations

import json

import pytest
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


# --- method selector ---------------------------------------------------------


def test_extrinsic_method_defaults_aprilgrid_and_persists():
    from isical.core.project import load_project
    with _client() as c:
        _mk_rig(c)
        assert c.get("/api/p/rig/extrinsic-method").json()["method"] == "aprilgrid"
        r = c.put("/api/p/rig/extrinsic-method", json={"method": "targetless"})
        assert r.json() == {"ok": True, "method": "targetless"}
        assert c.get("/api/p/rig/extrinsic-method").json()["method"] == "targetless"
        # persisted to calib.yaml
        data_dir = c.app.state.settings.data_dir
        assert load_project(data_dir / "rig").extrinsic_method == "targetless"


def test_extrinsic_method_rejects_unknown():
    with _client() as c:
        _mk_rig(c)
        assert c.put("/api/p/rig/extrinsic-method", json={"method": "nope"}).status_code == 422


# --- scale references --------------------------------------------------------


def _refs(n=3):
    return [{"p1_a": [10.0, 20.0], "p1_b": [12.0, 20.0],
             "p2_a": [50.0, 60.0], "p2_b": [52.0, 60.0],
             "distance_m": 1.0 + i} for i in range(n)]


def test_scale_references_roundtrip():
    with _client() as c:
        _mk_rig(c)
        assert c.get("/api/p/rig/scale-references").json() == {"references": [], "count": 0}
        r = c.put("/api/p/rig/scale-references", json={"references": _refs(3)})
        assert r.json()["ok"] is True and r.json()["enough"] is True
        got = c.get("/api/p/rig/scale-references").json()
        assert got["count"] == 3
        # persisted on disk where the runner reads it
        p = c.app.state.settings.data_dir / "rig" / "work" / "scale_references.json"
        assert len(json.loads(p.read_text())) == 3


def test_scale_references_under_three_flags_not_enough():
    with _client() as c:
        _mk_rig(c)
        r = c.put("/api/p/rig/scale-references", json={"references": _refs(2)})
        assert r.json()["enough"] is False


def test_scale_reference_rejects_nonpositive_distance():
    with _client() as c:
        _mk_rig(c)
        bad = _refs(1)
        bad[0]["distance_m"] = 0.0
        assert c.put("/api/p/rig/scale-references", json={"references": bad}).status_code == 422


# --- stage images + report ---------------------------------------------------


def test_targetless_stage_and_report_empty_before_solve():
    with _client() as c:
        _mk_rig(c)
        assert c.get("/api/p/rig/targetless-stages").json() == {"stages": []}
        assert c.get("/api/p/rig/targetless-report").json() == {"report": None}
        assert c.get("/targetless-stage/rig/pair").status_code == 404


def test_stage_image_path_traversal_guarded():
    with _client() as c:
        _mk_rig(c)
        assert c.get("/targetless-stage/rig/..%2f..%2fcalib").status_code == 404


# --- extrinsic job dispatch --------------------------------------------------


def test_extrinsic_job_dispatches_to_targetless_runner(monkeypatch):
    """When method=targetless, POST /run/extrinsic must call the targetless runner."""
    called = {}

    def _fake_targetless(d):
        called["targetless"] = str(d)
        return {"ok": True, "method": "targetless"}

    monkeypatch.setattr("isical.api.routes_jobs.run_extrinsic_targetless", _fake_targetless)
    with _client() as c:
        _mk_rig(c)
        c.put("/api/p/rig/extrinsic-method", json={"method": "targetless"})
        r = c.post("/api/p/rig/run/extrinsic")
        assert r.status_code == 200
        # let the single-worker JobRunner drain
        import time
        for _ in range(50):
            jobs = c.get("/api/jobs").json()["jobs"]
            if jobs and jobs[0]["state"] in ("done", "failed"):
                break
            time.sleep(0.02)
        assert called.get("targetless") is not None


def test_targetless_runner_missing_scale_refs_errors(tiny_project):
    """No scale references marked → a clear ValueError (before touching ONNX)."""
    from isical.core.runners import run_extrinsic_targetless
    pdir, _cfg = tiny_project
    # intrinsic.json + pairs so the failure is specifically the missing refs
    (pdir / "work").mkdir(parents=True, exist_ok=True)
    (pdir / "work" / "intrinsic.json").write_text(json.dumps({"cameras": {
        "cam_a": {"K": [[1000, 0, 640], [0, 1000, 480], [0, 0, 1]], "dist": [0, 0, 0, 0, 0],
                  "image_size": [1280, 960]},
        "cam_b": {"K": [[1000, 0, 640], [0, 1000, 480], [0, 0, 1]], "dist": [0, 0, 0, 0, 0],
                  "image_size": [1280, 960]}}}))
    import cv2
    import numpy as np
    for cid in ("cam_a", "cam_b"):
        d = pdir / "extrinsic" / cid
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / "000.jpg"), np.zeros((960, 1280, 3), np.uint8))
    with pytest.raises(ValueError, match="scale references"):
        run_extrinsic_targetless(pdir)
