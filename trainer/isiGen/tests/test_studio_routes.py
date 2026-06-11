"""Studio smoke via TestClient — no GPU, jobs run real phase-1/3 code on tiny images."""

import time

from fastapi.testclient import TestClient
from src.studio.app import create_app
from src.studio.config import Settings


def _client():
    return TestClient(create_app(Settings()))


def test_pages_and_project_crud(tmp_path):
    with _client() as c:
        assert c.get("/").status_code == 200
        assert c.get("/api/projects").json() == {"projects": []}
        body = {"name": "demo", "classes": [
            {"name": "palette", "trigger": "ISI_PLT", "color": [220, 40, 40]}]}
        assert c.post("/api/projects", json=body).json()["ok"] is True
        assert c.post("/api/projects", json=body).status_code == 409      # dup
        names = [p["name"] for p in c.get("/api/projects").json()["projects"]]
        assert names == ["demo"]
        assert c.get("/p/demo").status_code == 200
        assert c.get("/p/demo/curate").status_code == 200
        assert c.get("/p/nope").status_code == 404


def test_records_patch_prompts_caption(tiny_project):
    _pdir, _ = tiny_project
    with _client() as c:
        recs = c.get("/api/p/tiny/records").json()["records"]
        assert len(recs) == 3
        rid = recs[0]["id"]
        # PATCH class + exclude
        r = c.patch(f"/api/p/tiny/records/{rid}", json={"excluded": True}).json()
        assert r["record"]["excluded"] is True
        assert c.patch(f"/api/p/tiny/records/{rid}",
                       json={"class_name": "carton"}).json()["record"]["class_name"] == "carton"
        # prompts roundtrip
        prompts = [{"kind": "point", "class_name": "carton", "xy": [10, 20], "label": 1}]
        assert c.put(f"/api/p/tiny/records/{rid}/prompts",
                     json={"prompts": prompts}).json()["ok"] is True
        back = c.get("/api/p/tiny/records").json()["records"]
        rec = next(x for x in back if x["id"] == rid)
        assert rec["mask_prompts"][0]["xy"] == [10.0, 20.0]
        # caption PUT sets edited
        assert c.put(f"/api/p/tiny/records/{rid}/caption",
                     json={"caption": "hand written"}).json()["ok"] is True
        got = c.get(f"/api/p/tiny/records/{rid}/caption").json()
        assert got == {"caption": "hand written", "edited": True}
        # media: thumb works, unknown id 404s
        assert c.get(f"/media/tiny/thumb/{rid}").status_code == 200
        assert c.get("/media/tiny/thumb/ffffffffffff").status_code == 404


def test_phase_job_runs_captions(tiny_project):
    with _client() as c:
        r = c.post("/api/p/tiny/run/captions")
        assert r.status_code == 200
        job_id = r.json()["job"]["id"]
        deadline = time.time() + 10
        state = "queued"
        while time.time() < deadline:
            state = c.get(f"/api/jobs/{job_id}/log").json()["job"]["state"]
            if state in ("done", "failed"):
                break
            time.sleep(0.1)
        assert state == "done"
        st = c.get("/api/p/tiny/status").json()
        assert st["captioned"] == 3
        assert c.post("/api/p/tiny/run/not_a_phase").status_code == 404
