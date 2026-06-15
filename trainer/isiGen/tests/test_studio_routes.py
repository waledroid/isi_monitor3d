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
        assert c.get("/p/demo/maps").status_code == 200      # phase 2 (control maps)
        assert c.get("/p/demo/masks").status_code == 200     # phase 3 (SAM2 masks)
        assert c.get("/p/demo/scaffolds").status_code == 200  # phase 6 gallery
        assert c.get("/p/demo/mint").status_code == 200       # phase 7 gallery
        assert c.get("/p/demo/lora").status_code == 200       # phase 5 viewer
        assert c.get("/p/nope").status_code == 404


def test_scaffold_and_lora_galleries(tiny_project, tmp_path):
    import json
    pdir, _ = tiny_project
    # a fake scaffold pair + index
    sdir = pdir / "scaffolds"
    sdir.mkdir(exist_ok=True)
    import cv2
    import numpy as np
    cv2.imwrite(str(sdir / "sc000000_control.png"), np.zeros((20, 20), np.uint8))
    cv2.imwrite(str(sdir / "sc000000_mask.png"), np.zeros((20, 20, 3), np.uint8))
    (sdir / "index.jsonl").write_text(json.dumps(
        {"id": "sc000000", "control": "scaffolds/sc000000_control.png",
         "mask": "scaffolds/sc000000_mask.png", "classes": ["palette"],
         "source": "box3d_procedural", "status": "pending"}) + "\n")
    # a fake LoRA run with a plot
    run = tmp_path / "runs" / "lora" / "tiny_r16_x"
    run.mkdir(parents=True, exist_ok=True)
    (run / "report.md").write_text("# LoRA — tiny\n- final loss: 0.10\n")
    (run / "loss_curve.png").write_bytes(b"\x89PNG\r\n")
    (run / "pytorch_lora_weights.safetensors").write_bytes(b"x")
    with _client() as c:
        sc = c.get("/api/p/tiny/scaffolds").json()["scaffolds"]
        assert len(sc) == 1 and sc[0]["id"] == "sc000000"
        assert c.get("/media/tiny/scaffold/sc000000/control").status_code == 200
        assert c.get("/media/tiny/scaffold/sc000000/mask").status_code == 200
        assert c.get("/media/tiny/scaffold/nope/control").status_code == 404
        assert c.get("/media/tiny/scaffold/sc000000/bogus").status_code == 404
        runs = c.get("/api/p/tiny/lora-runs").json()["runs"]
        assert len(runs) == 1 and runs[0]["has_plot"] and runs[0]["has_weights"]
        assert "final loss" in runs[0]["report"]
        assert c.get("/media/tiny/lora/tiny_r16_x/plot").status_code == 200
        assert c.get("/media/tiny/lora/evil_x/plot").status_code == 404   # name guard


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


def test_clearing_prompts_drops_the_mask(tiny_project):
    """Studio 'Clear' PUTs empty prompts; that must also drop the saved mask so
    the stale (auto-everything) mask doesn't linger."""
    from src.core.manifest import Manifest
    pdir, _ = tiny_project
    m = Manifest.load(pdir)
    rid = next(iter(m.records))
    rec = m.get(rid)
    rec.mask = "maps/mask/whatever.png"          # pretend a mask exists
    m.upsert(rec)
    m.save()
    with _client() as c:
        r = c.put(f"/api/p/tiny/records/{rid}/prompts", json={"prompts": []})
        assert r.json()["ok"] is True
        back = next(x for x in c.get("/api/p/tiny/records").json()["records"]
                    if x["id"] == rid)
        assert back["mask_prompts"] == []
        assert back["mask"] is None              # cleared, not lingering


def test_status_phase_counts_ignore_synthetic_records(tiny_project):
    """Phases 1-4 measure REAL curated images — minted (synthetic) records must
    not dilute depth/canny/masked/captioned or the 'real' denominator, else those
    phases can never go green after minting."""
    from src.core.manifest import Manifest, ManifestRecord
    pdir, _ = tiny_project
    m = Manifest.load(pdir)
    real = [r for r in m.records.values() if not r.excluded]
    # mark all 3 real records fully processed
    for r in real:
        r.depth_map = f"maps/depth/{r.id}.png"
        r.canny_map = f"maps/canny/{r.id}.png"
        r.mask = f"maps/mask/{r.id}.png"
        r.caption_path = f"captions/{r.id}.txt"
        r.needs_review = False
        m.upsert(r)
    # add a synthetic record with NO depth/canny/caption (a minted image)
    m.upsert(ManifestRecord(id="synthetic001", sha256="s" * 12,
                            image="generated/synthetic001.png", class_name="palette",
                            width=64, height=64, mask="maps/mask/synthetic001.png",
                            synthetic=True))
    m.save()
    with _client() as c:
        s = c.get("/api/p/tiny/status").json()
    assert s["real"] == 3                 # 3 real curated, synthetic excluded
    assert s["depth"] == 3 and s["canny"] == 3      # not diluted to 3/4
    assert s["captioned"] == 3            # synthetic has no caption, not counted against real
    assert s["synthetic"] == 1


def test_status_reports_lora_trained(tiny_project, tmp_path):
    with _client() as c:
        assert c.get("/api/p/tiny/status").json()["lora_trained"] is False
        # drop a fake weights file where run_lora writes (settings.runs_dir/lora)
        wdir = tmp_path / "runs" / "lora" / "tiny_r16_x"
        wdir.mkdir(parents=True, exist_ok=True)
        (wdir / "pytorch_lora_weights.safetensors").write_bytes(b"x")
        assert c.get("/api/p/tiny/status").json()["lora_trained"] is True


def test_delete_project_removes_data_and_lora(tiny_project, tmp_path):
    pdir, _ = tiny_project
    wdir = tmp_path / "runs" / "lora" / "tiny_r16_x"
    wdir.mkdir(parents=True, exist_ok=True)
    (wdir / "pytorch_lora_weights.safetensors").write_bytes(b"x")
    with _client() as c:
        assert c.delete("/api/projects/tiny").json()["ok"] is True
        assert not pdir.exists()
        assert not wdir.exists()
        assert c.get("/api/p/tiny/status").status_code == 404
        assert c.delete("/api/projects/tiny").status_code == 404      # already gone


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
