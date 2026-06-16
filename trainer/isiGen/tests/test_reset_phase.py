"""Per-phase reset — wipe a phase's outputs so it re-runs clean. Hermetic.

Pins the decisions: mask reset keeps SAM2 prompts; caption reset keeps
hand-edited captions; mint reset removes synthetic records + re-pends scaffolds.
"""


import pytest
from src.core.manifest import Manifest, MaskPrompt
from src.core.project import ClassSpec, create_project, load_project
from src.core.runners import RESETTABLE, load_scaffold_index, reset_phase, save_scaffold_index


def _proj(tmp_path):
    pdir = create_project(tmp_path / "data", "r",
                          [ClassSpec(name="palette", trigger="ISI_PLT", color=[220, 40, 40])])
    return pdir


def _touch(p):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")


def test_unknown_phase_rejected(tmp_path):
    with pytest.raises(ValueError):
        reset_phase(_proj(tmp_path), "curate")


def test_reset_maps_clears_fields_and_files(tmp_path):
    pdir = _proj(tmp_path)
    m = Manifest.load(pdir)
    from src.core.manifest import ManifestRecord
    rec = ManifestRecord(id="a", sha256="a", image="raw/palette/a.jpg",
                         class_name="palette", depth_map="maps/depth/a.png",
                         canny_map="maps/canny/a.png")
    m.upsert(rec)
    m.save()
    _touch(pdir / "maps/depth/a.png")
    _touch(pdir / "maps/canny/a.png")
    out = reset_phase(pdir, "maps")
    assert out["files_deleted"] == 2 and out["records_cleared"] == 1
    rec2 = Manifest.load(pdir).get("a")
    assert rec2.depth_map is None and rec2.canny_map is None
    assert not (pdir / "maps/depth/a.png").exists()


def test_reset_masks_keeps_prompts(tmp_path):
    pdir = _proj(tmp_path)
    from src.core.manifest import ManifestRecord
    m = Manifest.load(pdir)
    m.upsert(ManifestRecord(id="a", sha256="a", image="raw/palette/a.jpg",
                            class_name="palette", mask="maps/mask/a.png",
                            mask_source="auto", needs_review=True,
                            mask_prompts=[MaskPrompt(kind="point", class_name="palette",
                                                     xy=[1, 2], label=1)]))
    m.save()
    _touch(pdir / "maps/mask/a.png")
    reset_phase(pdir, "masks")
    rec = Manifest.load(pdir).get("a")
    assert rec.mask is None and rec.mask_source is None and rec.needs_review is False
    assert len(rec.mask_prompts) == 1               # prompts preserved
    assert not (pdir / "maps/mask/a.png").exists()


def test_reset_captions_keeps_hand_edited(tmp_path):
    pdir = _proj(tmp_path)
    from src.core.manifest import ManifestRecord
    m = Manifest.load(pdir)
    m.upsert(ManifestRecord(id="auto", sha256="a", image="raw/palette/auto.jpg",
                            class_name="palette", caption_path="captions/auto.txt",
                            caption_edited=False))
    m.upsert(ManifestRecord(id="hand", sha256="b", image="raw/palette/hand.jpg",
                            class_name="palette", caption_path="captions/hand.txt",
                            caption_edited=True))
    m.save()
    _touch(pdir / "captions/auto.txt")
    _touch(pdir / "captions/hand.txt")
    out = reset_phase(pdir, "captions")
    assert out["records_cleared"] == 1
    recs = Manifest.load(pdir).records
    assert recs["auto"].caption_path is None
    assert recs["hand"].caption_path == "captions/hand.txt"     # edited kept
    assert not (pdir / "captions/auto.txt").exists()
    assert (pdir / "captions/hand.txt").exists()


def test_reset_scaffolds_deletes_dir(tmp_path):
    pdir = _proj(tmp_path)
    save_scaffold_index(pdir, [{"id": "sc0", "control": "scaffolds/sc0_control.png",
                                "mask": "scaffolds/sc0_mask.png", "status": "pending"}])
    _touch(pdir / "scaffolds/sc0_control.png")
    reset_phase(pdir, "scaffolds")
    assert not (pdir / "scaffolds").exists()


def test_run_scaffolds_writes_files_after_reset(tmp_path):
    """Regression: reset deletes scaffolds/, so the re-run MUST recreate the dir
    or cv2.imwrite fails silently → an index full of entries with no images."""
    import shutil

    from src.core.project import load_project, save_project
    from src.core.runners import run_scaffolds
    pdir = _proj(tmp_path)
    # box3d_procedural needs no real data; force single source for a hermetic run
    proj = load_project(pdir)
    proj.phases["scaffolds"]["sources"] = ["box3d_procedural"]
    save_project(pdir, proj)
    shutil.rmtree(pdir / "scaffolds")                 # simulate a prior reset
    out = run_scaffolds(pdir, count=3)
    assert out["scaffolds"] == 3
    pngs = list((pdir / "scaffolds").glob("*.png"))
    assert len(pngs) == 6                             # 3 pairs, control + mask each
    # every index entry has its files on disk (no phantom "pending without image")
    for e in load_scaffold_index(pdir):
        assert (pdir / e["control"]).exists() and (pdir / e["mask"]).exists()


def test_reset_generate_removes_synthetic_and_repends(tmp_path):
    pdir = _proj(tmp_path)
    from src.core.manifest import ManifestRecord
    m = Manifest.load(pdir)
    m.upsert(ManifestRecord(id="real1", sha256="r", image="raw/palette/r.jpg",
                            class_name="palette"))
    m.upsert(ManifestRecord(id="syn000000", sha256="s", image="generated/syn000000.png",
                            class_name="palette", synthetic=True))
    m.save()
    _touch(pdir / "generated/syn000000.png")
    save_scaffold_index(pdir, [{"id": "sc000000", "status": "generated"}])
    out = reset_phase(pdir, "generate")
    assert out["synthetic_removed"] == 1 and out["scaffolds_repending"] == 1
    recs = Manifest.load(pdir).records
    assert "syn000000" not in recs and "real1" in recs        # real kept
    assert load_scaffold_index(pdir)[0]["status"] == "pending"


def test_reset_lora_clears_weights_and_runs(tmp_path):
    pdir = _proj(tmp_path)
    proj = load_project(pdir)
    proj.phase("generation")["lora_weights"] = "/some/run"
    from src.core.project import save_project
    save_project(pdir, proj)
    runs = tmp_path / "runs"
    (runs / "lora" / "r_r16_x").mkdir(parents=True)
    (runs / "lora" / "r_r16_x" / "pytorch_lora_weights.safetensors").write_bytes(b"x")
    out = reset_phase(pdir, "lora", runs_dir=runs)
    assert out["runs_deleted"] == 1
    assert not (runs / "lora" / "r_r16_x").exists()
    assert load_project(pdir).phase("generation").get("lora_weights") is None


def test_resettable_list():
    assert set(RESETTABLE) == {"maps", "masks", "captions", "lora",
                               "scaffolds", "generate", "export"}


def test_captions_skip_synthetic(tmp_path):
    """Phase 4 captions only real curated images, not minted (synthetic) records."""
    import cv2
    import numpy as np
    from src.core.manifest import Manifest, ManifestRecord
    from src.core.runners import run_captions
    pdir = _proj(tmp_path)                              # class: palette
    m = Manifest.load(pdir)
    (pdir / "raw/palette").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(pdir / "raw/palette/r.jpg"), np.zeros((40, 40, 3), np.uint8))
    m.upsert(ManifestRecord(id="real1", sha256="r", image="raw/palette/r.jpg",
                            class_name="palette"))
    m.upsert(ManifestRecord(id="syn1", sha256="s", image="generated/syn1.png",
                            class_name="palette", synthetic=True))
    m.save()
    out = run_captions(pdir)
    assert out["captioned"] == 1                        # only the real record
    assert Manifest.load(pdir).get("syn1").caption_path is None
