"""Phase 8b exporters — color mask → YOLO-seg polygons / LabelMe JSON."""

import json

import cv2
import numpy as np
import yaml
from src.core.manifest import Manifest
from src.core.runners import run_export
from src.stages.exporting.yolo_seg import mask_to_polygons


def _paint_masks(pdir, project):
    m = Manifest.load(pdir)
    for i, rec in enumerate(m.active()):
        spec = project.classes[i % len(project.classes)]
        mask = np.zeros((rec.height, rec.width, 3), dtype=np.uint8)
        r, g, b = spec.color
        mask[100:400, 150:600] = (b, g, r)
        cv2.imwrite(str(pdir / f"maps/mask/{rec.id}.png"), mask)
        rec.mask = f"maps/mask/{rec.id}.png"
        m.upsert(rec)
    m.save()


def test_mask_to_polygons_rectangle():
    mask = np.zeros((200, 300, 3), dtype=np.uint8)
    mask[50:150, 100:200] = (10, 20, 30)            # BGR for color RGB(30,20,10)
    polys = mask_to_polygons(mask, [30, 20, 10])
    assert len(polys) == 1
    assert len(polys[0]) >= 4                        # ~rectangle
    xs, ys = polys[0][:, 0], polys[0][:, 1]
    assert 95 <= xs.min() <= 105 and 145 <= ys.max() <= 155


def test_run_export_yolo_and_labelme(tiny_project):
    pdir, project = tiny_project
    _paint_masks(pdir, project)
    # configure both exporters
    import yaml as _y
    cfg = _y.safe_load((pdir / "project.yaml").read_text())
    cfg["phases"]["export"]["exporters"] = ["yolo_seg", "labelme"]
    (pdir / "project.yaml").write_text(_y.safe_dump(cfg, sort_keys=False))

    out = run_export(pdir)
    assert out["records"] == 3
    # ---- yolo_seg ----
    root = pdir / "export" / "yolo_seg"
    data = yaml.safe_load((root / "data.yaml").read_text())
    assert data["nc"] == 3 and data["names"] == project.class_names()
    labels = list((root / "labels").rglob("*.txt"))
    images = list((root / "images").rglob("*.*"))
    assert len(labels) == 3 and len(images) == 3
    for lf in labels:
        for line in lf.read_text().splitlines():
            parts = line.split()
            cls = int(parts[0])
            coords = [float(v) for v in parts[1:]]
            assert 0 <= cls < 3
            assert len(coords) >= 6 and len(coords) % 2 == 0
            assert all(0.0 <= v <= 1.0 for v in coords)
    # ---- labelme ----
    lroot = pdir / "export" / "labelme"
    jsons = list(lroot.glob("*.json"))
    assert len(jsons) == 3
    doc = json.loads(jsons[0].read_text())
    assert doc["shapes"] and doc["shapes"][0]["shape_type"] == "polygon"
    assert doc["shapes"][0]["label"] in project.class_names()
    assert (lroot / doc["imagePath"]).exists()


def test_label_mode_export_includes_background_negatives(tmp_path):
    """A label-mode project exports object images (polygon shapes) AND background
    images (no mask) as empty-label LabelMe negatives — both with their images."""
    from src.core.manifest import Manifest, ManifestRecord
    from src.core.project import ClassSpec, create_project, load_project
    pdir = create_project(tmp_path / "data", "lbl",
                          [ClassSpec(name="polybag", trigger="ISI_PLYBG", color=[40, 90, 230])],
                          mode="label")
    project = load_project(pdir)
    assert project.mode == "label"
    assert project.phase("export")["exporters"] == ["labelme"]   # default for label mode

    for sub in ("raw/polybag", "raw/__bg__", "maps/mask"):
        (pdir / sub).mkdir(parents=True, exist_ok=True)
    m = Manifest.load(pdir)
    # object record (with a polybag-colored mask)
    cv2.imwrite(str(pdir / "raw/polybag/obj.jpg"), np.full((200, 300, 3), 50, np.uint8))
    mask = np.zeros((200, 300, 3), np.uint8)
    mask[50:150, 100:200] = (230, 90, 40)                         # BGR for RGB(40,90,230)
    cv2.imwrite(str(pdir / "maps/mask/obj.png"), mask)
    m.upsert(ManifestRecord(id="obj", sha256="o" * 12, image="raw/polybag/obj.jpg",
                            class_name="polybag", width=300, height=200,
                            mask="maps/mask/obj.png"))
    # background negative (no mask)
    cv2.imwrite(str(pdir / "raw/__bg__/bg.jpg"), np.full((200, 300, 3), 70, np.uint8))
    m.upsert(ManifestRecord(id="bg", sha256="b" * 12, image="raw/__bg__/bg.jpg",
                            class_name="", width=300, height=200, background=True))
    m.save()

    out = run_export(pdir)
    assert out["records"] == 2                                    # object + background both selected
    lroot = pdir / "export" / "labelme"
    obj_doc = json.loads((lroot / "obj.json").read_text())
    bg_doc = json.loads((lroot / "bg.json").read_text())
    assert obj_doc["shapes"] and obj_doc["shapes"][0]["label"] == "polybag"
    assert bg_doc["shapes"] == []                                 # background = empty negative
    assert (lroot / "obj.jpg").exists() and (lroot / "bg.jpg").exists()


def test_export_clip_two_versions(tiny_project, monkeypatch):
    """clip_filter=True drops low-score mints → export/ ; =False keeps all → export_noclip/."""
    from src.core.manifest import Manifest, ManifestRecord
    from src.stages.filtering.base import QUALITY_FILTERS
    pdir, project = tiny_project
    _paint_masks(pdir, project)                       # 3 real records get masks
    # add one synthetic mint (image + mask + caption)
    h, w = 480, 640
    (pdir / "generated").mkdir(exist_ok=True)
    cv2.imwrite(str(pdir / "generated/synZ.png"), np.full((h, w, 3), 60, np.uint8))
    c = project.classes[0].color
    msk = np.zeros((h, w, 3), np.uint8)
    msk[50:200, 50:200] = (c[2], c[1], c[0])
    cv2.imwrite(str(pdir / "maps/mask/synZ.png"), msk)
    (pdir / "captions").mkdir(exist_ok=True)
    (pdir / "captions/synZ.txt").write_text("a photo of ISI_PLT")
    m = Manifest.load(pdir)
    m.upsert(ManifestRecord(id="synZ", sha256="z" * 12, image="generated/synZ.png",
                            class_name=project.classes[0].name, width=w, height=h,
                            mask="maps/mask/synZ.png", caption_path="captions/synZ.txt",
                            synthetic=True))
    m.save()

    class _Scorer:                                    # stub CLIP: synthetic scores 0 → dropped
        def load(self): pass
        def score(self, img, prompt): return 0.0
        def close(self): pass
    monkeypatch.setattr(QUALITY_FILTERS, "create", lambda name, **k: _Scorer())

    w_clip = run_export(pdir, clip_filter=True)
    assert w_clip["clip_filtered"] is True and w_clip["dropped"] == 1
    assert (pdir / "export").exists() and not (pdir / "export_noclip").exists()

    wo_clip = run_export(pdir, clip_filter=False)
    assert wo_clip["clip_filtered"] is False and wo_clip["dropped"] == 0
    assert (pdir / "export_noclip").exists()
    assert wo_clip["records"] == w_clip["records"] + 1   # the dropped synthetic is back


def test_no_clip_in_label_mode(tmp_path, monkeypatch):
    """Dataset-only (label) mode never runs CLIP, even if clip_filter=True is passed."""
    from src.core.project import ClassSpec, create_project
    from src.stages.filtering.base import QUALITY_FILTERS

    def _boom(*a, **k):
        raise AssertionError("CLIP must not run in label mode")
    monkeypatch.setattr(QUALITY_FILTERS, "create", _boom)
    pdir = create_project(tmp_path / "data", "lbl",
                          [ClassSpec(name="polybag", trigger="ISI_PLYBG", color=[40, 90, 230])],
                          mode="label")
    out = run_export(pdir, clip_filter=True)          # requested, but label → ignored
    assert out["clip_filtered"] is False and out["dropped"] == 0


def test_export_bg_negatives_capped_to_rule(tiny_project):
    """bg_fraction adds empty-label background negatives, capped to frac x positives."""
    import glob
    from pathlib import Path

    from src.core.manifest import Manifest, ManifestRecord
    pdir, project = tiny_project
    _paint_masks(pdir, project)                       # 3 positives (real, masked)
    (pdir / "raw/__bg__").mkdir(parents=True, exist_ok=True)
    m = Manifest.load(pdir)
    for i in range(5):                                # 5 backgrounds available
        cv2.imwrite(str(pdir / f"raw/__bg__/bg{i}.jpg"), np.full((200, 300, 3), 70, np.uint8))
        m.upsert(ManifestRecord(id=f"bg{i}", sha256=f"bgx{i}", image=f"raw/__bg__/bg{i}.jpg",
                                class_name="", width=300, height=200, background=True))
    m.save()

    # 30% of 3 positives → cap = 1 bg (capped well below the 5 available)
    out = run_export(pdir, clip_filter=False, bg_fraction=0.3)
    assert out["backgrounds"] == 1 and out["records"] == 4
    lbls = glob.glob(str(pdir / "export_noclip/yolo_seg/labels/**/bg*.txt"), recursive=True)
    assert lbls and Path(lbls[0]).read_text().strip() == ""   # background → empty label

    # opt-out: no fraction → no background negatives
    out0 = run_export(pdir, clip_filter=False, bg_fraction=0.0)
    assert out0["backgrounds"] == 0 and out0["records"] == 3
