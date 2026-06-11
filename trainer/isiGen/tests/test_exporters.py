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
