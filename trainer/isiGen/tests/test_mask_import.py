"""LabelMe mask import at curate time — rasterization + ingest integration.

Hermetic: pure cv2/numpy + a tiny on-disk project. No SAM2/GPU.
"""

import json

import cv2
import numpy as np
from src.core.manifest import Manifest
from src.core.project import ClassSpec, ProjectConfig, create_project, load_project
from src.core.runners import run_curate
from src.stages.curate.mask_import import (
    CocoIndex,
    ImportContext,
    _paint,
    import_mask,
    labelme_to_mask,
    load_coco,
    load_yolo_names,
    sidecar_json,
    yolo_polys,
)


def _project(classes):
    return ProjectConfig(name="t", classes=classes, phases={})


PALETTE = ClassSpec(name="palette", trigger="ISI_PLT", color=[220, 40, 40])
CARTON = ClassSpec(name="carton", trigger="ISI_CRTN", color=[40, 200, 40])


def _labelme(shapes, w=100, h=100):
    return {"imageWidth": w, "imageHeight": h, "shapes": shapes}


def test_polygon_paints_class_color_inside_only(tmp_path):
    proj = _project([PALETTE])
    j = tmp_path / "a.json"
    j.write_text(json.dumps(_labelme([
        {"label": "palette", "shape_type": "polygon",
         "points": [[20, 20], [60, 20], [60, 60], [20, 60]]}])))
    mask = labelme_to_mask(j, proj, (100, 100))
    assert mask is not None and mask.shape == (100, 100, 3)
    # inside → palette color in BGR (b,g,r) = (40,40,220); outside → black
    assert tuple(int(v) for v in mask[40, 40]) == (40, 40, 220)
    assert tuple(int(v) for v in mask[5, 5]) == (0, 0, 0)


def test_rectangle_shape_supported(tmp_path):
    proj = _project([PALETTE])
    j = tmp_path / "r.json"
    j.write_text(json.dumps(_labelme([
        {"label": "palette", "shape_type": "rectangle", "points": [[10, 10], [50, 50]]}])))
    mask = labelme_to_mask(j, proj, (100, 100))
    assert tuple(int(v) for v in mask[30, 30]) == (40, 40, 220)


def test_unknown_label_skipped(tmp_path):
    proj = _project([PALETTE])
    j = tmp_path / "u.json"
    j.write_text(json.dumps(_labelme([
        {"label": "not_a_class", "shape_type": "polygon",
         "points": [[0, 0], [9, 0], [9, 9]]}])))
    assert labelme_to_mask(j, proj, (100, 100)) is None


def test_points_scaled_when_dims_differ(tmp_path):
    proj = _project([PALETTE])
    # JSON declares 50x50; target is 100x100 → points double
    j = tmp_path / "s.json"
    j.write_text(json.dumps(_labelme([
        {"label": "palette", "shape_type": "rectangle", "points": [[10, 10], [20, 20]]}],
        w=50, h=50)))
    mask = labelme_to_mask(j, proj, (100, 100))
    assert tuple(int(v) for v in mask[30, 30]) == (40, 40, 220)   # 15,15→30,30
    assert tuple(int(v) for v in mask[5, 5]) == (0, 0, 0)


def test_multiclass_paints_both(tmp_path):
    proj = _project([PALETTE, CARTON])
    j = tmp_path / "m.json"
    j.write_text(json.dumps(_labelme([
        {"label": "palette", "shape_type": "rectangle", "points": [[5, 5], [25, 25]]},
        {"label": "carton", "shape_type": "rectangle", "points": [[60, 60], [90, 90]]}])))
    mask = labelme_to_mask(j, proj, (100, 100))
    assert tuple(int(v) for v in mask[15, 15]) == (40, 40, 220)    # palette
    assert tuple(int(v) for v in mask[75, 75]) == (40, 200, 40)    # carton BGR


def test_sidecar_json_detection(tmp_path):
    img = tmp_path / "x.jpg"
    img.write_bytes(b"x")
    assert sidecar_json(img) is None
    (tmp_path / "x.json").write_text("{}")
    assert sidecar_json(img) == tmp_path / "x.json"


# --------------------------- YOLO ---------------------------

def test_yolo_polygon_and_bbox(tmp_path):
    proj = _project([PALETTE, CARTON])
    t = tmp_path / "y.txt"
    # class 0 polygon (square 0.2..0.8) ; class 1 bbox center
    t.write_text("0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8\n1 0.75 0.75 0.1 0.1\n")
    names = [PALETTE.name, CARTON.name]
    mask = _paint(proj, yolo_polys(t, names, (100, 100)), (100, 100))
    assert tuple(int(v) for v in mask[40, 40]) == (40, 40, 220)    # palette poly
    assert tuple(int(v) for v in mask[75, 75]) == (40, 200, 40)    # carton bbox


def test_yolo_index_out_of_range_skipped(tmp_path):
    proj = _project([PALETTE])
    t = tmp_path / "y.txt"
    t.write_text("5 0.1 0.1 0.2 0.2\n")
    assert _paint(proj, yolo_polys(t, [PALETTE.name], (100, 100)), (100, 100)) is None


def test_yolo_names_from_data_yaml(tmp_path):
    (tmp_path / "data.yaml").write_text("names: [palette, carton]\n")
    assert load_yolo_names(tmp_path) == ["palette", "carton"]
    (tmp_path / "data.yaml").unlink()
    (tmp_path / "classes.txt").write_text("palette\ncarton\n")
    assert load_yolo_names(tmp_path) == ["palette", "carton"]


def test_run_curate_imports_yolo_sidecar(tmp_path):
    data_dir = tmp_path / "data"
    pdir = create_project(data_dir, "y", [PALETTE])
    src = tmp_path / "in"
    src.mkdir()
    cv2.imwrite(str(src / "p.png"), np.full((100, 100, 3), 60, np.uint8))
    (src / "p.txt").write_text("0 0.3 0.3 0.7 0.3 0.7 0.7 0.3 0.7\n")   # palette poly
    res = run_curate(pdir, source=str(src), class_name="palette")
    assert res["masks_imported"] == 1
    rec = next(iter(Manifest.load(pdir).records.values()))
    assert rec.mask_source == "imported" and (pdir / rec.mask).exists()


# --------------------------- COCO ---------------------------

def _coco(file_name, w=100, h=100):
    return {
        "images": [{"id": 1, "file_name": file_name, "width": w, "height": h}],
        "categories": [{"id": 7, "name": "palette"}, {"id": 9, "name": "carton"}],
        "annotations": [
            {"image_id": 1, "category_id": 7,
             "segmentation": [[20, 20, 60, 20, 60, 60, 20, 60]]},
            {"image_id": 1, "category_id": 9, "bbox": [70, 70, 20, 20]},   # bbox fallback
        ],
    }


def test_coco_polygon_and_bbox(tmp_path):
    proj = _project([PALETTE, CARTON])
    (tmp_path / "annotations.json").write_text(json.dumps(_coco("p.png")))
    idx = load_coco(tmp_path)
    assert isinstance(idx, CocoIndex)
    mask = _paint(proj, idx.polys_for("p.png", (100, 100)), (100, 100))
    assert tuple(int(v) for v in mask[40, 40]) == (40, 40, 220)    # palette poly
    assert tuple(int(v) for v in mask[78, 78]) == (40, 200, 40)    # carton bbox


def test_coco_scales_to_target(tmp_path):
    proj = _project([PALETTE])
    (tmp_path / "a.json").write_text(json.dumps(_coco("p.png", w=50, h=50)))
    idx = load_coco(tmp_path)
    mask = _paint(proj, idx.polys_for("p.png", (100, 100)), (100, 100))   # 2x scale
    assert tuple(int(v) for v in mask[80, 80]) == (40, 40, 220)    # 40,40→80,80


def test_run_curate_imports_coco(tmp_path):
    data_dir = tmp_path / "data"
    pdir = create_project(data_dir, "c", [PALETTE])
    src = tmp_path / "in"
    src.mkdir()
    cv2.imwrite(str(src / "p.png"), np.full((100, 100, 3), 60, np.uint8))
    (src / "_annotations.coco.json").write_text(json.dumps({
        "images": [{"id": 1, "file_name": "p.png", "width": 100, "height": 100}],
        "categories": [{"id": 1, "name": "palette"}],
        "annotations": [{"image_id": 1, "category_id": 1,
                         "segmentation": [[20, 20, 80, 20, 80, 80, 20, 80]]}],
    }))
    res = run_curate(pdir, source=str(src), class_name="palette")
    assert res["masks_imported"] == 1
    rec = next(iter(Manifest.load(pdir).records.values()))
    assert rec.mask_source == "imported" and (pdir / rec.mask).exists()


def test_import_precedence_coco_over_sidecars(tmp_path):
    proj = _project([PALETTE])
    img = tmp_path / "p.png"
    cv2.imwrite(str(img), np.full((100, 100, 3), 60, np.uint8))
    # COCO present (covers p.png) plus a LabelMe sidecar — COCO wins
    (tmp_path / "ann.json").write_text(json.dumps(_coco("p.png")))
    (tmp_path / "p.json").write_text(json.dumps(_labelme([
        {"label": "palette", "shape_type": "rectangle", "points": [[0, 0], [5, 5]]}])))
    ctx = ImportContext(coco=load_coco(tmp_path), yolo_names=None)
    mask = import_mask(img, proj, (100, 100), ctx)
    # COCO's palette polygon fills (40,40); the tiny LabelMe rect would not
    assert tuple(int(v) for v in mask[40, 40]) == (40, 40, 220)


def test_run_curate_imports_sidecar_mask(tmp_path):
    data_dir = tmp_path / "data"
    pdir = create_project(data_dir, "imp", [PALETTE])
    src = tmp_path / "incoming"
    src.mkdir()
    cv2.imwrite(str(src / "img.png"), np.full((100, 100, 3), 60, np.uint8))
    (src / "img.json").write_text(json.dumps(_labelme([
        {"label": "palette", "shape_type": "polygon",
         "points": [[20, 20], [80, 20], [80, 80], [20, 80]]}])))

    res = run_curate(pdir, source=str(src), class_name="palette")
    assert res["masks_imported"] == 1

    m = Manifest.load(pdir)
    rec = next(iter(m.records.values()))
    assert rec.mask is not None
    assert rec.mask_source == "imported"
    assert rec.needs_review is False
    assert (pdir / rec.mask).exists()
    load_project(pdir)                                   # project still valid
