"""LabelMe mask import at curate time — rasterization + ingest integration.

Hermetic: pure cv2/numpy + a tiny on-disk project. No SAM2/GPU.
"""

import json

import cv2
import numpy as np
from src.core.manifest import Manifest
from src.core.project import ClassSpec, ProjectConfig, create_project, load_project
from src.core.runners import run_curate
from src.stages.curate.mask_import import labelme_to_mask, sidecar_json


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
