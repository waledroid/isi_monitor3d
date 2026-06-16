"""Copy-paste scaffold compositing + the inpaint generator's input guard.
Hermetic — pure cv2/numpy, no GPU, no model load."""

import cv2
import numpy as np
import pytest
from src.core.manifest import Manifest, ManifestRecord
from src.core.project import ClassSpec, create_project, load_project
from src.stages.generation.base import IMAGE_GENERATORS
from src.stages.scaffolds.copy_paste import CopyPasteScaffolds

POLY = ClassSpec(name="polybag", trigger="ISI_PLYBG", color=[40, 90, 230])  # BGR (230,90,40)


def _record(pdir, rid, *, obj_box):
    """Write image/depth/mask for a record with a polybag-colored object box."""
    h, w = 80, 80
    img = np.full((h, w, 3), 50, np.uint8)
    depth = np.full((h, w), 120, np.uint8)
    mask = np.zeros((h, w, 3), np.uint8)
    y0, y1, x0, x1 = obj_box
    img[y0:y1, x0:x1] = (200, 200, 200)
    depth[y0:y1, x0:x1] = 255
    mask[y0:y1, x0:x1] = (230, 90, 40)                # polybag color (BGR)
    cv2.imwrite(str(pdir / f"raw/polybag/{rid}.jpg"), img)
    cv2.imwrite(str(pdir / f"maps/depth/{rid}.png"), depth)
    cv2.imwrite(str(pdir / f"maps/mask/{rid}.png"), mask)
    return ManifestRecord(id=rid, sha256=rid, image=f"raw/polybag/{rid}.jpg",
                          class_name="polybag", width=w, height=h,
                          depth_map=f"maps/depth/{rid}.png", mask=f"maps/mask/{rid}.png")


def test_copy_paste_emits_base_inpaint_and_label(tmp_path):
    pdir = create_project(tmp_path / "data", "cp", [POLY])
    for sub in ("raw/polybag", "maps/depth", "maps/mask"):
        (pdir / sub).mkdir(parents=True, exist_ok=True)
    m = Manifest.load(pdir)
    m.upsert(_record(pdir, "bgrec", obj_box=(10, 25, 10, 25)))
    m.upsert(_record(pdir, "objrec", obj_box=(30, 60, 30, 60)))
    m.save()

    src = CopyPasteScaffolds(project_dir=str(pdir), seed=1)
    out = list(src.generate(load_project(pdir), 3))
    assert len(out) == 3
    depth, label, meta = out[0]
    base, inpaint = meta["base"], meta["inpaint"]

    # shapes + types
    assert base.shape == (80, 80, 3) and depth.shape == (80, 80)
    assert label.shape == (80, 80, 3) and inpaint.shape == (80, 80)
    assert meta["classes"] == ["polybag"]
    # the label carries the polybag class color, and the inpaint region is non-empty
    assert np.all(label == (230, 90, 40), axis=2).any()
    assert inpaint.max() == 255 and inpaint.any()
    # background's OWN object is still labeled (no false negative): label has >= the
    # bg object's pixels worth of polybag color
    assert int(np.all(label == (230, 90, 40), axis=2).sum()) >= 15 * 15


def test_inpaint_generator_requires_base_and_mask():
    gen = IMAGE_GENERATORS.create("sdxl_inpaint")
    with pytest.raises(ValueError):
        gen.generate("a polybag", np.zeros((8, 8), np.uint8), seed=0)   # no base/mask
