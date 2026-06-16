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


def test_paste_not_tiny_respects_min_frac(tmp_path):
    """A1: a small-source object is still pasted at >= min_frac of the frame."""
    pdir = create_project(tmp_path / "data", "cp2", [POLY])
    for sub in ("raw/polybag", "maps/depth", "maps/mask"):
        (pdir / sub).mkdir(parents=True, exist_ok=True)
    m = Manifest.load(pdir)
    m.upsert(_record(pdir, "a", obj_box=(20, 26, 20, 26)))   # 6x6 object (native ~0.075)
    m.upsert(_record(pdir, "b", obj_box=(50, 56, 50, 56)))
    m.save()
    src = CopyPasteScaffolds(project_dir=str(pdir), seed=2, min_frac=0.25, dilate=1)
    _depth, _label, meta = next(src.generate(load_project(pdir), 1))
    ys = np.nonzero(meta["inpaint"].any(axis=1))[0]          # inpaint is HxW
    height = int(ys.max() - ys.min() + 1)
    assert height >= int(0.20 * 80)                          # floored, not the 6px native


def test_place_avoids_background_object(tmp_path):
    """A2: placement keeps the paste clear of the background's own object."""
    import random
    src = CopyPasteScaffolds(project_dir=".", seed=1, avoid_overlap=True,
                             placement_tries=80)
    W = H = 100
    bg_obj = np.zeros((H, W), bool)
    bg_obj[:, :50] = True                                    # left half occupied
    bin_r = np.ones((20, 20), bool)                          # solid object
    px, py = src._place(random.Random(0), W, H, 20, 20, bin_r, bg_obj)
    overlap = int(np.logical_and(bg_obj[py:py + 20, px:px + 20], bin_r).sum())
    assert overlap == 0 and px >= 50                         # placed on the free (right) side
