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


def _bg_record(pdir, rid):
    """A background-only record: image + depth, NO mask, background=True."""
    h, w = 80, 80
    img = np.full((h, w, 3), 70, np.uint8)
    depth = np.full((h, w), 100, np.uint8)
    (pdir / "raw/__bg__").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(pdir / f"raw/__bg__/{rid}.jpg"), img)
    cv2.imwrite(str(pdir / f"maps/depth/{rid}.png"), depth)
    return ManifestRecord(id=rid, sha256=rid, image=f"raw/__bg__/{rid}.jpg",
                          class_name="", width=w, height=h, background=True,
                          depth_map=f"maps/depth/{rid}.png")


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


def _cp_project_with_bg(tmp_path, name):
    pdir = create_project(tmp_path / "data", name, [POLY])
    for sub in ("raw/polybag", "raw/__bg__", "maps/depth", "maps/mask"):
        (pdir / sub).mkdir(parents=True, exist_ok=True)
    m = Manifest.load(pdir)
    m.upsert(_record(pdir, "obj", obj_box=(30, 55, 30, 55)))   # the only OBJECT source
    m.upsert(_bg_record(pdir, "bgempty"))                      # an empty background
    m.save()
    return pdir


def test_copy_paste_prefers_empty_background_no_overlap(tmp_path):
    """D1: with a background-only image, paste onto it — the label contains ONLY
    the pasted object (no pre-existing object) and the bg is the empty record."""
    pdir = _cp_project_with_bg(tmp_path, "cpbg")
    src = CopyPasteScaffolds(project_dir=str(pdir), seed=3)
    _depth, label, meta = next(src.generate(load_project(pdir), 1))
    assert meta["from_bg"] == "bgempty"                        # pasted onto the empty bg
    assert "obj" in meta["from_obj"]
    # exactly the pasted region is labeled — the empty bg contributes no object,
    # so the polybag-colored area equals the pasted (inpaint-core) region.
    labeled = np.all(label == (230, 90, 40), axis=2)
    assert labeled.any()
    # nothing labeled outside the single pasted blob → no doubling/overlap artifact
    n_labels, _ = cv2.connectedComponents(labeled.astype(np.uint8))
    assert n_labels == 2                                       # background + 1 object


def test_paste_count_range_draws_one_or_two(tmp_path):
    """D2: paste_count=[1,2] lands 1 or 2 objects; the int form lands exactly N."""
    pdir = _cp_project_with_bg(tmp_path, "cpn")
    proj = load_project(pdir)
    rng_src = CopyPasteScaffolds(project_dir=str(pdir), seed=5, paste_count=[1, 2])
    counts = {len(meta["from_obj"]) for _d, _l, meta in rng_src.generate(proj, 8)}
    assert counts <= {1, 2} and counts                         # only 1s and 2s
    two = CopyPasteScaffolds(project_dir=str(pdir), seed=5, paste_count=2)
    _d, _l, meta = next(two.generate(proj, 1))
    assert len(meta["from_obj"]) == 2


def test_copy_paste_falls_back_to_object_images_without_backgrounds(tmp_path):
    """D1 fallback: no background-only images → paste onto an object image (the
    pre-S-bg behavior), and the bg's own object stays labeled."""
    pdir = create_project(tmp_path / "data", "cpfb", [POLY])
    for sub in ("raw/polybag", "maps/depth", "maps/mask"):
        (pdir / sub).mkdir(parents=True, exist_ok=True)
    m = Manifest.load(pdir)
    m.upsert(_record(pdir, "a", obj_box=(10, 25, 10, 25)))
    m.upsert(_record(pdir, "b", obj_box=(40, 65, 40, 65)))
    m.save()
    src = CopyPasteScaffolds(project_dir=str(pdir), seed=1)
    _d, label, meta = next(src.generate(load_project(pdir), 1))
    assert meta["from_bg"] in ("a", "b")                       # an object image, not a bg
    # both the bg's own object AND the pasted one are labeled
    assert int(np.all(label == (230, 90, 40), axis=2).sum()) >= 15 * 15
