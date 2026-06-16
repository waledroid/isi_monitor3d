import cv2
import numpy as np
import pytest
from src.core.manifest import Manifest
from src.core.project import ClassSpec, create_project
from src.core.runners import run_curate


def _setup(tmp_path):
    pdir = create_project(tmp_path / "data", "p", [
        ClassSpec(name="palette", trigger="ISI_PLT", color=[220, 40, 40])])
    src = tmp_path / "in"
    src.mkdir()
    img = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
    cv2.imwrite(str(src / "a.png"), img)
    cv2.imwrite(str(src / "b.png"), img[::-1])     # different content
    return pdir, src


def test_ingest_dedupe_idempotent(tmp_path):
    pdir, src = _setup(tmp_path)
    out1 = run_curate(pdir, source=str(src), class_name="palette")
    assert out1["added"] == 2
    out2 = run_curate(pdir, source=str(src), class_name="palette")
    assert out2["added"] == 0 and out2["skipped"] == 2        # idempotent
    m = Manifest.load(pdir)
    assert len(m.records) == 2
    for r in m.records.values():
        assert (pdir / r.image).exists()
        assert r.width == 800 and r.height == 600


def test_exif_stripped(tmp_path):
    from PIL import Image
    pdir, src = _setup(tmp_path)
    exif = Image.Exif()
    exif[0x010F] = "TestCam Corp"                              # Make tag
    Image.new("RGB", (700, 700), (90, 90, 90)).save(src / "c.jpg", exif=exif)
    run_curate(pdir, source=str(src), class_name="palette")
    m = Manifest.load(pdir)
    rec = next(r for r in m.records.values() if r.source_path.endswith("c.jpg"))
    with Image.open(pdir / rec.image) as out:
        assert dict(out.getexif()) == {}                       # metadata gone


def test_bad_class_rejected(tmp_path):
    pdir, src = _setup(tmp_path)
    with pytest.raises(KeyError):
        run_curate(pdir, source=str(src), class_name="not_a_class")


def test_auto_class_skips_unknown_dirs(tmp_path):
    pdir, src = _setup(tmp_path)
    (src / "palette").mkdir()
    cv2.imwrite(str(src / "palette" / "ok.png"),
                np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8))
    out = run_curate(pdir, source=str(src), auto_class=True)
    # a.png/b.png live in `in/` whose dir name isn't a class → skipped
    assert out["added"] == 1


def _img(seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (600, 800, 3), dtype=np.uint8)


def test_bg_folder_convention(tmp_path):
    """`<class>/img/*` tags the class; `<class>/bg/*` (or `background/`) becomes a
    background-only record — no class, no mask, marked background=True."""
    pdir = create_project(tmp_path / "data", "p", [
        ClassSpec(name="palette", trigger="ISI_PLT", color=[220, 40, 40])])
    src = tmp_path / "in"
    (src / "palette" / "img").mkdir(parents=True)
    (src / "palette" / "bg").mkdir(parents=True)
    cv2.imwrite(str(src / "palette" / "img" / "obj.png"), _img(1))    # object image
    cv2.imwrite(str(src / "palette" / "bg" / "empty1.png"), _img(2))  # background
    cv2.imwrite(str(src / "palette" / "bg" / "empty2.png"), _img(3))  # background

    out = run_curate(pdir, source=str(src), auto_class=True)
    assert out["added"] == 3 and out["backgrounds"] == 2

    m = Manifest.load(pdir)
    bgs = [r for r in m.records.values() if r.background]
    objs = [r for r in m.records.values() if not r.background]
    assert len(bgs) == 2 and len(objs) == 1
    # object image: tagged the class (img/ is a pass-through to <class>)
    assert objs[0].class_name == "palette"
    # backgrounds: no class, no mask, stored under raw/__bg__/
    for r in bgs:
        assert r.class_name == "" and r.mask is None
        assert r.image.startswith("raw/__bg__/") and (pdir / r.image).exists()


def test_bg_folder_wins_in_explicit_class_mode(tmp_path):
    """A `bg/` subfolder is treated as background even when a class is given."""
    pdir = create_project(tmp_path / "data", "p", [
        ClassSpec(name="palette", trigger="ISI_PLT", color=[220, 40, 40])])
    src = tmp_path / "in"
    src.mkdir()
    (src / "background").mkdir()
    cv2.imwrite(str(src / "obj.png"), _img(4))                # → palette
    cv2.imwrite(str(src / "background" / "e.png"), _img(5))   # → background
    out = run_curate(pdir, source=str(src), class_name="palette")
    assert out["added"] == 2 and out["backgrounds"] == 1
    m = Manifest.load(pdir)
    assert sum(r.background for r in m.records.values()) == 1
    assert sum(r.class_name == "palette" for r in m.records.values()) == 1
