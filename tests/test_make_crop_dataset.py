from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

# Loader block: import tools/make_crop_dataset.py as mcd
_spec = spec_from_file_location("mcd", Path(__file__).parent.parent / "tools" / "make_crop_dataset.py")
mcd = module_from_spec(_spec)
_spec.loader.exec_module(mcd)


def test_parse_label_file_denormalizes(tmp_path):
    p = tmp_path / "img.txt"
    p.write_text("1 0.25 0.5 0.75 0.5 0.5 1.0\n")
    objs = mcd.parse_label_file(p, 200, 100)
    assert len(objs) == 1
    cls, poly = objs[0]
    assert cls == 1
    assert np.allclose(poly, [[50, 50], [150, 50], [100, 100]])


def test_parse_label_file_skips_malformed(tmp_path):
    p = tmp_path / "img.txt"
    p.write_text("garbage line\n0 0.1 0.1 0.9 0.1 0.5 0.9\n")
    objs = mcd.parse_label_file(p, 100, 100)
    assert len(objs) == 1 and objs[0][0] == 0


def test_format_label_lines_clamps_and_rounds():
    poly = np.array([[0.123456789, -0.01], [1.02, 0.5], [0.5, 0.999999]])
    out = mcd.format_label_lines([(2, poly)])
    assert out == "2 0.123457 0.000000 1.000000 0.500000 0.500000 0.999999\n"
    assert mcd.format_label_lines([]) == ""


def test_poly_bbox_and_area():
    sq = np.array([[10.0, 20.0], [110.0, 20.0], [110.0, 70.0], [10.0, 70.0]])
    assert mcd.poly_bbox(sq) == (10.0, 20.0, 110.0, 70.0)
    assert mcd.poly_area(sq) == pytest.approx(100 * 50)


def test_clip_polygon_keeps_inside_region():
    sq = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]])
    clipped = mcd.clip_polygon(sq, 50, 0, 150, 100)
    assert clipped is not None
    assert mcd.poly_area(clipped) == pytest.approx(50 * 100)
    assert clipped[:, 0].min() >= 50 and clipped[:, 0].max() <= 100


def test_clip_polygon_outside_returns_none():
    sq = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    assert mcd.clip_polygon(sq, 50, 50, 100, 100) is None


def test_cluster_boxes_groups_overlapping():
    boxes = [(0, 0, 100, 100), (90, 0, 200, 100), (500, 500, 600, 600)]
    groups = mcd.cluster_boxes(boxes, expand_frac=0.1)
    assert sorted(map(sorted, groups)) == [[0, 1], [2]]


def test_cluster_boxes_far_apart_stay_separate():
    boxes = [(0, 0, 10, 10), (500, 500, 510, 510)]
    assert sorted(map(sorted, mcd.cluster_boxes(boxes, 0.25))) == [[0], [1]]


def test_crop_window_at_least_size_and_inside_image():
    rng = np.random.default_rng(0)
    x0, y0, x1, y1 = mcd.crop_window((800.0, 500.0, 900.0, 560.0),
                                     (1920, 1080), 384, (0.10, 0.25), rng)
    assert x1 - x0 >= 384 and y1 - y0 >= 384
    assert x0 >= 0 and y0 >= 0 and x1 <= 1920 and y1 <= 1080
    # window covers the bbox
    assert x0 <= 800 and x1 >= 900 and y0 <= 500 and y1 >= 560


def test_crop_window_clamps_on_small_image():
    rng = np.random.default_rng(0)
    x0, y0, x1, y1 = mcd.crop_window((10.0, 10.0, 40.0, 40.0),
                                     (200, 150), 384, (0.10, 0.25), rng)
    assert (x0, y0, x1, y1) == (0, 0, 200, 150)   # whole image, no overflow


def test_letterbox_downscales_and_pads():
    img = np.full((400, 800, 3), 200, np.uint8)
    canvas, scale, dx, dy = mcd.letterbox_to(img, 384)
    assert canvas.shape == (384, 384, 3)
    assert scale == pytest.approx(384 / 800)
    assert dx == 0 and dy == (384 - round(400 * scale)) // 2
    assert (canvas[0, 0] == mcd.GRAY).all()        # pad band
    assert (canvas[192, 192] == 200).all()         # content center


def test_letterbox_never_upscales():
    img = np.full((100, 100, 3), 50, np.uint8)
    canvas, scale, dx, dy = mcd.letterbox_to(img, 384)
    assert scale == 1.0 and dx == dy == (384 - 100) // 2
    assert (canvas[dy + 50, dx + 50] == 50).all()
    assert (canvas[0, 0] == mcd.GRAY).all()


def _sq(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)


def test_generate_crops_labels_full_object():
    img = np.full((1080, 1920, 3), 180, np.uint8)
    objs = [(0, _sq(800, 500, 1000, 650))]
    crops = mcd.generate_crops(img, objs, size=384, margin_range=(0.1, 0.25),
                               keep_frac=0.30, rng=np.random.default_rng(0))
    assert len(crops) == 1
    crop, labels = crops[0]
    assert crop.shape == (384, 384, 3)
    assert len(labels) == 1 and labels[0][0] == 0
    poly = labels[0][1]
    assert poly.min() >= 0.0 and poly.max() <= 1.0
    # object must occupy a plausible area share of the crop:
    # window >= 384 src px, object 200x150 -> >= (200*150)/(win_side^2) of crop
    assert mcd.poly_area(poly * 384) > 0.05 * 384 * 384


def test_generate_crops_two_far_objects_two_crops():
    img = np.full((1080, 1920, 3), 180, np.uint8)
    objs = [(0, _sq(100, 100, 260, 220)), (1, _sq(1500, 800, 1700, 950))]
    crops = mcd.generate_crops(img, objs, size=384, margin_range=(0.1, 0.25),
                               keep_frac=0.30, rng=np.random.default_rng(0))
    assert len(crops) == 2
    got = sorted(cls for _, labels in crops for cls, _ in labels)
    assert got == [0, 1]


def test_generate_crops_grayfills_barely_visible_neighbor():
    img = np.full((1080, 1920, 3), 180, np.uint8)
    # cls-0's window is deterministic with margin (0,0): the 384px square
    # clamps to x 0..384, y 288..672. The cls-1 slab (350..1200 px) pokes
    # 34 px into that window (~4% of its area, < keep-frac 0.30) ->
    # gray-filled, unlabeled. It gets its own second crop where it IS labeled.
    objs = [(0, _sq(50, 400, 250, 560)),
            (1, _sq(350, 380, 1200, 600))]
    crops = mcd.generate_crops(img, objs, size=384, margin_range=(0.0, 0.0),
                               keep_frac=0.30, rng=np.random.default_rng(0))
    assert len(crops) == 2
    crop, labels = next(c for c in crops if any(cl == 0 for cl, _ in c[1]))
    assert [cl for cl, _ in labels] == [0]       # slab below keep-frac: unlabeled
    assert (crop == mcd.GRAY).all(axis=2).any()  # ...and its sliver gray-filled


def test_generate_crops_keeps_neighbor_above_keep_frac():
    img = np.full((600, 600, 3), 180, np.uint8)
    objs = [(0, _sq(100, 100, 300, 300)), (1, _sq(320, 100, 500, 300))]
    crops = mcd.generate_crops(img, objs, size=384, margin_range=(0.1, 0.1),
                               keep_frac=0.30, rng=np.random.default_rng(0))
    assert len(crops) == 1                     # overlap after expansion -> one cluster
    _, labels = crops[0]
    assert sorted(cl for cl, _ in labels) == [0, 1]


def test_generate_crops_no_objects_full_frame_background():
    img = np.full((200, 300, 3), 90, np.uint8)
    crops = mcd.generate_crops(img, [], size=384, margin_range=(0.1, 0.25),
                               keep_frac=0.30, rng=np.random.default_rng(0))
    assert len(crops) == 1
    crop, labels = crops[0]
    assert labels == [] and crop.shape == (384, 384, 3)
    assert (crop[0, 0] == mcd.GRAY).all()      # padded, not upscaled


def test_generate_crops_small_image_not_upscaled():
    img = np.full((200, 200, 3), 90, np.uint8)
    objs = [(0, _sq(50, 50, 150, 150))]
    crops = mcd.generate_crops(img, objs, size=384, margin_range=(0.0, 0.0),
                               keep_frac=0.30, rng=np.random.default_rng(0))
    crop, labels = crops[0]
    # content occupies exactly 200x200 centered; object polygon maps inside it
    dx = (384 - 200) // 2
    poly = labels[0][1] * 384
    assert poly[:, 0].min() == pytest.approx(dx + 50, abs=1.5)
    assert (crop[0, 0] == mcd.GRAY).all()


@pytest.fixture
def _make_src(tmp_path, n_train=2, n_val=1):
    src = tmp_path / "src"
    for split, n in (("train", n_train), ("val", n_val)):
        (src / "images" / split).mkdir(parents=True)
        (src / "labels" / split).mkdir(parents=True)
        for i in range(n):
            img = np.full((600, 800, 3), 160, np.uint8)
            cv2.imwrite(str(src / "images" / split / f"{split}{i}.jpg"), img)
            (src / "labels" / split / f"{split}{i}.txt").write_text(
                "0 0.25 0.25 0.5 0.25 0.5 0.5 0.25 0.5\n")
    (src / "data.yaml").write_text(
        "path: X\ntrain: images/train\nval: images/val\nnc: 3\n"
        "names: ['palette', 'carton', 'polybag']\n")
    return src


def test_main_builds_split_preserving_dataset(_make_src, tmp_path):
    src = _make_src
    out = tmp_path / "out"
    rc = mcd.main(["--src", str(src), "--out", str(out), "--size", "384"])
    assert rc == 0
    train = sorted(p.name for p in (out / "images" / "train").glob("*.jpg"))
    val = sorted(p.name for p in (out / "images" / "val").glob("*.jpg"))
    assert train == ["train0_c0.jpg", "train1_c0.jpg"]
    assert val == ["val0_c0.jpg"]
    for stem in ("train/train0_c0", "train/train1_c0", "val/val0_c0"):
        txt = (out / "labels" / (stem + ".txt")).read_text()
        assert txt.startswith("0 ") and len(txt.split()) >= 7
    y = yaml.safe_load((out / "data.yaml").read_text())
    assert y["names"] == ["palette", "carton", "polybag"] and y["nc"] == 3
    img = cv2.imread(str(out / "images" / "train" / "train0_c0.jpg"))
    assert img.shape == (384, 384, 3)


def test_main_refuses_existing_out(_make_src, tmp_path):
    src = _make_src
    out = tmp_path / "out"
    (out / "images").mkdir(parents=True)
    assert mcd.main(["--src", str(src), "--out", str(out)]) == 2


def test_main_missing_src_exits_2_no_output(tmp_path):
    out = tmp_path / "o"
    rc = mcd.main(["--src", str(tmp_path / "nope"), "--out", str(out)])
    assert rc == 2
    assert not (out / "images").exists()


def test_main_backgrounds_skip_collision_and_convert_to_jpg(_make_src, tmp_path):
    src = _make_src
    bg = tmp_path / "bg"
    bg.mkdir()
    cv2.imwrite(str(bg / "extra.png"), np.full((300, 400, 3), 120, np.uint8))
    out = tmp_path / "out"
    rc = mcd.main(["--src", str(src), "--out", str(out),
                   "--backgrounds", str(bg)])
    assert rc == 0
    # png background written as .jpg, no label file created for it
    split = mcd.bg_split("extra.png")
    converted = out / "images" / split / "extra.jpg"
    assert converted.exists()
    assert not (out / "labels" / split / "extra.txt").exists()

    # a background named to collide with an existing crop must not overwrite
    # it. "train0_c0.jpg" is a real crop name from _make_src, and it hashes
    # (as a background name) into the "train" split -- the same split its
    # crop lives in -- so the collision is real, not coincidental.
    collide_name = "train0_c0"
    collide_split = "train"
    assert mcd.bg_split(f"{collide_name}.jpg") == collide_split
    crop_path = out / "images" / collide_split / (collide_name + ".jpg")
    assert crop_path.exists()
    before = crop_path.read_bytes()

    bg2 = tmp_path / "bg2"
    bg2.mkdir()
    cv2.imwrite(str(bg2 / f"{collide_name}.jpg"),
                np.full((300, 400, 3), 40, np.uint8))
    out2 = tmp_path / "out2"
    rc2 = mcd.main(["--src", str(src), "--out", str(out2),
                    "--backgrounds", str(bg2)])
    assert rc2 == 0
    crop2_path = out2 / "images" / collide_split / (collide_name + ".jpg")
    assert crop2_path.exists()
    assert crop2_path.read_bytes() == before


def test_main_folds_backgrounds_without_labels(_make_src, tmp_path):
    src = _make_src
    bg = tmp_path / "bg"
    bg.mkdir()
    for i in range(10):
        cv2.imwrite(str(bg / f"bg_{i:03d}.jpg"),
                    np.full((300, 400, 3), 120, np.uint8))
    out = tmp_path / "out"
    rc = mcd.main(["--src", str(src), "--out", str(out),
                   "--backgrounds", str(bg)])
    assert rc == 0
    bg_imgs = [p for s in ("train", "val")
               for p in (out / "images" / s).glob("bg_*.jpg")]
    assert len(bg_imgs) == 10
    for p in bg_imgs:
        assert not (out / "labels" / p.parent.name / (p.stem + ".txt")).exists()
        assert cv2.imread(str(p)).shape == (384, 384, 3)
    for p in bg_imgs:
        assert p.parent.name == mcd.bg_split(p.name)


def test_bg_split_deterministic():
    assert mcd.bg_split("a.jpg") == mcd.bg_split("a.jpg")
    assert all(mcd.bg_split(f"x{i}.jpg") in ("train", "val") for i in range(50))
    # same capture series (shared prefix before the trailing _NNNN.ext) stays
    # in one split, even though the frame numbers differ.
    assert mcd.bg_split("bg_cam_a_Sortie-1_0000.jpg") == \
        mcd.bg_split("bg_cam_a_Sortie-1_0042.jpg")
    # a name without a numeric suffix still resolves to a valid split.
    assert mcd.bg_split("no_suffix_here.jpg") in ("train", "val")


def test_main_preview_writes_only_preview(_make_src, tmp_path):
    src = _make_src
    out = tmp_path / "out"
    rc = mcd.main(["--src", str(src), "--out", str(out), "--preview", "2"])
    assert rc == 0
    previews = list((out / "_preview").glob("*.jpg"))
    assert len(previews) == 2
    assert not (out / "images").exists() and not (out / "labels").exists()
    # a full run AFTER preview is allowed (out contains only _preview)
    assert mcd.main(["--src", str(src), "--out", str(out)]) == 0
