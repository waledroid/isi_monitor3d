"""``preprocess.letterbox`` — aspect-preserving resize + padding + inverse map."""

from __future__ import annotations

import numpy as np
import pytest

from backbone.detection.preprocess import (
    LETTERBOX_PAD_VALUE,
    batch_letterbox,
    invert_letterbox_xyxy,
    letterbox,
)


def _bgr(h: int, w: int, fill: int = 0) -> np.ndarray:
    return np.full((h, w, 3), fill, dtype=np.uint8)


def test_square_input_produces_square_target() -> None:
    img = _bgr(640, 640, fill=10)
    res = letterbox(img, target=(640, 640))
    assert res.tensor.shape == (3, 640, 640)
    assert res.tensor.dtype == np.float32
    assert res.scale == 1.0
    assert res.pad_xy == (0, 0)
    assert res.source_shape_hw == (640, 640)


def test_landscape_input_letterboxed_with_padding_on_top_bottom() -> None:
    """1920x1080 → fits to 640x360 inside 640x640, padded top+bottom."""
    img = _bgr(1080, 1920, fill=200)
    res = letterbox(img, target=(640, 640))

    assert res.tensor.shape == (3, 640, 640)
    assert res.scale == pytest.approx(640 / 1920)
    new_h = round(1080 * res.scale)   # 360
    pad_y = (640 - new_h) // 2        # 140
    assert res.pad_xy == (0, pad_y)
    assert res.source_shape_hw == (1080, 1920)


def test_portrait_input_letterboxed_with_padding_on_sides() -> None:
    img = _bgr(1920, 1080, fill=50)
    res = letterbox(img, target=(640, 640))
    new_w = round(1080 * res.scale)
    pad_x = (640 - new_w) // 2
    assert res.pad_xy == (pad_x, 0)


def test_padding_uses_neutral_grey() -> None:
    """The padded region must be the Ultralytics default 114."""
    img = _bgr(100, 200, fill=255)   # white image, will be letterboxed top+bottom
    res = letterbox(img, target=(640, 640))
    # Sample a pixel deep in the top pad band — must be the pad value, normalized.
    expected = LETTERBOX_PAD_VALUE / 255.0
    assert res.tensor[0, 5, 5] == pytest.approx(expected)
    assert res.tensor[1, 5, 5] == pytest.approx(expected)
    assert res.tensor[2, 5, 5] == pytest.approx(expected)


def test_bgr_to_rgb_swap_applied() -> None:
    """A pure-red BGR pixel must become a pure-red RGB tensor."""
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    img[..., 2] = 255  # OpenCV BGR: red is channel 2
    res = letterbox(img)
    # RGB channels: red=0, green=1, blue=2. Channel 0 should be ~1.0, others ~0.
    cy, cx = 320, 320
    assert res.tensor[0, cy, cx] == pytest.approx(1.0)
    assert res.tensor[1, cy, cx] == pytest.approx(0.0)
    assert res.tensor[2, cy, cx] == pytest.approx(0.0)


def test_invert_letterbox_round_trip_for_known_box() -> None:
    """A bbox detected in target frame must map back to its source coords."""
    img = _bgr(1080, 1920)
    res = letterbox(img, target=(640, 640))

    # Source box covers (x=200..400, y=300..600) in the original 1920x1080.
    # Forward map: scale, then pad.
    src_box = np.array([[200.0, 300.0, 400.0, 600.0]])
    tgt_x1 = src_box[0, 0] * res.scale + res.pad_xy[0]
    tgt_y1 = src_box[0, 1] * res.scale + res.pad_xy[1]
    tgt_x2 = src_box[0, 2] * res.scale + res.pad_xy[0]
    tgt_y2 = src_box[0, 3] * res.scale + res.pad_xy[1]
    tgt_box = np.array([[tgt_x1, tgt_y1, tgt_x2, tgt_y2]])

    inverted = invert_letterbox_xyxy(tgt_box, res)
    np.testing.assert_allclose(inverted, src_box, atol=1e-4)


def test_invert_letterbox_clips_to_image_bounds() -> None:
    img = _bgr(100, 200)
    res = letterbox(img, target=(640, 640))
    # Box that extends past the source image after inversion.
    target_box = np.array([[0.0, 0.0, 640.0, 640.0]])
    out = invert_letterbox_xyxy(target_box, res)
    assert out[0, 0] >= 0.0
    assert out[0, 1] >= 0.0
    assert out[0, 2] <= 200 - 1
    assert out[0, 3] <= 100 - 1


def test_invert_empty_returns_empty() -> None:
    img = _bgr(100, 100)
    res = letterbox(img)
    empty = np.empty((0, 4))
    out = invert_letterbox_xyxy(empty, res)
    assert out.shape == (0, 4)


def test_letterbox_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
        letterbox(np.zeros((10, 10), dtype=np.uint8))


def test_letterbox_rejects_wrong_dtype() -> None:
    with pytest.raises(ValueError, match="uint8"):
        letterbox(np.zeros((10, 10, 3), dtype=np.float32))


def test_batch_letterbox_stacks_correctly() -> None:
    imgs = [_bgr(720, 1280), _bgr(480, 640)]
    batch, results = batch_letterbox(imgs)
    assert batch.shape == (2, 3, 640, 640)
    assert batch.dtype == np.float32
    assert len(results) == 2
    # Different aspect ratios should produce different scales.
    assert results[0].scale != results[1].scale


def test_batch_letterbox_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one"):
        batch_letterbox([])
