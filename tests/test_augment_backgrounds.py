"""Tests for tools/augment_backgrounds.py (hermetic — synthetic images)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

_ROOT = Path(__file__).resolve().parents[1]


def _load(modname):
    spec = importlib.util.spec_from_file_location(
        modname, _ROOT / "tools" / f"{modname}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ab = _load("augment_backgrounds")
mcd = _load("make_crop_dataset")


def _structured_img(size=64):
    img = np.full((size, size, 3), 120, np.uint8)
    cv2.circle(img, (size // 2, size // 2), size // 6, (30, 200, 60), -1)
    cv2.rectangle(img, (4, 4), (size // 3, size // 3), (200, 40, 40), -1)
    return img


def test_variant_name_encodes_index_in_thousands():
    assert ab.variant_name("foo_bar_0007.jpg", 1) == "foo_bar_1007.jpg"
    assert ab.variant_name("foo_bar_0007.jpg", 3) == "foo_bar_3007.jpg"
    assert ab.variant_name("x_0193.png", 2) == "x_2193.jpg"     # always .jpg out


def test_variant_name_suffixless_fallback():
    assert ab.variant_name("plain.jpg", 1) == "plain_1000.jpg"
    assert ab.variant_name("plain.jpg", 4) == "plain_4000.jpg"


def test_variant_split_matches_original_series():
    orig = "platform__bg_cam_a_Sortie_1_0007.jpg"
    for v in (1, 2, 3, 4):
        assert mcd.bg_split(ab.variant_name(orig, v)) == mcd.bg_split(orig)


def test_augment_image_shape_dtype_contiguous_and_differs():
    img = _structured_img()
    out = ab.augment_image(img, np.random.default_rng(0))
    assert out.shape == img.shape and out.dtype == np.uint8
    assert out.flags["C_CONTIGUOUS"]
    diff = np.abs(out.astype(np.float32) - img.astype(np.float32)).mean()
    assert diff > 1.0


def test_augment_image_deterministic_per_rng_state():
    img = _structured_img()
    a = ab.augment_image(img, np.random.default_rng(42))
    b = ab.augment_image(img, np.random.default_rng(42))
    assert np.array_equal(a, b)
    c = ab.augment_image(img, np.random.default_rng(43))
    assert not np.array_equal(a, c)
