# Background Photometric Augmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A standalone script `tools/augment_backgrounds.py` that copies background crops and writes N photometric lighting variants per original (shadows, gradients, color temperature, gamma, noise, hflip), with series-preserving filenames so `make_crop_dataset.bg_split` keeps variants in the same train/val split as their source.

**Architecture:** Pure effect functions composed by `augment_image(img, rng)` (2–4 seeded random effects + p=0.5 hflip), a `variant_name()` helper encoding the variant index in the thousands digit of the numeric suffix, and a thin `main()` that copies originals + writes variants with refusal/skip handling.

**Tech Stack:** Python 3.10, numpy, OpenCV, pytest (hermetic — synthetic images only).

**Spec:** `docs/superpowers/specs/2026-08-05-bg-augmentation-design.md`

## Global Constraints

- Python 3.10 syntax only; run tests in the monitor3d env: `conda run -n monitor3d pytest tests/test_augment_backgrounds.py -v`.
- Defaults: `--variants 4`, `--seed 0`. Variants are JPEG quality 95, dimensions/dtype unchanged.
- Refusal: existing `--out` → exit 2. Unreadable source images warn + skip.
- Originals are copied byte-identical (`shutil.copy2`); source dir never written.
- Variant naming: `<series>_<NNNN>.jpg` → variant v is `<series>_{v*1000+NNNN:04d}.jpg`; suffix-less fallback `<stem>_{v*1000:04d}.jpg`.
- Determinism: same `--seed` → byte-identical outputs (single `np.random.default_rng(seed)` consumed over `sorted()` files).
- Tests load modules via importlib by file path (tools/ is not a package); the split-cooperation test imports `make_crop_dataset.bg_split` the same way.
- Commit after each green task, ONLY that task's two files, `git commit --no-verify`, ending every message with exactly:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_01CF51vq87B9hoGAkRkzWjf7`

## File Structure

- `tools/augment_backgrounds.py` — the whole tool (~150 lines).
- `tests/test_augment_backgrounds.py` — all tests.

Test file header (verbatim, once, module scope):

```python
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
```

---

### Task 1: Effects, `augment_image`, `variant_name`

**Files:**
- Create: `tools/augment_backgrounds.py`
- Test: `tests/test_augment_backgrounds.py`

**Interfaces:**
- Produces:
  - `variant_name(name: str, v: int) -> str` (v is 1-based).
  - `augment_image(img: np.ndarray, rng: np.random.Generator) -> np.ndarray` — uint8 BGR in, same-shape contiguous uint8 out; composes 2–4 effects + hflip p=0.5.
  - Private effects `_gamma, _contrast_brightness, _color_temperature, _gradient, _shadows, _noise` each `(img_f32, rng) -> img_f32`, listed in `_EFFECTS`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_augment_backgrounds.py` with the header block above, then:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n monitor3d pytest tests/test_augment_backgrounds.py -v`
Expected: FAIL — `FileNotFoundError` (module doesn't exist).

- [ ] **Step 3: Write the module**

Create `tools/augment_backgrounds.py`:

```python
"""Photometric LIGHTING augmentation for background crops (no labels).

Copies each background crop and writes N variants composing 2-4 random
effects — gamma, contrast/brightness, color temperature, directional
gradient, soft shadow bands, sensor noise — plus horizontal flip at p=0.5.
Geometry is untouched (the platform's position is stable; the crops carry
no labels, so flips are safe). Effects deliberately cover what YOLO's
online HSV jitter does NOT: shadows, gradients, temperature, gamma crush.

Variant names keep the capture-series prefix that make_crop_dataset's
bg_split() hashes: `<series>_0007.jpg` -> `<series>_1007.jpg` (v1),
`<series>_2007.jpg` (v2)... so every variant lands in the SAME train/val
split as its source series.

Usage:
  conda activate monitor3d
  python tools/augment_backgrounds.py \
      --src trainer/isidet/data/grouped_backgrounds \
      --out trainer/isidet/data/grouped_backgrounds_aug \
      [--variants 4] [--seed 0]

Then rebuild the crop dataset pointing --backgrounds at the _aug folder.
"""
from __future__ import annotations

import argparse
import logging
import re
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("augment_backgrounds")


def variant_name(name: str, v: int) -> str:
    """`<series>_0007.jpg` -> v=1 `<series>_1007.jpg` (same bg_split group)."""
    m = re.match(r"^(.*)_(\d+)\.[A-Za-z]+$", name)
    if m:
        stem, num = m.groups()
        return f"{stem}_{v * 1000 + int(num):04d}.jpg"
    return f"{Path(name).stem}_{v * 1000:04d}.jpg"


# ---- effects: (float32 BGR image, rng) -> float32 image ----

def _gamma(img, rng):
    g = rng.uniform(0.55, 1.6)
    return 255.0 * np.power(np.clip(img, 0, 255) / 255.0, g)


def _contrast_brightness(img, rng):
    return img * rng.uniform(0.7, 1.3) + rng.uniform(-25.0, 25.0)


def _color_temperature(img, rng):
    s = rng.uniform(0.02, 0.12) * (1.0 if rng.random() < 0.5 else -1.0)
    out = img.copy()
    out[..., 0] *= 1.0 + s      # blue up = cooler (or down = warmer)
    out[..., 2] *= 1.0 - s
    return out


def _gradient(img, rng):
    h, w = img.shape[:2]
    low = rng.uniform(0.6, 1.0)
    ang = rng.uniform(0.0, 2.0 * np.pi)
    yy, xx = np.mgrid[0:h, 0:w]
    t = xx * np.cos(ang) + yy * np.sin(ang)
    t = (t - t.min()) / max(float(t.max() - t.min()), 1e-9)
    ramp = (low + (1.0 - low) * t).astype(np.float32)
    return img * ramp[..., None]


def _shadows(img, rng):
    h, w = img.shape[:2]
    mask = np.ones((h, w), np.float32)
    for _ in range(int(rng.integers(1, 3))):
        depth = rng.uniform(0.45, 0.75)
        pts = np.stack([rng.uniform(0, w, 4), rng.uniform(0, h, 4)], axis=1)
        m = np.zeros((h, w), np.float32)
        cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1.0)
        m = cv2.GaussianBlur(m, (31, 31), 0)
        mask *= 1.0 - m * (1.0 - depth)
    return img * mask[..., None]


def _noise(img, rng):
    sigma = rng.uniform(2.0, 6.0)
    return img + rng.normal(0.0, sigma, img.shape).astype(np.float32)


_EFFECTS = (_gamma, _contrast_brightness, _color_temperature,
            _gradient, _shadows, _noise)


def augment_image(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Compose 2-4 random effects + hflip p=0.5; uint8 in, uint8 out."""
    out = img.astype(np.float32)
    k = int(rng.integers(2, 5))
    for i in rng.choice(len(_EFFECTS), size=k, replace=False):
        out = _EFFECTS[int(i)](out, rng)
    if rng.random() < 0.5:
        out = out[:, ::-1, :]
    return np.ascontiguousarray(np.clip(out, 0, 255).astype(np.uint8))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n monitor3d pytest tests/test_augment_backgrounds.py -v`
Expected: 5 PASS, pristine output.

- [ ] **Step 5: Commit**

```bash
git add tools/augment_backgrounds.py tests/test_augment_backgrounds.py
git commit --no-verify -m "feat(tools): bg augmentation — lighting effects, variant naming"
```

---

### Task 2: CLI `main()` — copy originals, write variants, refusal, summary

**Files:**
- Modify: `tools/augment_backgrounds.py` (append)
- Test: `tests/test_augment_backgrounds.py` (append)

**Interfaces:**
- Consumes: `augment_image`, `variant_name` from Task 1.
- Produces: `main(argv: list[str] | None = None) -> int` (0 ok, 2 refusal), `if __name__ == "__main__": sys.exit(main())`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_augment_backgrounds.py`:

```python
def _write_src(tmp_path, names):
    src = tmp_path / "src"
    src.mkdir()
    for n in names:
        cv2.imwrite(str(src / n), _structured_img())
    return src


def test_main_copies_originals_and_writes_variants(tmp_path):
    src = _write_src(tmp_path, ["a_0000.jpg", "a_0001.jpg"])
    out = tmp_path / "out"
    rc = ab.main(["--src", str(src), "--out", str(out), "--variants", "3"])
    assert rc == 0
    files = sorted(p.name for p in out.glob("*.jpg"))
    assert files == ["a_0000.jpg", "a_0001.jpg",
                     "a_1000.jpg", "a_1001.jpg",
                     "a_2000.jpg", "a_2001.jpg",
                     "a_3000.jpg", "a_3001.jpg"]
    # originals byte-identical, variants not identical to source
    assert (out / "a_0000.jpg").read_bytes() == (src / "a_0000.jpg").read_bytes()
    assert (out / "a_1000.jpg").read_bytes() != (src / "a_0000.jpg").read_bytes()


def test_main_deterministic_by_seed(tmp_path):
    src = _write_src(tmp_path, ["a_0000.jpg"])
    o1, o2, o3 = tmp_path / "o1", tmp_path / "o2", tmp_path / "o3"
    assert ab.main(["--src", str(src), "--out", str(o1), "--seed", "7"]) == 0
    assert ab.main(["--src", str(src), "--out", str(o2), "--seed", "7"]) == 0
    assert ab.main(["--src", str(src), "--out", str(o3), "--seed", "8"]) == 0
    assert (o1 / "a_1000.jpg").read_bytes() == (o2 / "a_1000.jpg").read_bytes()
    assert (o1 / "a_1000.jpg").read_bytes() != (o3 / "a_1000.jpg").read_bytes()


def test_main_refuses_existing_out(tmp_path):
    src = _write_src(tmp_path, ["a_0000.jpg"])
    out = tmp_path / "out"
    out.mkdir()
    assert ab.main(["--src", str(src), "--out", str(out)]) == 2


def test_main_skips_unreadable(tmp_path):
    src = _write_src(tmp_path, ["a_0000.jpg"])
    (src / "broken_0001.jpg").write_bytes(b"not a jpeg")
    out = tmp_path / "out"
    rc = ab.main(["--src", str(src), "--out", str(out), "--variants", "1"])
    assert rc == 0
    names = sorted(p.name for p in out.glob("*.jpg"))
    assert names == ["a_0000.jpg", "a_1000.jpg"]     # broken file skipped entirely
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n monitor3d pytest tests/test_augment_backgrounds.py -v -k main`
Expected: FAIL with `AttributeError: main`.

- [ ] **Step 3: Implement `main()`**

Append to `tools/augment_backgrounds.py`:

```python
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Write photometric lighting variants of background "
                    "crops. See module docstring.")
    ap.add_argument("--src", required=True, help="folder of background crops")
    ap.add_argument("--out", required=True, help="output folder (must not exist)")
    ap.add_argument("--variants", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    src, out = Path(args.src), Path(args.out)
    if out.exists():
        logger.error("refusing: %s exists", out)
        return 2
    out.mkdir(parents=True)
    rng = np.random.default_rng(args.seed)

    copied = written = skipped = 0
    files = sorted(p for p in src.glob("*")
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            logger.warning("unreadable, skipped: %s", p)
            skipped += 1
            continue
        shutil.copy2(p, out / p.name)
        copied += 1
        for v in range(1, args.variants + 1):
            aug = augment_image(img, rng)
            cv2.imwrite(str(out / variant_name(p.name, v)), aug,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            written += 1
    logger.info("done: %d original(s) copied, %d variant(s) written, "
                "%d skipped -> %s", copied, written, skipped, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the whole test file, then the full suite**

Run: `conda run -n monitor3d pytest tests/test_augment_backgrounds.py -v`
Expected: 9 PASS.
Run: `conda run -n monitor3d pytest -q`
Expected: all green (1 known pre-existing PyGObject warning repo-wide).

- [ ] **Step 5: Lint**

Run: `conda run -n monitor3d ruff check tools/augment_backgrounds.py tests/test_augment_backgrounds.py`
Expected: clean (fix findings in these two files only).

- [ ] **Step 6: Commit**

```bash
git add tools/augment_backgrounds.py tests/test_augment_backgrounds.py
git commit --no-verify -m "feat(tools): bg augmentation CLI — copy originals, seeded variants, refusal"
```

---

### Manual verification (operator, not CI)

1. `python tools/augment_backgrounds.py --src trainer/isidet/data/grouped_backgrounds --out trainer/isidet/data/grouped_backgrounds_aug` → 89 originals + 356 variants = 445 files; eyeball a few variants (shadows/gradients look like plausible lighting, not artifacts).
2. `rm -rf trainer/isidet/data/pallet3_yolo_seg_crop384 && python tools/make_crop_dataset.py --src trainer/isidet/data/pallet3_yolo_seg --out trainer/isidet/data/pallet3_yolo_seg_crop384 --backgrounds trainer/isidet/data/grouped_backgrounds_aug` → summary shows ~445 backgrounds, series intact per split.
3. Train the nano at imgsz=320 on the new data.yaml.
