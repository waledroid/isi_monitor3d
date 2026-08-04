# Object-Centric Crop Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A standalone script `tools/make_crop_dataset.py` that derives an object-centric 384×384 cropped YOLO-seg dataset (`pallet3_yolo_seg_crop384`) from `pallet3_yolo_seg` using ground-truth labels for crop placement, with polygon remapping, a keep-frac/gray-fill rule, folded-in backgrounds, and a preview mode — leaving the source dataset untouched.

**Architecture:** One module of pure, individually-testable geometry functions (label parse/write, Sutherland–Hodgman polygon clip, shoelace area, union-find clustering, crop-window computation, no-upscale letterbox), a pure per-image `generate_crops()` composing them, and a thin CLI (`main()`) that walks the dataset splits, writes crops + remapped labels + `data.yaml`, folds in backgrounds, and offers `--preview N`.

**Tech Stack:** Python 3.10, numpy, OpenCV (`cv2`), PyYAML, pytest (hermetic — synthetic images only).

**Spec:** `docs/superpowers/specs/2026-08-04-crop-dataset-design.md`

## Global Constraints

- Python 3.10 syntax only.
- Run tests inside the monitor3d conda env: `conda run -n monitor3d pytest tests/test_make_crop_dataset.py -v`.
- The source dataset is read-only — the script never writes inside `--src`.
- Crop size: `--size` default **384**; gray value **114** everywhere (fill + letterbox pad).
- Defaults: `--margin 0.10 0.25`, `--keep-frac 0.30`, `--seed 0`.
- Never upscale image content: letterbox scale = `min(size/w, size/h, 1.0)`.
- Split preservation: crops from `images/train` go only to `train`, `val` only to `val`.
- Output naming: `images/<split>/<stem>_c<k>.jpg` (JPEG quality 95) + matching `labels/<split>/<stem>_c<k>.txt`; label lines `cls x1 y1 x2 y2 ...` normalized to 6 decimals, clamped to [0,1].
- Refusal: `--out` existing with any content other than a `_preview` subdir → exit 2.
- Backgrounds: 90/10 train/val, deterministic by filename hash (`md5 % 10 == 0` → val), letterboxed, NO label files.
- Tests load the module via importlib by file path (tools/ is not an installed package).
- Commit after every green task with `git commit --no-verify`, adding ONLY that task's two files; end every commit message with exactly:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_01CF51vq87B9hoGAkRkzWjf7`

## File Structure

- `tools/make_crop_dataset.py` — the whole tool (~320 lines): label I/O → polygon geometry → clustering/window/letterbox → `generate_crops` → CLI.
- `tests/test_make_crop_dataset.py` — all tests, hermetic.

The test file starts with this loader (verbatim, once, module scope):

```python
"""Tests for tools/make_crop_dataset.py (hermetic — synthetic images only)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "make_crop_dataset", _ROOT / "tools" / "make_crop_dataset.py")
mcd = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mcd)
```

---

### Task 1: Label I/O + polygon geometry (parse, write, bbox, area, clip)

**Files:**
- Create: `tools/make_crop_dataset.py`
- Test: `tests/test_make_crop_dataset.py`

**Interfaces:**
- Produces:
  - `parse_label_file(path: Path, img_w: int, img_h: int) -> list[tuple[int, np.ndarray]]` — YOLO-seg lines → `(cls, poly_px (N,2) float64)`; malformed lines warn + skip.
  - `format_label_lines(objs: list[tuple[int, np.ndarray]]) -> str` — polygons already normalized to [0,1]; 6 decimals, values clamped to [0,1]; one line per object, trailing newline (empty string for no objs).
  - `poly_bbox(poly: np.ndarray) -> tuple[float, float, float, float]`
  - `poly_area(poly: np.ndarray) -> float` (shoelace, absolute)
  - `clip_polygon(poly: np.ndarray, x0, y0, x1, y1) -> np.ndarray | None` — Sutherland–Hodgman vs axis-aligned rect; `None` when nothing remains.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_make_crop_dataset.py` with the loader block above, then:

```python
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
    assert out == "2 0.123457 0.000000 1.000000 0.500000 0.500000 1.000000\n"
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n monitor3d pytest tests/test_make_crop_dataset.py -v`
Expected: FAIL — `FileNotFoundError` (module doesn't exist yet).

- [ ] **Step 3: Write the module with the helpers**

Create `tools/make_crop_dataset.py`:

```python
"""Derive an object-centric cropped YOLO-seg dataset from pallet3_yolo_seg.

Crops square windows around clusters of GROUND-TRUTH objects (never a
model's detections), remaps the segmentation polygons into each crop, and
writes a new dataset sized for the isimonitor3d nano zone-inference domain.
The source dataset is never modified.

Rules (see docs/superpowers/specs/2026-08-04-crop-dataset-design.md):
- crop size --size (default 384); content is never upscaled (gray-114 pad);
- an object with < --keep-frac of its area inside a crop is NOT labeled —
  its in-crop pixels are painted gray 114 (like the inference polygon fill);
- split-preserving (train->train, val->val); backgrounds fold in 90/10.

Usage:
  conda activate monitor3d
  python tools/make_crop_dataset.py \
      --src trainer/isidet/data/pallet3_yolo_seg \
      --out trainer/isidet/data/pallet3_yolo_seg_crop384 \
      [--size 384] [--backgrounds trainer/isidet/data/grouped_backgrounds] \
      [--margin 0.10 0.25] [--keep-frac 0.30] [--seed 0] [--preview N]
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

logger = logging.getLogger("make_crop_dataset")

GRAY = 114


# ---- label I/O ----

def parse_label_file(path: Path, img_w: int, img_h: int):
    """YOLO-seg label file -> [(cls, poly_px (N,2) float64)]; bad lines skipped."""
    objs = []
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        try:
            cls = int(parts[0])
            vals = np.array([float(v) for v in parts[1:]], dtype=np.float64)
            if len(vals) < 6 or len(vals) % 2:
                raise ValueError("need >=3 x,y pairs")
        except (ValueError, IndexError) as exc:
            logger.warning("%s:%d: skipping malformed label line (%s)",
                           path, lineno, exc)
            continue
        poly = vals.reshape(-1, 2) * (img_w, img_h)
        objs.append((cls, poly))
    return objs


def format_label_lines(objs) -> str:
    """[(cls, poly_norm (N,2) in [0,1])] -> YOLO-seg text (clamped, 6dp)."""
    lines = []
    for cls, poly in objs:
        flat = np.clip(np.asarray(poly, dtype=np.float64), 0.0, 1.0).flatten()
        lines.append(str(cls) + " " + " ".join(f"{v:.6f}" for v in flat))
    return "".join(line + "\n" for line in lines)


# ---- polygon geometry ----

def poly_bbox(poly):
    return (float(poly[:, 0].min()), float(poly[:, 1].min()),
            float(poly[:, 0].max()), float(poly[:, 1].max()))


def poly_area(poly) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def clip_polygon(poly, x0, y0, x1, y1):
    """Sutherland-Hodgman clip vs axis-aligned rect; None when empty."""
    def ix_v(a, b, x):
        t = (x - a[0]) / (b[0] - a[0])
        return (x, a[1] + t * (b[1] - a[1]))

    def ix_h(a, b, y):
        t = (y - a[1]) / (b[1] - a[1])
        return (a[0] + t * (b[0] - a[0]), y)

    edges = (
        (lambda p: p[0] >= x0, lambda a, b: ix_v(a, b, x0)),
        (lambda p: p[0] <= x1, lambda a, b: ix_v(a, b, x1)),
        (lambda p: p[1] >= y0, lambda a, b: ix_h(a, b, y0)),
        (lambda p: p[1] <= y1, lambda a, b: ix_h(a, b, y1)),
    )
    pts = [tuple(p) for p in np.asarray(poly, dtype=np.float64)]
    for inside, intersect in edges:
        nxt = []
        n = len(pts)
        for i in range(n):
            a, b = pts[i], pts[(i + 1) % n]
            if inside(a):
                nxt.append(a)
                if not inside(b):
                    nxt.append(intersect(a, b))
            elif inside(b):
                nxt.append(intersect(a, b))
        pts = nxt
        if not pts:
            return None
    return np.array(pts, dtype=np.float64)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n monitor3d pytest tests/test_make_crop_dataset.py -v`
Expected: 6 PASS, pristine output.

- [ ] **Step 5: Commit**

```bash
git add tools/make_crop_dataset.py tests/test_make_crop_dataset.py
git commit --no-verify -m "feat(tools): crop dataset — label I/O and polygon geometry"
```

---

### Task 2: Clustering, crop window, letterbox

**Files:**
- Modify: `tools/make_crop_dataset.py` (append)
- Test: `tests/test_make_crop_dataset.py` (append)

**Interfaces:**
- Consumes: `poly_bbox` from Task 1.
- Produces:
  - `cluster_boxes(boxes: list[tuple[float,float,float,float]], expand_frac: float) -> list[list[int]]` — union-find groups of boxes whose expanded rects intersect; deterministic order (each group sorted, groups sorted by first index).
  - `crop_window(bbox: tuple, img_wh: tuple[int,int], size: int, margin_range: tuple[float,float], rng: np.random.Generator) -> tuple[int,int,int,int]` — square-ish window ≥ `size` px when the image allows, clamped inside the image.
  - `letterbox_to(img: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]` — `(canvas size×size BGR, scale, dx, dy)`; scale ≤ 1 relative to input only when downscaling is needed (`min(size/w, size/h, 1.0)`), gray-114 padding, content centered.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_make_crop_dataset.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n monitor3d pytest tests/test_make_crop_dataset.py -v -k "cluster or window or letterbox"`
Expected: FAIL with `AttributeError: cluster_boxes`.

- [ ] **Step 3: Implement**

Append to `tools/make_crop_dataset.py`:

```python
# ---- clustering & windows ----

def cluster_boxes(boxes, expand_frac: float):
    """Union-find over boxes whose expanded rects intersect."""
    n = len(boxes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    exp = []
    for x0, y0, x1, y1 in boxes:
        mx, my = (x1 - x0) * expand_frac, (y1 - y0) * expand_frac
        exp.append((x0 - mx, y0 - my, x1 + mx, y1 + my))
    for i in range(n):
        for j in range(i + 1, n):
            a, b = exp[i], exp[j]
            if a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]:
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return sorted((sorted(g) for g in groups.values()), key=lambda g: g[0])


def crop_window(bbox, img_wh, size, margin_range, rng):
    """Square window >= size px covering bbox+margin, clamped to the image."""
    iw, ih = img_wh
    x0, y0, x1, y1 = bbox
    bw, bh = x1 - x0, y1 - y0
    lo, hi = margin_range
    x0 -= bw * rng.uniform(lo, hi)
    x1 += bw * rng.uniform(lo, hi)
    y0 -= bh * rng.uniform(lo, hi)
    y1 += bh * rng.uniform(lo, hi)
    side = max(x1 - x0, y1 - y0, float(size))
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    wx0 = int(round(cx - side / 2.0))
    wy0 = int(round(cy - side / 2.0))
    wx1 = wx0 + int(round(side))
    wy1 = wy0 + int(round(side))
    if wx0 < 0:
        wx1 -= wx0
        wx0 = 0
    if wy0 < 0:
        wy1 -= wy0
        wy0 = 0
    if wx1 > iw:
        wx0 -= wx1 - iw
        wx1 = iw
    if wy1 > ih:
        wy0 -= wy1 - ih
        wy1 = ih
    return max(0, wx0), max(0, wy0), wx1, wy1


def letterbox_to(img, size):
    """Fit img into size x size with gray padding; NEVER upscale content."""
    h, w = img.shape[:2]
    scale = min(size / w, size / h, 1.0)
    nw, nh = round(w * scale), round(h * scale)
    if (nw, nh) != (w, h):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), GRAY, np.uint8)
    dx, dy = (size - nw) // 2, (size - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw] = img
    return canvas, float(scale), dx, dy
```

- [ ] **Step 4: Run the whole test file**

Run: `conda run -n monitor3d pytest tests/test_make_crop_dataset.py -v`
Expected: 12 PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/make_crop_dataset.py tests/test_make_crop_dataset.py
git commit --no-verify -m "feat(tools): crop dataset — clustering, crop window, no-upscale letterbox"
```

---

### Task 3: `generate_crops` — remap, keep-frac gray-fill, per-image pipeline

**Files:**
- Modify: `tools/make_crop_dataset.py` (append)
- Test: `tests/test_make_crop_dataset.py` (append)

**Interfaces:**
- Consumes: everything from Tasks 1–2.
- Produces: `generate_crops(img: np.ndarray, objs: list[tuple[int, np.ndarray]], *, size: int, margin_range: tuple[float,float], keep_frac: float, rng: np.random.Generator) -> list[tuple[np.ndarray, list[tuple[int, np.ndarray]]]]` — list of `(crop size×size BGR, [(cls, poly_norm[0,1])])`; an image with no objs yields one letterboxed full-frame crop with empty labels.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_make_crop_dataset.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n monitor3d pytest tests/test_make_crop_dataset.py -v -k "generate_crops"`
Expected: FAIL with `AttributeError: generate_crops`.

- [ ] **Step 3: Implement**

Append to `tools/make_crop_dataset.py`:

```python
# ---- per-image pipeline ----

def generate_crops(img, objs, *, size, margin_range, keep_frac, rng):
    """One (crop, labels) per GT cluster; keep-frac rule gray-fills partials.

    An image with no objects becomes a single letterboxed full-frame
    background crop (empty labels).
    """
    h, w = img.shape[:2]
    if not objs:
        canvas, _, _, _ = letterbox_to(img, size)
        return [(canvas, [])]
    boxes = [poly_bbox(p) for _, p in objs]
    out = []
    for group in cluster_boxes(boxes, expand_frac=margin_range[1]):
        gx0 = min(boxes[i][0] for i in group)
        gy0 = min(boxes[i][1] for i in group)
        gx1 = max(boxes[i][2] for i in group)
        gy1 = max(boxes[i][3] for i in group)
        wx0, wy0, wx1, wy1 = crop_window((gx0, gy0, gx1, gy1), (w, h),
                                         size, margin_range, rng)
        cw, ch = wx1 - wx0, wy1 - wy0
        if cw < 8 or ch < 8:
            continue
        crop = img[wy0:wy1, wx0:wx1].copy()
        kept, fills = [], []
        for cls, poly in objs:
            clipped = clip_polygon(poly - (wx0, wy0), 0, 0, cw, ch)
            if clipped is None or len(clipped) < 3:
                continue
            area = poly_area(clipped)
            if area < 1.0:
                continue
            if area / max(poly_area(poly), 1e-9) >= keep_frac:
                kept.append((cls, clipped))
            else:
                fills.append(clipped)
        for f in fills:
            cv2.fillPoly(crop, [np.round(f).astype(np.int32)],
                         (GRAY, GRAY, GRAY))
        canvas, scale, dx, dy = letterbox_to(crop, size)
        labels = [(cls, (p * scale + (dx, dy)) / size) for cls, p in kept]
        out.append((canvas, labels))
    return out
```

- [ ] **Step 4: Run the whole test file**

Run: `conda run -n monitor3d pytest tests/test_make_crop_dataset.py -v`
Expected: 18 PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/make_crop_dataset.py tests/test_make_crop_dataset.py
git commit --no-verify -m "feat(tools): crop dataset — generate_crops with keep-frac gray-fill remap"
```

---

### Task 4: CLI — dataset walk, backgrounds, data.yaml, refusal, preview, summary

**Files:**
- Modify: `tools/make_crop_dataset.py` (append)
- Test: `tests/test_make_crop_dataset.py` (append)

**Interfaces:**
- Consumes: everything above.
- Produces: `main(argv: list[str] | None = None) -> int` (0 ok, 2 refusal), `bg_split(name: str) -> str` (`"val"` when `int(hashlib.md5(name.encode()).hexdigest(), 16) % 10 == 0` else `"train"`), `if __name__ == "__main__": sys.exit(main())`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_make_crop_dataset.py`. The fixture builds a tiny synthetic source dataset:

```python
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


def test_main_builds_split_preserving_dataset(tmp_path):
    src = _make_src(tmp_path)
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


def test_main_refuses_existing_out(tmp_path):
    src = _make_src(tmp_path)
    out = tmp_path / "out"
    (out / "images").mkdir(parents=True)
    assert mcd.main(["--src", str(src), "--out", str(out)]) == 2


def test_main_folds_backgrounds_without_labels(tmp_path):
    src = _make_src(tmp_path)
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
    # deterministic split
    assert {p.parent.name for p in bg_imgs} == {
        mcd.bg_split(p.name) for p in bg_imgs} or True
    for p in bg_imgs:
        assert p.parent.name == mcd.bg_split(p.name)


def test_bg_split_deterministic():
    assert mcd.bg_split("a.jpg") == mcd.bg_split("a.jpg")
    assert all(mcd.bg_split(f"x{i}.jpg") in ("train", "val") for i in range(50))


def test_main_preview_writes_only_preview(tmp_path):
    src = _make_src(tmp_path)
    out = tmp_path / "out"
    rc = mcd.main(["--src", str(src), "--out", str(out), "--preview", "2"])
    assert rc == 0
    previews = list((out / "_preview").glob("*.jpg"))
    assert len(previews) == 2
    assert not (out / "images").exists() and not (out / "labels").exists()
    # a full run AFTER preview is allowed (out contains only _preview)
    assert mcd.main(["--src", str(src), "--out", str(out)]) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n monitor3d pytest tests/test_make_crop_dataset.py -v -k "main or bg_split"`
Expected: FAIL with `AttributeError: main`.

- [ ] **Step 3: Implement**

Append to `tools/make_crop_dataset.py`:

```python
# ---- CLI ----

def bg_split(name: str) -> str:
    """Deterministic 90/10 background split by filename hash."""
    return "val" if int(hashlib.md5(name.encode()).hexdigest(), 16) % 10 == 0 \
        else "train"


def _iter_images(split_dir: Path):
    return sorted(p for p in split_dir.glob("*")
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Derive an object-centric cropped YOLO-seg dataset from "
                    "GT labels. See module docstring.")
    ap.add_argument("--src", required=True, help="source dataset root")
    ap.add_argument("--out", required=True, help="output dataset root")
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--backgrounds", default=None,
                    help="folder of label-free background images to fold in")
    ap.add_argument("--margin", type=float, nargs=2, default=(0.10, 0.25),
                    metavar=("LO", "HI"))
    ap.add_argument("--keep-frac", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preview", type=int, default=0, metavar="N",
                    help="write N annotated sample crops to <out>/_preview "
                         "and exit (no dataset generated)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    src, out = Path(args.src), Path(args.out)
    if out.exists() and any(p.name != "_preview" for p in out.iterdir()):
        logger.error("refusing: %s exists and is not empty", out)
        return 2
    rng = np.random.default_rng(args.seed)

    pairs = []          # (split, img_path, label_path)
    for split in ("train", "val"):
        for img_path in _iter_images(src / "images" / split):
            pairs.append((split, img_path,
                          src / "labels" / split / (img_path.stem + ".txt")))

    if args.preview:
        pv = out / "_preview"
        pv.mkdir(parents=True, exist_ok=True)
        idxs = rng.choice(len(pairs), size=min(args.preview, len(pairs)),
                          replace=False)
        written = 0
        for i in idxs:
            split, img_path, lbl_path = pairs[int(i)]
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            objs = (parse_label_file(lbl_path, img.shape[1], img.shape[0])
                    if lbl_path.exists() else [])
            for k, (crop, labels) in enumerate(generate_crops(
                    img, objs, size=args.size, margin_range=tuple(args.margin),
                    keep_frac=args.keep_frac, rng=rng)):
                for cls, poly in labels:
                    pts = np.round(poly * args.size).astype(np.int32)
                    cv2.polylines(crop, [pts], True, (0, 255, 0), 2)
                    cv2.putText(crop, str(cls), tuple(pts[0]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imwrite(str(pv / f"{img_path.stem}_c{k}.jpg"), crop)
                written += 1
                if written >= args.preview:
                    break
            if written >= args.preview:
                break
        logger.info("preview: %d annotated crop(s) in %s", written, pv)
        return 0

    stats = {"images": 0, "unreadable": 0, "crops": 0, "labels": 0,
             "grayfilled_or_dropped": 0, "backgrounds": 0}
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    for split, img_path, lbl_path in pairs:
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning("unreadable image skipped: %s", img_path)
            stats["unreadable"] += 1
            continue
        stats["images"] += 1
        objs = (parse_label_file(lbl_path, img.shape[1], img.shape[0])
                if lbl_path.exists() else [])
        n_src = len(objs)
        n_kept = 0
        for k, (crop, labels) in enumerate(generate_crops(
                img, objs, size=args.size, margin_range=tuple(args.margin),
                keep_frac=args.keep_frac, rng=rng)):
            name = f"{img_path.stem}_c{k}"
            cv2.imwrite(str(out / "images" / split / (name + ".jpg")), crop,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            if labels:
                (out / "labels" / split / (name + ".txt")).write_text(
                    format_label_lines(labels))
            stats["crops"] += 1
            stats["labels"] += len(labels)
            n_kept += len(labels)
        stats["grayfilled_or_dropped"] += max(0, n_src - n_kept)

    if args.backgrounds:
        for p in _iter_images(Path(args.backgrounds)):
            img = cv2.imread(str(p))
            if img is None:
                logger.warning("unreadable background skipped: %s", p)
                continue
            canvas, _, _, _ = letterbox_to(img, args.size)
            cv2.imwrite(str(out / "images" / bg_split(p.name) / p.name),
                        canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
            stats["backgrounds"] += 1

    names = ["palette", "carton", "polybag"]
    src_yaml = src / "data.yaml"
    if src_yaml.exists():
        loaded = yaml.safe_load(src_yaml.read_text()) or {}
        names = list(loaded.get("names", names))
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\n"
        f"nc: {len(names)}\nnames: {names}\n")

    for split in ("train", "val"):
        n_img = len(list((out / "images" / split).glob("*.jpg")))
        n_lbl = len(list((out / "labels" / split).glob("*.txt")))
        logger.info("%s: %d images (%d labeled, %d backgrounds)",
                    split, n_img, n_lbl, n_img - n_lbl)
    logger.info("summary: %s", stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Note: the "gray-filled or dropped" stat counts source objects that produced
no label in any crop of their image — an approximation for the summary only
(an object labeled in two overlapping crops counts twice in `labels`).

- [ ] **Step 4: Run the whole test file, then the full suite**

Run: `conda run -n monitor3d pytest tests/test_make_crop_dataset.py -v`
Expected: 23 PASS.
Run: `conda run -n monitor3d pytest -q`
Expected: all green (nothing outside the two new files changed).

- [ ] **Step 5: Lint**

Run: `conda run -n monitor3d ruff check tools/make_crop_dataset.py tests/test_make_crop_dataset.py`
Expected: clean (fix findings in these two files only).

- [ ] **Step 6: Commit**

```bash
git add tools/make_crop_dataset.py tests/test_make_crop_dataset.py
git commit --no-verify -m "feat(tools): crop dataset CLI — walk, backgrounds, data.yaml, preview, refusal"
```

---

### Manual verification (operator, not CI)

1. `python tools/make_crop_dataset.py --src trainer/isidet/data/pallet3_yolo_seg --out trainer/isidet/data/pallet3_yolo_seg_crop384 --preview 24` → eyeball `_preview/`: polygons hug the pallets, partial neighbors are gray, nothing upscaled/stretched.
2. Full run with `--backgrounds trainer/isidet/data/grouped_backgrounds` (AFTER the human review of that folder) → check the printed split totals ≈ source ratios; spot-check a few train crops + labels in LabelMe or a viewer.
3. Train the nano from `trainer/isidet/` at `imgsz=320` pointing at the new `data.yaml`.
