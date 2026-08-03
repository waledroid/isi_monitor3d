# Zone Background-Crop Capture Tool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A standalone CPU-only script `tools/capture_zone_bg.py` that saves gray polygon-filled zone crops — byte-matching what `ZoneScopedDetector` feeds the model — so they can be added to `trainer/isidet/data/pallet3_yolo_seg` as YOLO background images (hard negatives for the empty wooden pallet support).

**Architecture:** One new module in `tools/` with pure, individually-testable helpers (`scale_box`, `fill_crop`, `CropDeduper`), an injectable `capture_loop`, frame providers (shm bus preferred, RTSP software-decode fallback), and a thin `main()`. Read-only reuse of `backbone.detection.zone_scope` geometry (`zone_crop_boxes`, `zone_fill_polygons`, `_FILL_GRAY`) — no changes to isistream or the Backbone.

**Tech Stack:** Python 3.10, OpenCV (`cv2`), numpy, PyYAML, `backbone.shared.frame_shm`, `backbone.core.registry` (rtsp plugin). Tests: pytest, hermetic (no camera, no GPU).

**Spec:** `docs/superpowers/specs/2026-08-03-zone-bg-capture-design.md`

## Global Constraints

- Python 3.10 syntax only (no 3.12 `type` keyword etc.).
- Run everything inside the `monitor3d` conda env (`conda activate monitor3d`).
- CPU-only: never build a detector; RTSP fallback forces `decoder="software"` (never nvdec — the live stack may own the GPU).
- No modifications to `backbone/` or `isistream/` — the tool only imports from them.
- Crop geometry and fill must mirror `ZoneScopedDetector.detect` / `_fill_outside` exactly (`backbone/detection/zone_scope.py:453-466` and `:552-572`).
- Filenames: `{prefix}_{cam}_{zone-slug}_{NNNN}.jpg`, JPEG quality 95.
- Exit codes: `2` = no zones configured, `1` = no camera delivered frames, `0` = normal.
- `tests/test_capture_zone_bg.py` loads the module via `importlib` by file path (tools/ is not an installed package).
- Commit after every green task; end commit messages with the session trailer used in this repo.

## File Structure

- `tools/capture_zone_bg.py` — the whole tool (single focused module, ~230 lines): pure helpers → dedup → capture loop → providers → CLI.
- `tests/test_capture_zone_bg.py` — all tests for the tool.

Every test file starts with this loader (repeat it verbatim in the test file, once):

```python
"""Tests for tools/capture_zone_bg.py (hermetic — no cameras, no GPU)."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "capture_zone_bg", _ROOT / "tools" / "capture_zone_bg.py")
czb = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(czb)
```

---

### Task 1: Pure helpers — `scale_box`, `fill_crop`, `CropDeduper`

**Files:**
- Create: `tools/capture_zone_bg.py`
- Test: `tests/test_capture_zone_bg.py`

**Interfaces:**
- Produces:
  - `scale_box(box: tuple[int,int,int,int], calib_wh: tuple[int,int], frame_wh: tuple[int,int]) -> tuple[int,int,int,int,float,float]` — returns `(fx0, fy0, fx1, fy1, sx, sy)` in frame pixels.
  - `fill_crop(crop_img: np.ndarray, fill: tuple[np.ndarray, float], sx: float, sy: float, fx0: int, fy0: int) -> np.ndarray` — returns a COPY with outside-polygon pixels set to `_FILL_GRAY` (114).
  - `CropDeduper(min_diff: float = 4.0)` with `.should_save(cam_id: str, zone_name: str, crop: np.ndarray) -> bool`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_capture_zone_bg.py` with the loader block above, then:

```python
def test_scale_box_matches_zone_scoped_detector():
    # Same formula as ZoneScopedDetector.detect (zone_scope.py:456-459):
    # int() floor on the min corner, ceil on the max corner, clamped.
    box = (100, 200, 900, 1000)
    fx0, fy0, fx1, fy1, sx, sy = czb.scale_box(box, (1920, 1080), (1280, 720))
    assert sx == 1280 / 1920 and sy == 720 / 1080
    assert fx0 == max(0, int(100 * sx))
    assert fy0 == max(0, int(200 * sy))
    assert fx1 == min(1280, math.ceil(900 * sx))
    assert fy1 == min(720, math.ceil(1000 * sy))


def test_scale_box_identity_when_sizes_match():
    assert czb.scale_box((10, 20, 30, 40), (640, 480), (640, 480))[:4] == (10, 20, 30, 40)


def test_fill_crop_grays_outside_polygon_and_copies():
    img = np.full((100, 100, 3), 200, np.uint8)
    poly = np.array([[30, 30], [70, 30], [70, 70], [30, 70]], dtype=np.float64)
    out = czb.fill_crop(img, (poly, 4.0), 1.0, 1.0, 0, 0)
    assert (out[0, 0] == czb._FILL_GRAY).all()      # corner: outside → gray
    assert (out[50, 50] == 200).all()               # center: inside → preserved
    assert (out[50, 72] == 200).all()               # dilation keeps the edge band
    assert (img[0, 0] == 200).all()                 # original frame untouched


def test_fill_crop_respects_crop_origin_offset():
    # Polygon at frame px (130..170); crop starts at fx0=100, fy0=100.
    img = np.full((100, 100, 3), 200, np.uint8)
    poly = np.array([[130, 130], [170, 130], [170, 170], [130, 170]], dtype=np.float64)
    out = czb.fill_crop(img, (poly, 4.0), 1.0, 1.0, 100, 100)
    assert (out[50, 50] == 200).all()               # inside shifted polygon
    assert (out[5, 5] == czb._FILL_GRAY).all()


def test_deduper_first_always_then_threshold():
    d = czb.CropDeduper(min_diff=4.0)
    flat = np.zeros((80, 80, 3), np.uint8)
    assert d.should_save("cam_a", "z1", flat) is True     # first crop always saves
    assert d.should_save("cam_a", "z1", flat) is False    # identical → skip
    brighter = np.full((80, 80, 3), 60, np.uint8)
    assert d.should_save("cam_a", "z1", brighter) is True # big change → save
    assert d.should_save("cam_a", "z2", flat) is True     # other zone independent
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n monitor3d pytest tests/test_capture_zone_bg.py -v`
Expected: FAIL — `FileNotFoundError` (module doesn't exist) or `AttributeError: scale_box`.

- [ ] **Step 3: Write the module with the helpers**

Create `tools/capture_zone_bg.py`:

```python
"""Capture inference-identical zone crops as YOLO background images.

Saves gray polygon-filled zone crops — exactly the pixels ZoneScopedDetector
feeds the model — so an empty scene (e.g. the flat wooden pallet support) can
be added to a dataset as hard negatives.

Merge procedure (after HUMAN review of every crop):
  1. Delete any crop containing ANY instance of ANY class — an unlabeled
     object in a background image teaches the model to miss that class.
  2. Copy ~90% into <dataset>/images/train/ and ~10% into images/val/ with
     NO label files (YOLO's background convention; labels/ stays untouched).
  3. Retrain. Start ~250 backgrounds (5% of train), ceiling ~500 (10%).

Usage:
  conda activate monitor3d
  python tools/capture_zone_bg.py --config config/backbone.yaml \
      [--out trainer/isidet/data/bg_captures] [--prefix bg] [--interval 2.0] \
      [--count 300] [--min-diff 4.0] [--cams cam_a,cam_b]

Frames come from the /dev/shm frame bus when isistream is running (zero extra
RTSP session, zero GPU); otherwise the tool opens its own RTSP session with
SOFTWARE decode. Run `--prefix pos` sessions with palettes present to also
collect in-domain positives (label those via the LabelMe flow).
"""
from __future__ import annotations

import argparse
import logging
import math
import re
import signal
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from backbone.detection.zone_scope import _FILL_GRAY  # noqa: F401  (re-exported for tests)

logger = logging.getLogger("capture_zone_bg")


def scale_box(box, calib_wh, frame_wh):
    """Calibration-frame box → actual-frame box: ``(fx0, fy0, fx1, fy1, sx, sy)``.

    Same arithmetic as ``ZoneScopedDetector.detect`` (zone_scope.py) so the
    saved crop covers exactly the pixels the detector sees on a possibly
    ingest-downscaled frame.
    """
    x0, y0, x1, y1 = box
    calib_w, calib_h = calib_wh
    fw, fh = frame_wh
    sx, sy = fw / calib_w, fh / calib_h
    fx0, fy0 = max(0, int(x0 * sx)), max(0, int(y0 * sy))
    fx1, fy1 = min(fw, math.ceil(x1 * sx)), min(fh, math.ceil(y1 * sy))
    return fx0, fy0, fx1, fy1, sx, sy


def fill_crop(crop_img, fill, sx, sy, fx0, fy0):
    """Gray-out crop pixels outside the (dilated) zone polygon — a COPY.

    Mirrors ``ZoneScopedDetector._fill_outside`` (minus its cache): polygon is
    calibration-frame px, scaled by (sx, sy) and shifted by the crop origin.
    """
    poly_px, dilate_px = fill
    ch, cw = crop_img.shape[:2]
    pts = np.round(poly_px * (sx, sy) - (fx0, fy0)).astype(np.int32)
    inside = np.zeros((ch, cw), np.uint8)
    cv2.fillPoly(inside, [pts], 255)
    r = max(1, round(dilate_px * (sx + sy) / 2.0))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    inside = cv2.dilate(inside, kernel)
    out = crop_img.copy()
    out[inside == 0] = _FILL_GRAY
    return out


class CropDeduper:
    """Save a crop only when it visibly differs from the LAST SAVED one.

    Signature = 64×64 grayscale; difference = mean absolute pixel delta.
    A static empty zone then costs one file, not one per interval.
    """

    def __init__(self, min_diff: float = 4.0) -> None:
        self._min = float(min_diff)
        self._last: dict[tuple[str, str], np.ndarray] = {}

    def should_save(self, cam_id: str, zone_name: str, crop: np.ndarray) -> bool:
        sig = cv2.resize(
            cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (64, 64)).astype(np.float32)
        last = self._last.get((cam_id, zone_name))
        if last is not None and float(np.abs(sig - last).mean()) < self._min:
            return False
        self._last[(cam_id, zone_name)] = sig
        return True
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n monitor3d pytest tests/test_capture_zone_bg.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/capture_zone_bg.py tests/test_capture_zone_bg.py
git commit -m "feat(tools): zone bg capture — crop scaling, polygon fill, dedup helpers"
```

---

### Task 2: `capture_loop` — crop, fill, dedup, save, stop conditions

**Files:**
- Modify: `tools/capture_zone_bg.py` (append after `CropDeduper`)
- Test: `tests/test_capture_zone_bg.py` (append)

**Interfaces:**
- Consumes: `scale_box`, `fill_crop`, `CropDeduper` from Task 1.
- Produces: `capture_loop(providers, boxes, fill_polys, calib_wh, out_dir, *, prefix="bg", interval_s=2.0, count=300, min_diff=4.0, max_idle_polls=60, stop=None, sleep=time.sleep) -> dict[str, int]` where `providers: dict[str, Callable[[], np.ndarray | None]]`, `boxes`/`fill_polys`/`calib_wh` are the `zone_crop_boxes` / `zone_fill_polygons` / `{cam: image_size_wh}` shapes, and the return is a `{"cam/zone": n_saved}` tally. Also `_slug(name: str) -> str`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_capture_zone_bg.py`:

```python
def _frames_provider(frames):
    it = iter(frames)
    return lambda: next(it, None)


def test_capture_loop_saves_dedups_names_and_stops_when_idle(tmp_path):
    frames = [np.full((720, 1280, 3), v, np.uint8) for v in (10, 10, 200)]
    boxes = {"cam_a": [("Zone 1", (100, 100, 600, 600))]}
    tally = czb.capture_loop(
        {"cam_a": _frames_provider(frames)}, boxes, {"cam_a": {}},
        {"cam_a": (1280, 720)}, tmp_path,
        interval_s=0, count=10, min_diff=4.0, max_idle_polls=3)
    files = sorted(p.name for p in tmp_path.glob("*.jpg"))
    assert files == ["bg_cam_a_Zone-1_0000.jpg", "bg_cam_a_Zone-1_0001.jpg"]
    assert tally == {"cam_a/Zone 1": 2}          # middle frame dedup-skipped


def test_capture_loop_applies_fill(tmp_path):
    frames = [np.full((720, 1280, 3), 200, np.uint8)]
    boxes = {"cam_a": [("z", (0, 0, 1280, 720))]}
    poly = np.array([[300, 300], [900, 300], [900, 600], [300, 600]], dtype=np.float64)
    fills = {"cam_a": {"z": (poly, 4.0)}}
    czb.capture_loop(
        {"cam_a": _frames_provider(frames)}, boxes, fills,
        {"cam_a": (1280, 720)}, tmp_path,
        interval_s=0, count=1, max_idle_polls=2)
    img = cv2.imread(str(next(tmp_path.glob("*.jpg"))))
    assert abs(int(img[5, 5, 0]) - czb._FILL_GRAY) <= 3      # outside → gray (± JPEG)
    assert int(img[450, 640, 0]) > 180                        # inside preserved


def test_capture_loop_stops_at_count(tmp_path):
    frames = [np.full((720, 1280, 3), v, np.uint8) for v in (10, 200, 90, 250)]
    boxes = {"cam_a": [("z", (100, 100, 600, 600))]}
    tally = czb.capture_loop(
        {"cam_a": _frames_provider(frames)}, boxes, {"cam_a": {}},
        {"cam_a": (1280, 720)}, tmp_path,
        interval_s=0, count=2, max_idle_polls=3)
    assert sum(tally.values()) == 2


def test_slug_sanitizes_zone_names():
    assert czb._slug("Zone 1") == "Zone-1"
    assert czb._slug("étagère/2") == "tag-re-2"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n monitor3d pytest tests/test_capture_zone_bg.py -v -k "capture_loop or slug"`
Expected: FAIL with `AttributeError: capture_loop`.

- [ ] **Step 3: Implement `capture_loop`**

Append to `tools/capture_zone_bg.py`:

```python
def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-") or "zone"


def capture_loop(providers, boxes, fill_polys, calib_wh, out_dir, *,
                 prefix: str = "bg", interval_s: float = 2.0, count: int = 300,
                 min_diff: float = 4.0, max_idle_polls: int = 60,
                 stop: threading.Event | None = None,
                 sleep=time.sleep) -> dict[str, int]:
    """Poll every provider each tick; save deduped filled crops until ``count``
    images exist, ``stop`` is set, or ``max_idle_polls`` consecutive ticks
    yield no frame from any camera. Returns a ``{"cam/zone": n}`` tally."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dedup = CropDeduper(min_diff)
    counters: dict[tuple[str, str], int] = {}
    tally: dict[str, int] = {}
    saved_total = 0
    idle = 0
    while saved_total < count and (stop is None or not stop.is_set()):
        got_any = False
        for cam_id, provider in providers.items():
            if saved_total >= count:
                break
            frame = provider()
            if frame is None:
                continue
            got_any = True
            fh, fw = frame.shape[:2]
            for zone_name, box in boxes.get(cam_id) or []:
                if saved_total >= count:
                    break
                fx0, fy0, fx1, fy1, sx, sy = scale_box(
                    box, calib_wh.get(cam_id, (fw, fh)), (fw, fh))
                if fx1 - fx0 < 8 or fy1 - fy0 < 8:
                    continue
                crop = frame[fy0:fy1, fx0:fx1]
                fill = (fill_polys.get(cam_id) or {}).get(zone_name)
                if fill is not None:
                    crop = fill_crop(crop, fill, sx, sy, fx0, fy0)
                if not dedup.should_save(cam_id, zone_name, crop):
                    continue
                key = (cam_id, zone_name)
                n = counters.get(key, 0)
                counters[key] = n + 1
                path = out_dir / f"{prefix}_{cam_id}_{_slug(zone_name)}_{n:04d}.jpg"
                cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                tally_key = f"{cam_id}/{zone_name}"
                tally[tally_key] = tally.get(tally_key, 0) + 1
                saved_total += 1
                logger.info("saved %s (%d/%d)", path.name, saved_total, count)
        idle = 0 if got_any else idle + 1
        if idle >= max_idle_polls:
            logger.warning("no camera delivered frames for %d polls — stopping", idle)
            break
        if interval_s:
            sleep(interval_s)
    return tally
```

- [ ] **Step 4: Run the whole test file**

Run: `conda run -n monitor3d pytest tests/test_capture_zone_bg.py -v`
Expected: 9 PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/capture_zone_bg.py tests/test_capture_zone_bg.py
git commit -m "feat(tools): zone bg capture loop — dedup saves, count/idle/stop conditions"
```

---

### Task 3: Frame providers — shm bus preferred, RTSP software-decode fallback

**Files:**
- Modify: `tools/capture_zone_bg.py` (append)
- Test: `tests/test_capture_zone_bg.py` (append)

**Interfaces:**
- Consumes: `backbone.shared.frame_shm.FrameShmReader(camera_id, directory=None)` → `.latest() -> (ndarray, ts) | None`; `FrameShmWriter(camera_id, directory).write(image, capture_ts)` (tests only); `backbone.core.registry.frame_source_registry.create("rtsp", camera_id=…, url=…, decoder="software", …)` → `FrameSource` with `.frames()` iterator and `.stop()`.
- Produces:
  - `BusProvider(camera_id: str, directory: str | None = None)` — callable, returns latest BGR frame or `None`.
  - `RtspProvider(camera_id: str, source_cfg: dict)` — callable, `.stop()`; pumps `frames()` in a daemon thread, forces `decoder="software"`.
  - `make_provider(cam_id: str, source_cfg: dict, *, bus_wait_s=5.0, frame_wait_s=15.0, directory=None, poll_s=0.25) -> callable | None`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_capture_zone_bg.py`:

```python
def test_bus_provider_roundtrip(tmp_path):
    from backbone.shared.frame_shm import FrameShmWriter
    img = np.full((48, 64, 3), 37, np.uint8)
    writer = FrameShmWriter("camx", directory=str(tmp_path))
    try:
        writer.write(img, time.time())
        provider = czb.BusProvider("camx", directory=str(tmp_path))
        got = provider()
        assert got is not None and got.shape == (48, 64, 3) and (got == 37).all()
    finally:
        writer.close()


def test_bus_provider_none_when_bus_absent(tmp_path):
    assert czb.BusProvider("ghost", directory=str(tmp_path))() is None


def test_make_provider_prefers_bus(tmp_path):
    from backbone.shared.frame_shm import FrameShmWriter
    writer = FrameShmWriter("camy", directory=str(tmp_path))
    try:
        writer.write(np.zeros((8, 8, 3), np.uint8), time.time())
        p = czb.make_provider("camy", {"name": "rtsp", "url": "rtsp://x"},
                              bus_wait_s=1.0, directory=str(tmp_path))
        assert isinstance(p, czb.BusProvider)
    finally:
        writer.close()


def test_make_provider_none_without_bus_or_rtsp(tmp_path):
    p = czb.make_provider("ghost", {}, bus_wait_s=0.2, directory=str(tmp_path))
    assert p is None
```

Add `import time` to the test file's imports.

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n monitor3d pytest tests/test_capture_zone_bg.py -v -k "provider"`
Expected: FAIL with `AttributeError: BusProvider`.

- [ ] **Step 3: Implement the providers**

Append to `tools/capture_zone_bg.py`:

```python
class BusProvider:
    """Latest frame from the /dev/shm bus (isistream running) — zero RTSP."""

    def __init__(self, camera_id: str, directory: str | None = None) -> None:
        from backbone.shared.frame_shm import FrameShmReader
        self.camera_id = camera_id
        self._reader = FrameShmReader(camera_id, directory=directory)

    def __call__(self):
        got = self._reader.latest()
        return None if got is None else got[0]

    def stop(self) -> None:
        pass


class RtspProvider:
    """Own RTSP session (SOFTWARE decode — never touch the GPU) pumping the
    newest frame into a slot; used only when the frame bus is absent/stale."""

    def __init__(self, camera_id: str, source_cfg: dict) -> None:
        import backbone.ingestion  # noqa: F401  auto-registration fires @register
        from backbone.core.registry import frame_source_registry
        kwargs = {k: source_cfg[k]
                  for k in ("latency_ms", "capture_fps", "output_wh")
                  if source_cfg.get(k) is not None}
        self.camera_id = camera_id
        self._src = frame_source_registry.create(
            "rtsp", camera_id=camera_id, url=source_cfg["url"],
            decoder="software", **kwargs)
        self._latest = None
        self._lock = threading.Lock()
        threading.Thread(target=self._pump, daemon=True,
                         name=f"rtsp-pump-{camera_id}").start()

    def _pump(self) -> None:
        try:
            for frame in self._src.frames():
                with self._lock:
                    self._latest = frame.image
        except Exception:
            logger.exception("%s: RTSP pump died", self.camera_id)

    def __call__(self):
        with self._lock:
            return self._latest

    def stop(self) -> None:
        self._src.stop()


def make_provider(cam_id: str, source_cfg: dict, *, bus_wait_s: float = 5.0,
                  frame_wait_s: float = 15.0, directory: str | None = None,
                  poll_s: float = 0.25):
    """Bus if it delivers within ``bus_wait_s``; else RTSP fallback (when the
    camera's config source is rtsp) if IT delivers within ``frame_wait_s``;
    else ``None`` (caller skips the camera)."""
    bus = BusProvider(cam_id, directory=directory)
    deadline = time.monotonic() + bus_wait_s
    while time.monotonic() < deadline:
        if bus() is not None:
            logger.info("%s: using /dev/shm frame bus", cam_id)
            return bus
        time.sleep(poll_s)
    source_cfg = source_cfg or {}
    if source_cfg.get("name") == "rtsp" and source_cfg.get("url"):
        logger.info("%s: bus absent — opening RTSP (software decode)", cam_id)
        try:
            rtsp = RtspProvider(cam_id, source_cfg)
        except Exception:
            logger.exception("%s: RTSP fallback failed to build", cam_id)
            return None
        deadline = time.monotonic() + frame_wait_s
        while time.monotonic() < deadline:
            if rtsp() is not None:
                return rtsp
            time.sleep(poll_s)
        rtsp.stop()
        logger.warning("%s: RTSP delivered no frame in %.0fs", cam_id, frame_wait_s)
    return None
```

- [ ] **Step 4: Run the whole test file**

Run: `conda run -n monitor3d pytest tests/test_capture_zone_bg.py -v`
Expected: 13 PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/capture_zone_bg.py tests/test_capture_zone_bg.py
git commit -m "feat(tools): zone bg capture providers — shm bus preferred, RTSP software fallback"
```

---

### Task 4: CLI `main()` — config load, geometry, refusals, wiring

**Files:**
- Modify: `tools/capture_zone_bg.py` (append)
- Test: `tests/test_capture_zone_bg.py` (append)

**Interfaces:**
- Consumes: everything above; `CameraRig.from_file(path)`, `ZoneRegistry.load(path)` / `.empty()` (`len(zones)` supported); `zone_crop_boxes(rig, zones, crop_height_m=…)`, `zone_fill_polygons(rig, zones, crop_height_m=…)` from `backbone.detection.zone_scope`.
- Produces: `main(argv: list[str] | None = None) -> int` (exit codes 0/1/2), `if __name__ == "__main__": sys.exit(main())`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_capture_zone_bg.py`:

```python
def test_main_refuses_without_zones(tmp_path, monkeypatch):
    cfg = tmp_path / "backbone.yaml"
    cfg.write_text("calibration_path: /nonexistent.json\n")   # no zones_path
    monkeypatch.setattr(czb.CameraRig, "from_file",
                        staticmethod(lambda p: object()))
    rc = czb.main(["--config", str(cfg), "--out", str(tmp_path / "o")])
    assert rc == 2


def test_main_exits_1_when_no_camera_delivers(tmp_path, monkeypatch):
    cfg = tmp_path / "backbone.yaml"
    cfg.write_text("calibration_path: /nonexistent.json\nzones_path: z.yaml\n")

    class _FakeView:
        image_size_wh = (1920, 1080)

    class _FakeRig:
        camera_ids = ["cam_a"]
        def __getitem__(self, k):
            return _FakeView()

    class _FakeZones:
        def __len__(self):
            return 1

    monkeypatch.setattr(czb.CameraRig, "from_file",
                        staticmethod(lambda p: _FakeRig()))
    monkeypatch.setattr(czb.ZoneRegistry, "load",
                        staticmethod(lambda p: _FakeZones()))
    monkeypatch.setattr(czb, "zone_crop_boxes",
                        lambda rig, zones, crop_height_m: {"cam_a": [("z", (0, 0, 100, 100))]})
    monkeypatch.setattr(czb, "zone_fill_polygons",
                        lambda rig, zones, crop_height_m: {"cam_a": {}})
    monkeypatch.setattr(czb, "make_provider", lambda *a, **k: None)
    rc = czb.main(["--config", str(cfg), "--out", str(tmp_path / "o")])
    assert rc == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n monitor3d pytest tests/test_capture_zone_bg.py -v -k "main"`
Expected: FAIL with `AttributeError: main` (or `CameraRig`).

- [ ] **Step 3: Implement `main()`**

Append to `tools/capture_zone_bg.py` (note: the module-level imports `CameraRig`, `ZoneRegistry`, `zone_crop_boxes`, `zone_fill_polygons`, `yaml` are added HERE, at the top of the file with the other imports — tests monkeypatch them as module attributes):

At the top of the file, extend the imports:

```python
import yaml

from backbone.detection.zone_scope import (  # replaces the Task-1 single import
    _FILL_GRAY,
    zone_crop_boxes,
    zone_fill_polygons,
)
from backbone.shared.camera_rig import CameraRig
from backbone.shared.zones import ZoneRegistry
```

At the bottom of the file:

```python
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Save inference-identical (gray-filled) zone crops as "
                    "YOLO background images. See module docstring for the "
                    "dataset merge procedure.")
    ap.add_argument("--config", required=True, help="backbone.yaml path")
    ap.add_argument("--out", default="trainer/isidet/data/bg_captures")
    ap.add_argument("--prefix", default="bg",
                    help="filename prefix; use 'pos' for occupied-zone sessions")
    ap.add_argument("--interval", type=float, default=2.0, help="seconds between polls")
    ap.add_argument("--count", type=int, default=300, help="stop after N saved images")
    ap.add_argument("--min-diff", type=float, default=4.0,
                    help="mean abs gray delta vs last saved crop to count as new")
    ap.add_argument("--cams", default=None, help="comma list; default: all in rig")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    cfg = yaml.safe_load(Path(args.config).read_text()) or {}
    rig = CameraRig.from_file(cfg["calibration_path"])
    zones_path = cfg.get("zones_path")
    zones = ZoneRegistry.load(zones_path) if zones_path else ZoneRegistry.empty()
    if len(zones) == 0:
        logger.error("no zones configured (%s) — nothing to crop; draw zones first",
                     zones_path or "no zones_path in config")
        return 2

    det = cfg.get("detection", {}) or {}
    crop_h = float(det.get("zone_crop_height_m", 0.0) or 0.0)
    boxes = zone_crop_boxes(rig, zones, crop_height_m=crop_h)
    fills = zone_fill_polygons(rig, zones, crop_height_m=crop_h)
    calib_wh = {cid: rig[cid].image_size_wh for cid in rig.camera_ids}

    cams = ([c.strip() for c in args.cams.split(",") if c.strip()]
            if args.cams else list(rig.camera_ids))
    providers = {}
    for cam_id in cams:
        source_cfg = ((cfg.get("cameras", {}).get(cam_id) or {}).get("source")) or {}
        provider = make_provider(cam_id, source_cfg)
        if provider is not None:
            providers[cam_id] = provider
        else:
            logger.warning("skipping %s: no frames from bus or RTSP", cam_id)
    if not providers:
        logger.error("no camera delivered frames — is the system (or a camera) up?")
        return 1

    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())

    try:
        tally = capture_loop(
            providers, boxes, fills, calib_wh, Path(args.out),
            prefix=args.prefix, interval_s=args.interval, count=args.count,
            min_diff=args.min_diff, stop=stop)
    finally:
        for provider in providers.values():
            provider.stop()

    total = sum(tally.values())
    logger.info("done: %d image(s) in %s", total, args.out)
    for key in sorted(tally):
        logger.info("  %-30s %d", key, tally[key])
    logger.info("next: review every crop (delete any containing an object), then "
                "copy ~90%% to images/train and ~10%% to images/val with NO label files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the whole test file, then the full suite**

Run: `conda run -n monitor3d pytest tests/test_capture_zone_bg.py -v`
Expected: 15 PASS.
Run: `conda run -n monitor3d pytest -q`
Expected: everything green (the tool imports `backbone.*` read-only; nothing else changed).

- [ ] **Step 5: Lint**

Run: `conda run -n monitor3d ruff check tools/capture_zone_bg.py tests/test_capture_zone_bg.py`
Expected: clean (fix any findings).

- [ ] **Step 6: Commit**

```bash
git add tools/capture_zone_bg.py tests/test_capture_zone_bg.py
git commit -m "feat(tools): zone bg capture CLI — config wiring, refusals, tally"
```

---

### Manual verification (operator, not CI)

1. With isistream running: `python tools/capture_zone_bg.py --config config/backbone.yaml --count 20` → files appear in `trainer/isidet/data/bg_captures/`, log says "using /dev/shm frame bus", GPU untouched.
2. With the stack stopped: same command → log says "opening RTSP (software decode)".
3. Eyeball crops: gray outside the zone polygon, support fully inside, matches the dashboard zone panel's framing.
