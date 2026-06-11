# Camera-Traced 2.5D Warehouse Map — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a static, styled-2.5D warehouse layout (racks/walls/obstacles) to the operator floor map, traced from the calibrated camera's rectified bird's-eye view used as a metric-aligned underlay.

**Architecture:** Entirely consumer-side in `monitor_web` (no Backbone changes). A new `warehouse_map.yaml` stores layout elements as metric floor footprints + heights. New endpoints serve the layout and a rectified-floor snapshot with its metric bounds. The Pixi floor map (`floor_map.js`) gains an underlay sprite (opacity-slider) and a procedurally-extruded 2.5D layout layer. Authoring reuses the existing map draw-mode with a new rectangle-snap aid.

**Tech Stack:** FastAPI + Pydantic-less plain dict validation, PyYAML (atomic write), NumPy/OpenCV (`floor_rectify`), Pixi.js v8, Alpine.js, vanilla ES modules. Tests: pytest (hermetic), `node --check` for JS parse.

**Conventions:**
- Run backend tests from `monitor_web/` with the `monitor3d` env active: `cd monitor_web && pytest`.
- Lint: `ruff check monitor_web` and `node --check <file>`.
- This repo may not be git-initialized; commit steps are the convention — run `git init` in the repo root first, or skip the commit steps if you aren't using git.

---

## File Structure

**Backend (`monitor_web/monitor_web/`)**
- Create `warehouse_map.py` — load / validate / atomic-write `warehouse_map.yaml`. One responsibility: the layout data model.
- Create `api/routes_map.py` — `GET/POST /api/warehouse-map`, `GET /api/warp-snapshot/{camera_id}`.
- Modify `floor_rectify.py` — extract `fit_rectify_bounds()` (the metric placement of the rectified image) and reuse it inside `build_fit_rectify_matrix()`.
- Modify `config.py` — add `warehouse_map_path` setting.
- Modify `app.py` — register the `routes_map` router.

**Frontend (`monitor_web/monitor_web/static/`)**
- Modify `js/floor_map.js` — `underlayLayer` (rectified photo sprite + opacity) and `layoutLayer` (2.5D extrusion); fetch `/api/warehouse-map`; expose setters on `window.__floor_map`.
- Modify `js/draw_mode.js` — add `rectSnap` mode (2 corners → axis-aligned rectangle, optional grid snap) for the map target.
- Create `js/layout_manager.js` — authoring flow (pick camera, load underlay, opacity slider, type/height, trace, save).
- Modify `templates/dashboard.html` — Layout authoring panel markup + load `layout_manager.js`.
- Modify `static/css/dashboard.css` — Layout panel + opacity slider styles.
- Modify `static/i18n/en.json` + `fr.json` — Layout UI strings.

**Tests (`monitor_web/tests/`)**
- Modify `test_floor_rectify.py` — `fit_rectify_bounds` inverts a known point.
- Create `test_warehouse_map.py` — loader round-trip + validation.
- Create `test_routes_map.py` — `/api/warehouse-map` round-trip, `/api/warp-snapshot` uncalibrated 404 + calibrated bounds.

---

## Task 1: Expose metric bounds from the rectifier

**Files:**
- Modify: `monitor_web/monitor_web/floor_rectify.py`
- Test: `monitor_web/tests/test_floor_rectify.py`

- [ ] **Step 1: Write the failing test**

Add to `monitor_web/tests/test_floor_rectify.py`:

```python
def test_fit_rectify_bounds_inverts_a_known_point() -> None:
    """The bounds (px_per_m, x_min, y_min) let a rectified pixel be mapped back to
    world metres: X = x_min + u/px, Y = y_min + v/px. Round-tripping a world point
    through the rectify matrix and back via the bounds must return the original."""
    from monitor_web.floor_rectify import build_fit_rectify_matrix, fit_rectify_bounds
    cam = _cam_wide()
    b = fit_rectify_bounds(cam.H_np(), (1920, 1080))
    assert b is not None
    M, out_wh = build_fit_rectify_matrix(cam.H_np(), (1920, 1080))
    assert out_wh == b["out_wh"]
    # a source pixel inside the frame → rectified pixel via M → world via bounds
    src = np.array([[[960.0, 540.0]]], np.float64)        # image centre
    u, v = cv2.perspectiveTransform(src, M)[0][0]
    X = b["x_min"] + u / b["px_per_m"]
    Y = b["y_min"] + v / b["px_per_m"]
    # …must equal the floor point H maps that source pixel to
    wx = cam.H_np() @ np.array([960.0, 540.0, 1.0])
    assert abs(X - wx[0] / wx[2]) < 1e-3
    assert abs(Y - wx[1] / wx[2]) < 1e-3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd monitor_web && pytest tests/test_floor_rectify.py::test_fit_rectify_bounds_inverts_a_known_point -v`
Expected: FAIL with `ImportError: cannot import name 'fit_rectify_bounds'`

- [ ] **Step 3: Refactor `build_fit_rectify_matrix` to extract the bounds**

In `monitor_web/monitor_web/floor_rectify.py`, add `fit_rectify_bounds` and rewrite `build_fit_rectify_matrix` to use it (behaviour unchanged):

```python
def fit_rectify_bounds(
    H: np.ndarray,
    source_wh: tuple[int, int],
    *,
    max_dim: int = 820,
    max_extent_m: float = 30.0,
) -> dict | None:
    """Metric placement of the auto-fit bird's-eye image for this camera.

    Returns ``{"px_per_m", "x_min", "y_min", "out_wh"}`` such that rectified pixel
    ``(u, v)`` ↔ world metres ``X = x_min + u/px_per_m``, ``Y = y_min + v/px_per_m``.
    Returns ``None`` when most of the frame is beyond the floor horizon (degenerate).
    """
    H = np.asarray(H, dtype=np.float64)
    w, h = source_wh
    img_corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=np.float64).T
    wc = H @ img_corners
    depth = wc[2]
    valid = depth > 1e-6
    if int(valid.sum()) < 3:
        return None
    xy = (wc[:2, valid] / depth[valid]).T
    cx, cy = float(np.median(xy[:, 0])), float(np.median(xy[:, 1]))
    half = max_extent_m / 2.0
    xs = np.clip(xy[:, 0], cx - half, cx + half)
    ys = np.clip(xy[:, 1], cy - half, cy + half)
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    ext_x = max(x_max - x_min, 1e-3)
    ext_y = max(y_max - y_min, 1e-3)
    px_per_m = max_dim / max(ext_x, ext_y)
    out_w = max(1, min(max_dim, round(ext_x * px_per_m)))
    out_h = max(1, min(max_dim, round(ext_y * px_per_m)))
    return {"px_per_m": px_per_m, "x_min": x_min, "y_min": y_min, "out_wh": (out_w, out_h)}
```

Then replace the body of `build_fit_rectify_matrix` (keep its signature + docstring) with:

```python
    b = fit_rectify_bounds(H, source_wh, max_dim=max_dim, max_extent_m=max_extent_m)
    if b is None:
        center = floor_world_center(H, source_wh)
        return build_rectify_matrix(H, _DEFAULT_PX_PER_M, _DEFAULT_OUT_WH, center), _DEFAULT_OUT_WH
    px_per_m = b["px_per_m"]
    x_min, y_min = b["x_min"], b["y_min"]
    out_w, out_h = b["out_wh"]
    S = np.array(
        [[px_per_m, 0.0, -px_per_m * x_min],
         [0.0, px_per_m, -px_per_m * y_min],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return S @ np.asarray(H, dtype=np.float64), (out_w, out_h)
```

- [ ] **Step 4: Run tests to verify pass (incl. existing)**

Run: `cd monitor_web && pytest tests/test_floor_rectify.py -v`
Expected: PASS — the new test **and** all existing `test_floor_rectify.py` tests (the matrix output is byte-identical to before).

- [ ] **Step 5: Commit**

```bash
git add monitor_web/monitor_web/floor_rectify.py monitor_web/tests/test_floor_rectify.py
git commit -m "feat(map): expose fit_rectify_bounds for metric image placement"
```

---

## Task 2: Warehouse-map data model

**Files:**
- Create: `monitor_web/monitor_web/warehouse_map.py`
- Test: `monitor_web/tests/test_warehouse_map.py`

- [ ] **Step 1: Write the failing test**

Create `monitor_web/tests/test_warehouse_map.py`:

```python
"""warehouse_map — load / validate / write the layout YAML."""
from __future__ import annotations

import pytest

from monitor_web.warehouse_map import read_map, validate_map, write_map


def test_validate_accepts_a_rack():
    data = {"elements": [{
        "id": "rack_a1", "type": "rack", "shape": "rectangle",
        "footprint": [[2.1, 0.4], [3.6, 0.4], [3.6, 1.2], [2.1, 1.2]],
        "height_m": 2.5, "label": "Rack A1",
    }]}
    out = validate_map(data)
    assert out["elements"][0]["type"] == "rack"
    assert out["outline"] is None


def test_validate_rejects_bad_type():
    with pytest.raises(ValueError, match="type"):
        validate_map({"elements": [{"id": "x", "type": "spaceship",
                                    "footprint": [[0, 0], [1, 0], [1, 1]], "height_m": 1}]})


def test_validate_rejects_short_footprint():
    with pytest.raises(ValueError, match="footprint"):
        validate_map({"elements": [{"id": "x", "type": "wall",
                                    "footprint": [[0, 0], [1, 0]], "height_m": 1}]})


def test_round_trip(tmp_path):
    p = tmp_path / "warehouse_map.yaml"
    data = {"elements": [{"id": "w1", "type": "wall", "shape": "rectangle",
                          "footprint": [[0, 0], [6, 0], [6, 0.2], [0, 0.2]],
                          "height_m": 3.0, "label": ""}],
            "outline": {"footprint": [[0, 0], [12, 0], [12, 8], [0, 8]]}}
    write_map(p, data)
    loaded = read_map(p)
    assert loaded["elements"][0]["id"] == "w1"
    assert loaded["outline"]["footprint"][2] == [12, 8]


def test_read_missing_file_returns_empty(tmp_path):
    assert read_map(tmp_path / "nope.yaml") == {"elements": [], "outline": None}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd monitor_web && pytest tests/test_warehouse_map.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'monitor_web.warehouse_map'`

- [ ] **Step 3: Implement `warehouse_map.py`**

Create `monitor_web/monitor_web/warehouse_map.py`:

```python
"""Warehouse layout twin — load / validate / write ``warehouse_map.yaml``.

A consumer-side config (sibling to ``zones.yaml``) describing the *static*
structure of the floor: racks, walls, obstacles as metric floor-contact
footprints plus a height for the 2.5D render. No Backbone import.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import yaml

ELEMENT_TYPES = {"rack", "wall", "obstacle"}
SHAPES = {"rectangle", "polygon"}


def _validate_footprint(fp, where: str) -> list[list[float]]:
    if not isinstance(fp, list) or len(fp) < 3:
        raise ValueError(f"{where}: footprint must be a polygon of >=3 [x, y] points")
    out = []
    for pt in fp:
        if not (isinstance(pt, (list, tuple)) and len(pt) == 2):
            raise ValueError(f"{where}: each footprint point must be [x, y] metres")
        out.append([float(pt[0]), float(pt[1])])
    return out


def validate_map(data: dict) -> dict:
    """Normalise + validate a layout dict. Raises ``ValueError`` on bad input."""
    elements = []
    for i, el in enumerate(data.get("elements") or []):
        etype = el.get("type")
        if etype not in ELEMENT_TYPES:
            raise ValueError(f"element[{i}]: type must be one of {sorted(ELEMENT_TYPES)}")
        shape = el.get("shape", "polygon")
        if shape not in SHAPES:
            raise ValueError(f"element[{i}]: shape must be one of {sorted(SHAPES)}")
        try:
            height = float(el.get("height_m", 0.0))
        except (TypeError, ValueError):
            raise ValueError(f"element[{i}]: height_m must be a number") from None
        elements.append({
            "id": str(el.get("id") or f"el_{i}"),
            "type": etype,
            "shape": shape,
            "footprint": _validate_footprint(el.get("footprint"), f"element[{i}]"),
            "height_m": height,
            "label": str(el.get("label") or ""),
        })
    outline = None
    raw_outline = data.get("outline")
    if raw_outline and raw_outline.get("footprint"):
        outline = {"footprint": _validate_footprint(raw_outline["footprint"], "outline")}
    return {"elements": elements, "outline": outline}


def read_map(path: Path) -> dict:
    """Load + validate the layout YAML. Missing/empty file → empty layout."""
    path = Path(path)
    if not path.exists():
        return {"elements": [], "outline": None}
    raw = yaml.safe_load(path.read_text()) or {}
    return validate_map(raw)


def write_map(path: Path, data: dict) -> None:
    """Validate then atomically write the layout YAML (tempfile + os.replace)."""
    validated = validate_map(data)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            yaml.safe_dump(validated, f, sort_keys=False, allow_unicode=True)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cd monitor_web && pytest tests/test_warehouse_map.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add monitor_web/monitor_web/warehouse_map.py monitor_web/tests/test_warehouse_map.py
git commit -m "feat(map): warehouse_map.yaml data model (load/validate/write)"
```

---

## Task 3: Config setting + `/api/warehouse-map` endpoints

**Files:**
- Modify: `monitor_web/monitor_web/config.py`
- Create: `monitor_web/monitor_web/api/routes_map.py`
- Modify: `monitor_web/monitor_web/app.py`
- Test: `monitor_web/tests/test_routes_map.py`

- [ ] **Step 1: Write the failing test**

Create `monitor_web/tests/test_routes_map.py`:

```python
"""/api/warehouse-map round-trip + /api/warp-snapshot."""
from __future__ import annotations

from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from monitor_web.app import create_app
from monitor_web.config import Settings


def _app(tmp_path: Path):
    bb = tmp_path / "backbone.yaml"
    bb.write_text(yaml.safe_dump({"cameras": {}, "metadata": {"sinks": []}}))
    return create_app(Settings(backbone_config_path=bb, udp_port=0, port=0,
                               warehouse_map_path=tmp_path / "warehouse_map.yaml"))


def test_warehouse_map_empty_then_round_trip(tmp_path):
    with TestClient(_app(tmp_path)) as client:
        assert client.get("/api/warehouse-map").json() == {"elements": [], "outline": None}
        payload = {"elements": [{"id": "rack_a1", "type": "rack", "shape": "rectangle",
                                 "footprint": [[2, 0], [3.5, 0], [3.5, 1], [2, 1]],
                                 "height_m": 2.5, "label": "A1"}], "outline": None}
        assert client.post("/api/warehouse-map", json=payload).status_code == 200
        got = client.get("/api/warehouse-map").json()
        assert got["elements"][0]["id"] == "rack_a1"
        assert (tmp_path / "warehouse_map.yaml").exists()


def test_warehouse_map_rejects_bad_payload(tmp_path):
    with TestClient(_app(tmp_path)) as client:
        bad = {"elements": [{"id": "x", "type": "ufo",
                             "footprint": [[0, 0], [1, 0], [1, 1]], "height_m": 1}]}
        assert client.post("/api/warehouse-map", json=bad).status_code == 400
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd monitor_web && pytest tests/test_routes_map.py -v`
Expected: FAIL with `TypeError: ... unexpected keyword argument 'warehouse_map_path'`

- [ ] **Step 3: Add the config setting**

In `monitor_web/monitor_web/config.py`, after the `danger_zones_object_path` field (~line 34-38), add:

```python
    warehouse_map_path: Path = Field(
        default=_REPO_ROOT / "config" / "warehouse_map.yaml",
        description="Static warehouse layout twin (racks/walls/obstacles) for the floor map.",
    )
```

- [ ] **Step 4: Implement the router**

Create `monitor_web/monitor_web/api/routes_map.py`:

```python
"""Warehouse layout twin endpoints + rectified-floor snapshot for tracing.

Consumer-side: reads the calibration (via routes_projection) and the layout YAML.
No Backbone import.
"""
from __future__ import annotations

import base64
import logging

import cv2
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from ..floor_rectify import build_fit_rectify_matrix, fit_rectify_bounds, rectify_frame
from ..warehouse_map import read_map, write_map
from .routes_video import _frame_iter, _load_cameras_from_backbone_yaml, _warp_camera

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/api/warehouse-map")
async def get_warehouse_map(request: Request) -> JSONResponse:
    cfg = request.app.state.settings
    return JSONResponse(read_map(cfg.warehouse_map_path))


@router.post("/api/warehouse-map")
async def post_warehouse_map(request: Request) -> JSONResponse:
    cfg = request.app.state.settings
    body = await request.json()
    try:
        write_map(cfg.warehouse_map_path, body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(read_map(cfg.warehouse_map_path))


@router.get("/api/warp-snapshot/{camera_id}")
async def warp_snapshot(camera_id: str, request: Request) -> JSONResponse:
    """One rectified floor frame + its metric bounds, for use as a tracing underlay.

    Returns ``{image (b64 jpeg|null), x_min, y_min, px_per_m, out_wh}``. 404 when the
    camera isn't configured or isn't calibrated for the current mode.
    """
    cfg = request.app.state.settings
    cameras = _load_cameras_from_backbone_yaml(cfg.backbone_config_path)
    if camera_id not in cameras:
        raise HTTPException(status_code=404, detail=f"camera {camera_id!r} not configured")
    cam = _warp_camera(cfg, camera_id)
    if cam is None:
        raise HTTPException(status_code=404, detail="camera not calibrated for current mode")
    bounds = fit_rectify_bounds(cam.H, cam.image_size_wh)
    if bounds is None:
        raise HTTPException(status_code=409, detail="degenerate rectification (re-calibrate)")
    M, out_wh = build_fit_rectify_matrix(cam.H, cam.image_size_wh)

    image_b64 = None
    try:                                  # best-effort single frame; bounds still returned if it fails
        src = cameras[camera_id].get("source", {})
        frames = _frame_iter(camera_id, src)
        raw = next(frames)
        frames.close()
        warped = rectify_frame(raw, cam.K, cam.D, cam.H, out_wh=out_wh, M=M)
        ok, buf = cv2.imencode(".jpg", warped, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            image_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
    except Exception as exc:  # noqa: BLE001 — snapshot image is optional
        logger.warning("warp-snapshot %s: no frame (%s)", camera_id, exc)

    return JSONResponse({
        "image": image_b64,
        "x_min": bounds["x_min"], "y_min": bounds["y_min"],
        "px_per_m": bounds["px_per_m"], "out_wh": list(bounds["out_wh"]),
    })
```

- [ ] **Step 5: Register the router**

In `monitor_web/monitor_web/app.py`, add the import alongside the other `from .api import (...)` names:

```python
    routes_map,
```

and after `app.include_router(routes_video.router)` add:

```python
    app.include_router(routes_map.router)
```

- [ ] **Step 6: Run tests to verify pass**

Run: `cd monitor_web && pytest tests/test_routes_map.py -v`
Expected: PASS (2 tests — the round-trip and the 400 rejection)

- [ ] **Step 7: Commit**

```bash
git add monitor_web/monitor_web/config.py monitor_web/monitor_web/api/routes_map.py monitor_web/monitor_web/app.py monitor_web/tests/test_routes_map.py
git commit -m "feat(map): /api/warehouse-map + /api/warp-snapshot endpoints"
```

---

## Task 4: warp-snapshot uncalibrated-404 test (hardening)

**Files:**
- Test: `monitor_web/tests/test_routes_map.py`

- [ ] **Step 1: Add the test**

Append to `monitor_web/tests/test_routes_map.py`:

```python
def test_warp_snapshot_uncalibrated_404(tmp_path):
    bb = tmp_path / "backbone.yaml"
    bb.write_text(yaml.safe_dump(
        {"cameras": {"cam_a": {"source": {"name": "rtsp", "url": "rtsp://x/y"}}},
         "metadata": {"sinks": []}}))
    app = create_app(Settings(backbone_config_path=bb, udp_port=0, port=0,
                              warehouse_map_path=tmp_path / "warehouse_map.yaml"))
    with TestClient(app) as client:
        # configured but no calibration file present for the mode
        assert client.get("/api/warp-snapshot/cam_a").status_code == 404
        # unknown camera
        assert client.get("/api/warp-snapshot/cam_z").status_code == 404
```

- [ ] **Step 2: Run test to verify it passes**

Run: `cd monitor_web && pytest tests/test_routes_map.py::test_warp_snapshot_uncalibrated_404 -v`
Expected: PASS (the endpoint returns 404 for uncalibrated/unknown cameras without touching a live camera)

- [ ] **Step 3: Commit**

```bash
git add monitor_web/tests/test_routes_map.py
git commit -m "test(map): warp-snapshot 404 for uncalibrated/unknown camera"
```

---

## Task 5: Rectangle-snap draw mode

**Files:**
- Modify: `monitor_web/monitor_web/static/js/draw_mode.js`
- Test: `monitor_web/tests/test_draw_rect_snap.mjs` (node, pure-logic)

- [ ] **Step 1: Write the failing pure-logic test**

Create `monitor_web/tests/test_draw_rect_snap.mjs`:

```javascript
// Pure-geometry test for the rectangle-snap helper (no DOM/Pixi).
import assert from "node:assert";
import { rectFromCorners, snapToGrid } from "../monitor_web/static/js/draw_mode.js";

// two opposite corners → 4 axis-aligned corners (TL, TR, BR, BL order)
const r = rectFromCorners([2.0, 1.0], [3.5, 2.0]);
assert.deepStrictEqual(r, [[2.0, 1.0], [3.5, 1.0], [3.5, 2.0], [2.0, 2.0]]);

// grid snap to 0.1 m
assert.deepStrictEqual(snapToGrid([2.04, 1.07], 0.1), [2.0, 1.1]);
console.log("ok");
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd monitor_web && node tests/test_draw_rect_snap.mjs`
Expected: FAIL with `SyntaxError`/`does not provide an export named 'rectFromCorners'`

- [ ] **Step 3: Add the pure helpers + a rectSnap mode to `draw_mode.js`**

In `monitor_web/monitor_web/static/js/draw_mode.js`, add these exported pure functions near the top (after the module comment):

```javascript
// Two opposite corners → axis-aligned rectangle as 4 world points (TL,TR,BR,BL).
export function rectFromCorners([x1, y1], [x2, y2]) {
  const xa = Math.min(x1, x2), xb = Math.max(x1, x2);
  const ya = Math.min(y1, y2), yb = Math.max(y1, y2);
  return [[xa, ya], [xb, ya], [xb, yb], [xa, yb]];
}

// Snap a world point to a metric grid (e.g. 0.1 m). step<=0 → no snap.
export function snapToGrid([x, y], step) {
  if (!step || step <= 0) return [x, y];
  const r = (v) => Math.round(v / step) * step;
  // round to the grid then to 4 dp to kill FP dust (2.04→2.0, 1.07→1.1)
  return [Number(r(x).toFixed(4)), Number(r(y).toFixed(4))];
}
```

Then extend `onMapPointerDown` so that when the active session has `mode === "rectSnap"`, points are grid-snapped and the 2nd click finalises a rectangle. Replace the body of `onMapPointerDown` with:

```javascript
function onMapPointerDown(ev) {
  if (!active || active.target !== "map") return;
  const { fm, canvas, points } = active;
  const rect = canvas.getBoundingClientRect();
  const sx = ev.clientX - rect.left;
  const sy = ev.clientY - rect.top;
  let x = fm.transform.stageToWorldX(sx);
  let y = fm.transform.stageToWorldY(sy);
  if (active.mode === "rectSnap") {
    [x, y] = snapToGrid([x, y], active.gridStep || 0);
    points.push([x, y]);
    if (points.length === 2) {                       // 2 corners → finish a rectangle
      const r = rectFromCorners(points[0], points[1]);
      const onDone = active.onDone;
      cleanup();
      onDone?.(r);
      return;
    }
  } else {
    points.push([x, y]);
  }
  active.toolbar.count.textContent = String(points.length);
  renderMapPreview();
}
```

Finally, in `startDraw(...)`, store the new options on the `active` map object (in the `target === "map"` branch, where `active = { target: "map", mode, ... }`), adding `gridStep`:

```javascript
    active = {
      target: "map", mode, fm, canvas, preview,
      points: [], toolbar, onDone, minPoints, maxPoints,
      gridStep: arguments[0].gridStep || 0,
    };
```

(`mode` already flows through `startDraw({ mode })`; pass `mode: "rectSnap", gridStep: 0.1` from the layout manager.)

- [ ] **Step 4: Run tests to verify pass**

Run: `cd monitor_web && node tests/test_draw_rect_snap.mjs && node --check monitor_web/static/js/draw_mode.js`
Expected: `ok` then no output (parse clean)

- [ ] **Step 5: Commit**

```bash
git add monitor_web/monitor_web/static/js/draw_mode.js monitor_web/tests/test_draw_rect_snap.mjs
git commit -m "feat(map): rectangle-snap + grid-snap draw mode"
```

---

## Task 6: Pixi underlay + 2.5D layout layer

**Files:**
- Modify: `monitor_web/monitor_web/static/js/floor_map.js`

- [ ] **Step 1: Add the layers (after the existing layer block ~line 182-188)**

In `monitor_web/monitor_web/static/js/floor_map.js`, where the layers are created and added to the stage, insert an underlay layer **first** (drawn under everything) and a layout layer **between grid and zone**:

```javascript
  const underlayLayer = new PIXI.Container();   // rectified-photo tracing underlay
  const layoutLayer   = new PIXI.Container();   // 2.5D static warehouse twin
  const gridLayer  = new PIXI.Container();
  const zoneLayer  = new PIXI.Container();
  const arrowLayer = new PIXI.Container();
  const lineLayer  = new PIXI.Container();
  const objectLayer = new PIXI.Container();
  const drawLayer = new PIXI.Container();
  app.stage.addChild(underlayLayer, gridLayer, layoutLayer, zoneLayer,
                     arrowLayer, lineLayer, objectLayer, drawLayer);
```

- [ ] **Step 2: Add the underlay + layout render functions (near `drawZones`)**

```javascript
  let underlaySprite = null;

  // Place the rectified floor photo at its true world rectangle, with opacity.
  function setUnderlay(snapshot) {            // {image, x_min, y_min, px_per_m, out_wh} | null
    if (underlaySprite) { underlayLayer.removeChildren(); underlaySprite = null; }
    if (!snapshot || !snapshot.image) return;
    const [ow, oh] = snapshot.out_wh;
    const wx0 = snapshot.x_min, wy0 = snapshot.y_min;
    const wx1 = wx0 + ow / snapshot.px_per_m;
    const wy1 = wy0 + oh / snapshot.px_per_m;
    PIXI.Assets.load(snapshot.image).then((tex) => {
      const s = new PIXI.Sprite(tex);
      const sx0 = transform.worldToStageX(wx0), sy0 = transform.worldToStageY(wy0);
      const sx1 = transform.worldToStageX(wx1), sy1 = transform.worldToStageY(wy1);
      s.x = Math.min(sx0, sx1); s.y = Math.min(sy0, sy1);
      s.width = Math.abs(sx1 - sx0); s.height = Math.abs(sy1 - sy0);
      s.alpha = 0.5;
      underlayLayer.addChild(s);
      underlaySprite = s;
    });
  }
  function setUnderlayOpacity(a) { if (underlaySprite) underlaySprite.alpha = a; }

  const LAYOUT_FILL = { rack: 0x5b6b7a, wall: 0x444b52, obstacle: 0x8a5a3c };
  // Procedural 2.5D: footprint shadow → extruded sides → top face (iso offset ∝ height).
  function drawLayout(elements) {
    layoutLayer.removeChildren();
    const ISO = 6;  // screen px of "up" per metre of height
    for (const el of elements || []) {
      const base = el.footprint.map(([x, y]) => [transform.worldToStageX(x), transform.worldToStageY(y)]);
      const dy = -(el.height_m || 0) * ISO;            // extrude upward on screen
      const top = base.map(([x, y]) => [x, y + dy]);
      const fill = LAYOUT_FILL[el.type] ?? 0x5b6b7a;
      // shadow
      const sh = new PIXI.Graphics();
      sh.poly(base.flat()).fill({ color: 0x000000, alpha: 0.25 });
      layoutLayer.addChild(sh);
      // side faces (darker)
      for (let i = 0; i < base.length; i++) {
        const j = (i + 1) % base.length;
        const side = new PIXI.Graphics();
        side.poly([base[i][0], base[i][1], base[j][0], base[j][1],
                   top[j][0], top[j][1], top[i][0], top[i][1]])
            .fill({ color: fill, alpha: 0.55 }).stroke({ color: 0x000000, width: 1, alpha: 0.3 });
        layoutLayer.addChild(side);
      }
      // top face (lighter)
      const tf = new PIXI.Graphics();
      tf.poly(top.flat()).fill({ color: fill, alpha: 0.9 }).stroke({ color: 0xffffff, width: 1, alpha: 0.25 });
      layoutLayer.addChild(tf);
    }
  }
```

- [ ] **Step 3: Fetch the layout on load + on a refresh event, and expose setters**

Where `main()` fetches `/api/zones` (~line 192), add a layout fetch and initial render, then near where `window.__floor_map` is assigned, expose the new functions:

```javascript
  fetch("/api/warehouse-map").then(r => r.json()).then(d => drawLayout(d.elements)).catch(() => {});
  document.addEventListener("layout:changed", () => {
    fetch("/api/warehouse-map").then(r => r.json()).then(d => drawLayout(d.elements)).catch(() => {});
  });

  window.__floor_map = Object.assign(window.__floor_map || {}, {
    app, transform, drawLayer,
    setUnderlay, setUnderlayOpacity, drawLayout,
  });
```

(If `window.__floor_map` is already assigned with specific fields, merge — don't drop `drawLayer`/`transform`, which `draw_mode.js` depends on.)

- [ ] **Step 4: Verify parse + manual render check**

Run: `cd monitor_web && node --check monitor_web/static/js/floor_map.js`
Expected: no output (parse clean).

Manual: start the dashboard, `POST /api/warehouse-map` a sample rack via curl, switch to MAP — a shaded 2.5D box appears under the tracks. (Full UI wiring is Task 7.)

- [ ] **Step 5: Commit**

```bash
git add monitor_web/monitor_web/static/js/floor_map.js
git commit -m "feat(map): Pixi underlay sprite + 2.5D layout layer"
```

---

## Task 7: Layout authoring UI

**Files:**
- Create: `monitor_web/monitor_web/static/js/layout_manager.js`
- Modify: `monitor_web/monitor_web/templates/dashboard.html`
- Modify: `monitor_web/monitor_web/static/css/dashboard.css`
- Modify: `monitor_web/monitor_web/static/i18n/en.json`, `fr.json`

- [ ] **Step 1: Add i18n strings (en + fr, keep key sets identical)**

In `static/i18n/en.json` add before the closing brace (add a comma to the previous last line):

```json
  "layout_title": "Warehouse layout",
  "layout_camera": "Camera",
  "layout_underlay_opacity": "Underlay opacity",
  "layout_element_type": "Element",
  "layout_type_rack": "Rack",
  "layout_type_wall": "Wall",
  "layout_type_obstacle": "Obstacle",
  "layout_height": "Height (m)",
  "layout_trace_hint": "Click the two opposite floor corners of the base (not the top).",
  "layout_add": "Add element",
  "layout_save": "Save layout"
```

In `static/i18n/fr.json` add the same keys with French values:

```json
  "layout_title": "Plan de l'entrepôt",
  "layout_camera": "Caméra",
  "layout_underlay_opacity": "Opacité du fond",
  "layout_element_type": "Élément",
  "layout_type_rack": "Rayonnage",
  "layout_type_wall": "Mur",
  "layout_type_obstacle": "Obstacle",
  "layout_height": "Hauteur (m)",
  "layout_trace_hint": "Cliquez les deux coins opposés de la base au sol (pas le sommet).",
  "layout_add": "Ajouter un élément",
  "layout_save": "Enregistrer le plan"
```

- [ ] **Step 2: Verify i18n parity**

Run:
```bash
cd monitor_web && python -c "import json; e=json.load(open('monitor_web/static/i18n/en.json')); f=json.load(open('monitor_web/static/i18n/fr.json')); assert set(e)==set(f), set(e)^set(f); print('parity ok', len(e))"
```
Expected: `parity ok <n>`

- [ ] **Step 3: Add the authoring panel markup**

In `monitor_web/monitor_web/templates/dashboard.html`, add a Layout panel near the zone-manager markup (a sibling of the existing modals). Minimal markup:

```html
<div id="layout-panel" class="layout-panel hidden">
  <div class="layout-row">
    <label data-i18n="layout_camera">Camera</label>
    <select id="layout-cam"><option value="cam_a">Cam 1</option><option value="cam_b">Cam 2</option></select>
  </div>
  <div class="layout-row">
    <label data-i18n="layout_underlay_opacity">Underlay opacity</label>
    <input id="layout-opacity" type="range" min="0" max="1" step="0.05" value="0.5" />
  </div>
  <div class="layout-row">
    <label data-i18n="layout_element_type">Element</label>
    <select id="layout-type">
      <option value="rack" data-i18n="layout_type_rack">Rack</option>
      <option value="wall" data-i18n="layout_type_wall">Wall</option>
      <option value="obstacle" data-i18n="layout_type_obstacle">Obstacle</option>
    </select>
    <label data-i18n="layout_height">Height (m)</label>
    <input id="layout-height" type="number" min="0" step="0.1" value="2.5" style="width:5em" />
  </div>
  <p class="layout-hint" data-i18n="layout_trace_hint">Click the two opposite floor corners of the base.</p>
  <div class="layout-row">
    <button id="layout-add" data-i18n="layout_add">Add element</button>
    <button id="layout-save" data-i18n="layout_save">Save layout</button>
  </div>
</div>
```

Load the script before `</body>` alongside the other module scripts:

```html
<script type="module" src="/static/js/layout_manager.js"></script>
```

- [ ] **Step 4: Implement `layout_manager.js`**

Create `monitor_web/monitor_web/static/js/layout_manager.js`:

```javascript
// Warehouse-layout authoring: load a rectified underlay, trace rack/wall/obstacle
// rectangles on the MAP (grid-snapped), accumulate elements, save. Reuses draw_mode.
import { startDraw } from "/static/js/draw_mode.js";

const el = (id) => document.getElementById(id);
let elements = [];                 // working set, committed on Save

async function loadUnderlay() {
  const cam = el("layout-cam").value;
  const fm = window.__floor_map;
  try {
    const res = await fetch(`/api/warp-snapshot/${cam}`);
    fm.setUnderlay(res.ok ? await res.json() : null);
    fm.setUnderlayOpacity(parseFloat(el("layout-opacity").value));
  } catch { fm.setUnderlay(null); }
}

function addElement() {
  const type = el("layout-type").value;
  const height = parseFloat(el("layout-height").value) || 0;
  startDraw({
    target: "map",
    mode: "rectSnap",
    gridStep: 0.1,
    label: `${type} · click 2 opposite base corners`,
    onDone: (footprint) => {                       // world (X,Y) rectangle
      elements.push({ id: `${type}_${elements.length + 1}`, type, shape: "rectangle",
                      footprint, height_m: height, label: "" });
      window.__floor_map.drawLayout(elements);     // live preview
    },
  });
}

async function save() {
  const res = await fetch("/api/warehouse-map", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ elements, outline: null }),
  });
  if (res.ok) document.dispatchEvent(new CustomEvent("layout:changed"));
}

function wire() {
  if (!el("layout-panel")) return;
  // seed working set from the saved layout
  fetch("/api/warehouse-map").then(r => r.json()).then(d => { elements = d.elements || []; }).catch(() => {});
  el("layout-cam").addEventListener("change", loadUnderlay);
  el("layout-opacity").addEventListener("input", () =>
    window.__floor_map.setUnderlayOpacity(parseFloat(el("layout-opacity").value)));
  el("layout-add").addEventListener("click", addElement);
  el("layout-save").addEventListener("click", save);
  loadUnderlay();
}

if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", wire);
else wire();
```

- [ ] **Step 5: Add panel CSS**

In `monitor_web/monitor_web/static/css/dashboard.css` append:

```css
.layout-panel { position: absolute; top: 64px; right: 16px; z-index: 1500;
  background: rgba(20,24,28,0.92); border: 1px solid rgba(255,255,255,0.12);
  border-radius: 12px; padding: 12px 14px; width: 280px; color: #e6e6e6; }
.layout-panel.hidden { display: none; }
.layout-row { display: flex; align-items: center; gap: 8px; margin: 6px 0; }
.layout-row label { font-size: 12px; color: #9aa5b1; }
.layout-hint { font-size: 11px; color: #9aa5b1; margin: 4px 0; }
```

- [ ] **Step 6: Verify parse + page render**

Run:
```bash
cd monitor_web && node --check monitor_web/static/js/layout_manager.js && pytest tests/test_routes_pages.py -q
```
Expected: parse clean + dashboard template still renders (pages test green).

- [ ] **Step 7: Commit**

```bash
git add monitor_web/monitor_web/static/js/layout_manager.js monitor_web/monitor_web/templates/dashboard.html monitor_web/monitor_web/static/css/dashboard.css monitor_web/monitor_web/static/i18n/en.json monitor_web/monitor_web/static/i18n/fr.json
git commit -m "feat(map): layout authoring UI (underlay, opacity, rect-snap trace, save)"
```

---

## Task 8: Full-suite regression + manual verification

**Files:** none (verification only)

- [ ] **Step 1: Run the whole backend suite + lint**

Run:
```bash
cd monitor_web && pytest -q && ruff check monitor_web
```
Expected: all tests pass (existing + the new `test_warehouse_map.py`, `test_routes_map.py`, `test_floor_rectify.py` additions), ruff clean.

- [ ] **Step 2: Parse-check all touched JS**

Run:
```bash
cd monitor_web && for f in draw_mode floor_map layout_manager; do node --check monitor_web/static/js/$f.js && echo "$f ok"; done
```
Expected: `draw_mode ok`, `floor_map ok`, `layout_manager ok`.

- [ ] **Step 3: Manual end-to-end (requires a calibrated camera)**

1. Start the dashboard (`cd monitor_web && python -m monitor_web`), open `http://localhost:8000/`, MAP view.
2. Open the Layout panel; pick the calibrated camera → the rectified floor appears as an underlay; the **opacity slider** fades it.
3. Pick **Rack**, height 2.5, **Add element**, click the two opposite **base** corners of a rack on the map → a shaded 2.5D box appears (grid-snapped, axis-aligned).
4. Add a wall + obstacle; **Save** → confirm `config/warehouse_map.yaml` was written and the boxes persist on reload.
5. Confirm live `Track2D` sprites still render **above** the layout, and the trapezoid/black of the underlay does **not** appear once opacity is at 0.

- [ ] **Step 4: Final commit**

```bash
git add -A && git commit -m "chore(map): camera-traced 2.5D warehouse map complete"
```

---

## Self-Review Notes (completed)

- **Spec coverage:** data model (T2), endpoints incl. warp-snapshot + S-bounds (T1,T3,T4), underlay+opacity (T6,T7), rectangle/grid-snap (T5), 2.5D procedural render (T6), warehouse outline (data model supports `outline`; authoring of the outline rectangle is a thin follow-up reusing `rectSnap` — flagged below), tests (T1–T4, T5), Y-axis placement handled in `setUnderlay` via `worldToStage`. ✅
- **Deferred from spec (call out, don't silently drop):** the **warehouse `outline` authoring button** (place the floor rectangle) is not wired in T7 — the data model + render support it; add a "Set floor outline" action reusing `startDraw({mode:'rectSnap'})` writing to `outline` when desired. **Element delete/edit** is out of scope per spec §11.
- **Placeholder scan:** none — every step has runnable code/commands.
- **Type consistency:** `setUnderlay`/`setUnderlayOpacity`/`drawLayout` defined in T6 and called by the same names in T7; `rectFromCorners`/`snapToGrid` defined + tested in T5; snapshot JSON keys (`image,x_min,y_min,px_per_m,out_wh`) consistent across T3 endpoint, T6 `setUnderlay`, T7 fetch.
- **Prerequisite reminder:** accuracy depends on a **spread-point Mode-1 calibration** (spec §3) — not a code task, but tracing far racks on a single-pallet calibration will be imprecise.
