# isical Intrinsic Capture-Complete State + Shot Gallery — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show capture progress in the calibration Studio — a distinct "captured" phase-card state, a per-camera tabbed gallery of ingested shots once a camera hits its target, and per-shot quality (corners + sharpness) plus a board-coverage map.

**Architecture:** Five focused changes to `isical/`: (1) write a per-shot metadata sidecar at capture time; (2) add `*_captured` booleans to the phase-status payload; (3) two new endpoints to list shot metadata (lazily backfilling sidecars) and serve the jpgs path-guarded; (4) a `captured` phase-card state in the board UI; (5) a live→gallery swap with camera tabs on the capture page. The state logic lives server-side (Python, testable); the gallery/coverage rendering is client-side SVG with no new dependencies.

**Tech Stack:** Python 3.10, FastAPI, pydantic, OpenCV (cv2), vanilla ES modules + Jinja2 templates, pytest + FastAPI `TestClient`.

## Global Constraints

- **Python target: 3.10** — no 3.11+/3.12+ syntax.
- **No new third-party dependencies** and **no Node/JS toolchain** — coverage map is hand-written SVG; tests are pytest only.
- **Tests are hermetic** — no real cameras, Multical, or GStreamer. Use `TestClient(create_app(Settings()))`; the `conftest.py` autouse fixture redirects `ISICAL_DATA_DIR` etc. to a tmp tree.
- **Lint clean:** `ruff check isical` must pass.
- **Run tests with the env python:** `/home/aatanda/miniforge3/envs/monitor3d/bin/python -m pytest isical/tests/...` (run from repo root `/home/aatanda/isi_monitor3d`).
- **Gallery is view-only** (no shot deletion). `captured` is presentation-only — it never changes solve/gating; next-phase unlock still requires `*_done`.
- Shot metadata schema (used verbatim across tasks): `{"corners": int, "centroid": [x, y] | null, "blur_var": float}` where `centroid` is normalized [0,1].

---

### Task 1: Per-shot metadata sidecar on capture

**Files:**
- Modify: `isical/capture/session.py` (add `_write_shot_meta` helper; call it in `IntrinsicWorker._save`, ~line 143)
- Test: `isical/tests/test_shot_meta.py` (create)

**Interfaces:**
- Produces: `isical.capture.session._write_shot_meta(jpg_path: pathlib.Path, det) -> None` — writes `jpg_path.with_suffix(".json")` containing the shot-metadata schema. `det` is a `capture.detect.Detection` (attrs `n: int`, `centroid: tuple|None`, `blur_var: float`).

- [ ] **Step 1: Write the failing test**

Create `isical/tests/test_shot_meta.py`:

```python
"""Per-shot metadata sidecar written next to each captured jpg."""

from __future__ import annotations

import json

from isical.capture.detect import Detection
from isical.capture.session import _write_shot_meta


def test_write_shot_meta_writes_sidecar(tmp_path):
    jpg = tmp_path / "cam_a_000.jpg"
    jpg.write_bytes(b"not-a-real-jpg")
    _write_shot_meta(jpg, Detection(n=18, centroid=(0.4, 0.6), blur_var=123.4))
    meta = json.loads((tmp_path / "cam_a_000.json").read_text())
    assert meta == {"corners": 18, "centroid": [0.4, 0.6], "blur_var": 123.4}


def test_write_shot_meta_handles_no_board(tmp_path):
    jpg = tmp_path / "cam_a_001.jpg"
    jpg.write_bytes(b"x")
    _write_shot_meta(jpg, Detection())          # n=0, centroid=None, blur_var=0.0
    meta = json.loads((tmp_path / "cam_a_001.json").read_text())
    assert meta == {"corners": 0, "centroid": None, "blur_var": 0.0}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/aatanda/miniforge3/envs/monitor3d/bin/python -m pytest isical/tests/test_shot_meta.py -v`
Expected: FAIL with `ImportError: cannot import name '_write_shot_meta'`

- [ ] **Step 3: Write minimal implementation**

In `isical/capture/session.py`, ensure `import json` is present near the top imports (add it if missing). Add this module-level helper just above the `IntrinsicWorker` class (search for `class IntrinsicWorker` or the existing `_save`; the helper must be importable at module scope):

```python
def _write_shot_meta(jpg_path: Path, det) -> None:
    """Persist per-shot quality next to the jpg (powers the Studio gallery).

    Schema: {"corners": int, "centroid": [x, y] | null, "blur_var": float},
    centroid normalized to [0, 1]. Best-effort: never raises into the capture loop.
    """
    centroid = getattr(det, "centroid", None)
    meta = {
        "corners": int(getattr(det, "n", 0) or 0),
        "centroid": [float(centroid[0]), float(centroid[1])] if centroid else None,
        "blur_var": float(getattr(det, "blur_var", 0.0) or 0.0),
    }
    try:
        jpg_path.with_suffix(".json").write_text(json.dumps(meta))
    except OSError:
        pass
```

Then in `IntrinsicWorker._save` (currently lines ~143-148), add the sidecar write right after `cv2.imwrite`:

```python
    def _save(self, raw: np.ndarray, det) -> None:
        idx = self.count
        path = self.out_dir / f"{self.camera_id}_{idx:03d}.jpg"
        cv2.imwrite(str(path), raw)
        _write_shot_meta(path, det)
        self.count += 1
        self._gate.note_kept(det)
```

(`Path` is already imported in `session.py`. If `import json` was not already present at the top, add it with the other stdlib imports.)

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/aatanda/miniforge3/envs/monitor3d/bin/python -m pytest isical/tests/test_shot_meta.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Verify the existing session test still passes**

Run: `/home/aatanda/miniforge3/envs/monitor3d/bin/python -m pytest isical/tests/test_session.py -v`
Expected: PASS (the stub-driven capture run now also drops sidecars; no assertions there break)

- [ ] **Step 6: Commit**

```bash
git add isical/capture/session.py isical/tests/test_shot_meta.py
git commit -m "isical: write per-shot metadata sidecar on intrinsic capture"
```

---

### Task 2: `*_captured` flags in phase status

**Files:**
- Modify: `isical/core/runners.py` — `phase_status` (lines ~228-259)
- Test: `isical/tests/test_routes.py` (append a test)

**Interfaces:**
- Consumes: `phase_status` already returns `intrinsic_counts: dict[str,int]`, `extrinsic_counts: dict[str,int]`, `cameras: list[str]`, `targets: {"intrinsic": int, "extrinsic": int}`.
- Produces: `phase_status(...)` dict gains `"intrinsic_captured": bool` and `"extrinsic_captured": bool` — True iff there is ≥1 configured camera AND every configured camera's count ≥ that phase's target. Independent of `*_done`.

- [ ] **Step 1: Write the failing test**

Append to `isical/tests/test_routes.py`:

```python
def test_status_captured_flags():
    from isical.config import Settings
    with _client() as c:
        c.post("/api/projects", json={"name": "rig", "cam_a": {"type": "rtsp", "url": "rtsp://x/a"}})
        st = c.get("/api/p/rig/status").json()
        assert st["intrinsic_captured"] is False
        assert st["extrinsic_captured"] is False
        # fill cam_a's intrinsic dir up to target
        d = Settings().data_dir / "rig" / "intrinsic" / "cam_a"
        target = st["targets"]["intrinsic"]
        for i in range(target):
            (d / f"cam_a_{i:03d}.jpg").write_bytes(b"x")
        st2 = c.get("/api/p/rig/status").json()
        assert st2["intrinsic_captured"] is True       # capture complete
        assert st2["intrinsic_done"] is False           # but not solved
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/aatanda/miniforge3/envs/monitor3d/bin/python -m pytest isical/tests/test_routes.py::test_status_captured_flags -v`
Expected: FAIL with `KeyError: 'intrinsic_captured'`

- [ ] **Step 3: Write minimal implementation**

In `isical/core/runners.py` `phase_status`, after the line `floors = {...}` and before the `return`, compute the flags (reuse the existing `cfg.capture.*` targets):

```python
    t_intr = cfg.capture.target_per_camera
    t_extr = cfg.capture.extrinsic_target
    intrinsic_captured = bool(cams) and all(intr.get(c, 0) >= t_intr for c in cams)
    extrinsic_captured = bool(cams) and all(extr.get(c, 0) >= t_extr for c in cams)
```

Add the two keys to the returned dict, and reuse `t_intr`/`t_extr` in the existing `targets` entry:

```python
    return {
        "cameras": cams, "mode2": cfg.is_mode2(),
        "intrinsic_counts": intr, "extrinsic_counts": extr, "floor": floors,
        "intrinsic_done": intrinsic_done,
        "extrinsic_done": extrinsic_done, "rms": rms,
        "calibration_json": str(calibration) if extrinsic_done else None,
        "installed": installed,
        "intrinsic_captured": intrinsic_captured,
        "extrinsic_captured": extrinsic_captured,
        "targets": {"intrinsic": t_intr, "extrinsic": t_extr},
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/aatanda/miniforge3/envs/monitor3d/bin/python -m pytest isical/tests/test_routes.py::test_status_captured_flags -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add isical/core/runners.py isical/tests/test_routes.py
git commit -m "isical: add intrinsic_captured/extrinsic_captured to phase status"
```

---

### Task 3: Shot-list + image-serve endpoints with lazy backfill

**Files:**
- Modify: `isical/api/routes_capture.py` (add imports, a `_shot_meta` helper, and two routes)
- Test: `isical/tests/test_routes.py` (append tests)

**Interfaces:**
- Consumes: `project_cfg`/`project_dir` from `.deps`; `_PHASES` constant already defined in the module; `cfg.configured_cameras()`, `cfg.capture.target_per_camera`, `cfg.capture.extrinsic_target`, `cfg.capture.blur_min_var`, `cfg.board`; `charuco_spec` from `..core.project`; `CharucoBoardDetector` from `..capture.detect`.
- Produces:
  - `GET /api/p/{name}/shots/{phase}/{cam}` → `{"target": int, "count": int, "blur_min_var": float, "shots": [{"file": str, "corners": int, "centroid": [x,y]|null, "blur_var": float}, ...]}` (shots sorted by filename).
  - `GET /shots/{name}/{phase}/{cam}/{file}` → `FileResponse` (image/jpeg), 404 on traversal / non-`.jpg` / missing.

- [ ] **Step 1: Write the failing tests**

Append to `isical/tests/test_routes.py`:

```python
def test_list_shots_and_serve_with_backfill():
    import cv2
    import numpy as np

    from isical.config import Settings
    with _client() as c:
        c.post("/api/projects", json={"name": "rig", "cam_a": {"type": "rtsp", "url": "rtsp://x/a"}})
        d = Settings().data_dir / "rig" / "intrinsic" / "cam_a"
        cv2.imwrite(str(d / "cam_a_000.jpg"), np.zeros((48, 64, 3), np.uint8))
        assert not (d / "cam_a_000.json").exists()              # no sidecar yet
        r = c.get("/api/p/rig/shots/intrinsic/cam_a").json()
        assert r["count"] == 1
        assert r["shots"][0]["file"] == "cam_a_000.jpg"
        assert "corners" in r["shots"][0] and "blur_var" in r["shots"][0]
        assert "blur_min_var" in r
        assert (d / "cam_a_000.json").exists()                   # backfilled + cached
        assert c.get("/shots/rig/intrinsic/cam_a/cam_a_000.jpg").status_code == 200


def test_shot_serve_path_guarded():
    with _client() as c:
        c.post("/api/projects", json={"name": "rig", "cam_a": {"type": "rtsp", "url": "rtsp://x/a"}})
        assert c.get("/shots/rig/intrinsic/cam_a/evil.png").status_code == 404      # not .jpg
        assert c.get("/shots/rig/intrinsic/cam_a/missing.jpg").status_code == 404   # absent


def test_list_shots_unknown_cam_404():
    with _client() as c:
        c.post("/api/projects", json={"name": "rig", "cam_a": {"type": "rtsp", "url": "rtsp://x/a"}})
        assert c.get("/api/p/rig/shots/intrinsic/cam_b").status_code == 404
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/aatanda/miniforge3/envs/monitor3d/bin/python -m pytest isical/tests/test_routes.py::test_list_shots_and_serve_with_backfill isical/tests/test_routes.py::test_shot_serve_path_guarded isical/tests/test_routes.py::test_list_shots_unknown_cam_404 -v`
Expected: FAIL with 404 (route not defined) on the list/serve calls

- [ ] **Step 3: Write minimal implementation**

In `isical/api/routes_capture.py`, add to the imports at the top:

```python
import json
import re
from pathlib import Path

from fastapi.responses import FileResponse
```

(The file already imports `StreamingResponse` from `fastapi.responses`; combine the import or add a second line — keep `ruff` happy.) Then add the helper + routes (place after the existing `stream` route at the bottom):

```python
_SHOT_FILE_RE = re.compile(r"^[A-Za-z0-9_\-]+\.jpg$")


def _shot_meta(jpg: Path, cfg) -> dict:
    """Metadata for one shot, reading the sidecar or backfilling via ChArUco detection.

    Backfill detects once on the saved jpg and caches the sidecar, so already-captured
    projects (no sidecars) work without re-capture. Never raises into the request.
    """
    side = jpg.with_suffix(".json")
    if side.exists():
        try:
            return json.loads(side.read_text())
        except (OSError, ValueError):
            pass
    meta = {"corners": 0, "centroid": None, "blur_var": 0.0}
    try:
        import cv2

        from ..capture.detect import CharucoBoardDetector
        from ..core.project import charuco_spec
        img = cv2.imread(str(jpg))
        if img is not None:
            det = CharucoBoardDetector(charuco_spec(cfg.board)).detect(img)
            meta = {
                "corners": int(det.n),
                "centroid": [float(det.centroid[0]), float(det.centroid[1])] if det.centroid else None,
                "blur_var": float(det.blur_var),
            }
            try:
                side.write_text(json.dumps(meta))
            except OSError:
                pass
    except Exception:
        pass
    return meta


@router.get("/api/p/{name}/shots/{phase}/{cam}")
def list_shots(request: Request, name: str, phase: str, cam: str) -> dict:
    if phase not in _PHASES:
        raise HTTPException(status_code=404, detail=f"phase must be one of {_PHASES}")
    d, cfg = project_cfg(request, name)
    if cam not in cfg.configured_cameras():
        raise HTTPException(status_code=404, detail=f"camera {cam!r} not configured")
    cam_dir = d / phase / cam
    jpgs = sorted(cam_dir.glob("*.jpg")) if cam_dir.is_dir() else []
    shots = [{"file": p.name, **_shot_meta(p, cfg)} for p in jpgs]
    target = cfg.capture.target_per_camera if phase == "intrinsic" else cfg.capture.extrinsic_target
    return {"target": target, "count": len(shots),
            "blur_min_var": float(cfg.capture.blur_min_var), "shots": shots}


@router.get("/shots/{name}/{phase}/{cam}/{file}")
def shot_image(request: Request, name: str, phase: str, cam: str, file: str) -> FileResponse:
    if phase not in _PHASES or not _SHOT_FILE_RE.match(file):
        raise HTTPException(status_code=404, detail="not found")
    d = project_dir(request, name)
    base = (d / phase / cam).resolve()
    target = (base / file).resolve()
    if base not in target.parents or not target.is_file():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(str(target), media_type="image/jpeg")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/aatanda/miniforge3/envs/monitor3d/bin/python -m pytest isical/tests/test_routes.py -v`
Expected: PASS (all, including the three new tests)

- [ ] **Step 5: Commit**

```bash
git add isical/api/routes_capture.py isical/tests/test_routes.py
git commit -m "isical: shot-list + image-serve endpoints with lazy metadata backfill"
```

---

### Task 4: `captured` phase-card state (board UI)

**Files:**
- Modify: `isical/static/js/phases.js` (the `PHASES` `state` fns + the card render block)
- Modify: `isical/static/css/studio.css` (add a `.phase-card.captured` block)
- Test: none new (state logic is server-side, covered by Task 2). Manual verification step included.

**Interfaces:**
- Consumes: status fields `intrinsic_done`, `intrinsic_captured`, `extrinsic_done`, `extrinsic_captured`, `*_counts` (from Task 2).

- [ ] **Step 1: Update the two `state` functions in `phases.js`**

In `isical/static/js/phases.js`, change the intrinsic phase's `state`:

```js
    state: (s) => s.intrinsic_done ? "done"
      : s.intrinsic_captured ? "captured"
      : Object.values(s.intrinsic_counts || {}).some((n) => n > 0) ? "partial" : "todo",
```

and the extrinsic phase's `state`:

```js
    state: (s) => s.extrinsic_done ? "done"
      : s.extrinsic_captured ? "captured"
      : Object.values(s.extrinsic_counts || {}).some((n) => n > 0) ? "partial" : "todo",
```

- [ ] **Step 2: Update the card render block in `phases.js`**

In `render()`, update the `glyph` and `card.className` lines and add a "Solve now" hint. Replace:

```js
    const glyph = locked ? "🔒" : st === "done" ? "✓" : st === "partial" ? "◐" : "";
    const card = document.createElement("div");
    card.className = "phase-card" + (locked ? " locked" : "") +
      (st === "done" ? " done" : st === "partial" ? " partial" : "");
```

with:

```js
    const glyph = locked ? "🔒" : (st === "done" || st === "captured") ? "✓"
      : st === "partial" ? "◐" : "";
    const card = document.createElement("div");
    card.className = "phase-card" + (locked ? " locked" : "") +
      (st === "done" ? " done" : st === "captured" ? " captured"
        : st === "partial" ? " partial" : "");
    const hint = st === "captured" ? `<div class="counts solve-hint">captured ✓ — Solve now ↓</div>` : "";
```

Then add `${hint}` to the card's `innerHTML`, right after the second `.counts` div:

```js
    card.innerHTML =
      `<div class="phase-head"><span class="phase-num">${ph.n}</span>
         <span class="phase-title">${ph.title}</span>
         <span class="phase-status">${glyph}</span></div>
       <div class="counts">${ph.counts(s)}</div>
       <div class="counts">${ph.extra(s)}</div>
       ${hint}
       <div class="phase-actions"></div>`;
```

- [ ] **Step 3: Add the `.captured` CSS block to `studio.css`**

In `isical/static/css/studio.css`, after the `.phase-card.partial ...` rules (around line 719), add a blue "captured" palette mirroring the partial/done blocks:

```css
/* Capture complete, not yet solved — solid light-blue card with dark text */
.phase-card.captured {
  background: #e8f1fc;
  border-color: #bcd6f2;
}
.phase-card.captured .phase-title { color: #0f2a4a; }
.phase-card.captured .counts { color: #2c5a8c; }
.phase-card.captured .phase-num { background: #cfe2f7; color: #1858a8; }
.phase-card.captured .phase-status { color: #1858a8; }
.phase-card.captured a { color: #1858a8; }
.phase-card.captured .solve-hint { font-weight: 600; color: #1858a8; }
.phase-card.captured .phase-actions button {
  background: #ffffff;
  border-color: #bcd6f2;
  color: #0f2a4a;
}
.phase-card.captured .phase-actions button:hover {
  background: #d8e8fb;
  border-color: #1858a8;
  color: #0f2a4a;
}
```

- [ ] **Step 4: Verify existing route tests still pass (no regressions)**

Run: `/home/aatanda/miniforge3/envs/monitor3d/bin/python -m pytest isical/tests/test_routes.py -v`
Expected: PASS

- [ ] **Step 5: Manual verification**

Start the Studio and open project `c1` (which has cam_a 25/25, cam_b 25/25, unsolved):

```bash
cd /home/aatanda/isi_monitor3d && /home/aatanda/miniforge3/envs/monitor3d/bin/python -m isical
```

Open `http://localhost:8300/p/c1`. Confirm the **Intrinsic** card is now **blue** with "captured ✓ — Solve now ↓" (not amber ◐). Hit **Solve**; once `intrinsic.json` is written it turns **green**. Stop the server (Ctrl-C).

- [ ] **Step 6: Commit**

```bash
git add isical/static/js/phases.js isical/static/css/studio.css
git commit -m "isical: blue 'captured' phase-card state (green only after solve)"
```

---

### Task 5: Capture page — live→gallery swap with camera tabs

**Files:**
- Modify: `isical/templates/capture.html` (cam tabs + per-figure gallery container)
- Modify: `isical/static/js/capture.js` (tabs, live↔gallery switching, gallery render, coverage map)
- Modify: `isical/static/css/studio.css` (gallery/tab/coverage styles)
- Test: `isical/tests/test_routes.py` (assert the new markup renders)

**Interfaces:**
- Consumes: `GET /api/p/{name}/shots/intrinsic/{cam}` and `GET /shots/{name}/intrinsic/{cam}/{file}` (Task 3); `GET /api/p/{name}/status` (`intrinsic_counts`, `targets.intrinsic`); existing `getJSON` from `/static/js/api.js`.

- [ ] **Step 1: Write the failing markup test**

Append to `isical/tests/test_routes.py`:

```python
def test_intrinsic_capture_page_has_tabs_and_gallery_markup():
    with _client() as c:
        c.post("/api/projects", json={"name": "rig", "cam_a": {"type": "rtsp", "url": "rtsp://x/a"},
                                      "cam_b": {"type": "rtsp", "url": "rtsp://x/b"}})
        html = c.get("/p/rig/capture/intrinsic").text
        assert "cam-tab" in html          # per-camera tab buttons
        assert "shot-gallery" in html      # gallery container per figure
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/aatanda/miniforge3/envs/monitor3d/bin/python -m pytest isical/tests/test_routes.py::test_intrinsic_capture_page_has_tabs_and_gallery_markup -v`
Expected: FAIL (`cam-tab`/`shot-gallery` not in HTML)

- [ ] **Step 3: Update `capture.html`**

In `isical/templates/capture.html`, replace the intrinsic branch of the toolbar (the `{% if phase == "intrinsic" %}` block containing the `<select id="cam-select">`) with tab buttons backed by a hidden select (keeps `camSelect.value` as the single source of truth for existing logic):

```html
    {% if phase == "intrinsic" %}
    <div class="cam-tabs">
      {% for c in cameras %}<button type="button" class="cam-tab" data-cam="{{ c }}">{{ c }}</button>{% endfor %}
    </div>
    <select id="cam-select" hidden>
      {% for c in cameras %}<option value="{{ c }}">{{ c }}</option>{% endfor %}
    </select>
    {% else %}
    <button id="cap-start">Start capture</button>
    {% endif %}
```

And add a gallery container inside each capture figure — replace the `{% for c in cameras %}...{% endfor %}` figure block in `#cap-views` with:

```html
    {% for c in cameras %}
    <figure class="cap-figure" data-cam="{{ c }}">
      <figcaption>{{ c }} <span class="counts" data-cam="{{ c }}">—</span></figcaption>
      <div class="canvas-wrap"><img class="cap-stream" data-cam="{{ c }}" alt="{{ c }} live"></div>
      <div class="shot-gallery" data-cam="{{ c }}" hidden></div>
    </figure>
    {% endfor %}
```

- [ ] **Step 4: Add gallery logic to `capture.js`**

In `isical/static/js/capture.js`, add these helpers near the top (after the `const` declarations). They are intrinsic-only:

```js
// ---- ingested-shot gallery (intrinsic only) ----
function coverageSVG(shots) {
  const W = 160, H = 90;
  const dots = shots.filter((s) => s.centroid).map((s) => {
    const [x, y] = s.centroid;
    const col = s.corners >= 16 ? "#1c7a3f" : s.corners >= 8 ? "#95680f" : "#b00020";
    return `<circle cx="${(x * W).toFixed(1)}" cy="${(y * H).toFixed(1)}" r="3" `
         + `fill="${col}" fill-opacity="0.8"/>`;
  }).join("");
  return `<svg class="cov-svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">`
       + `<rect x="0.5" y="0.5" width="${W - 1}" height="${H - 1}" fill="none" stroke="#cbd5e1"/>`
       + `${dots}</svg>`;
}

function thumbHTML(cam, s, blurMin) {
  const sharp = s.blur_var >= blurMin * 1.5 ? "ok" : s.blur_var >= blurMin ? "warn" : "bad";
  return `<figure class="shot">
    <img loading="lazy" src="/shots/${project}/intrinsic/${cam}/${s.file}" alt="${s.file}">
    <figcaption><span class="badge">${s.corners} ⌗</span>
      <span class="dot ${sharp}" title="sharpness ${Math.round(s.blur_var)}"></span></figcaption>
  </figure>`;
}

async function showGallery(cam) {
  const fig = document.querySelector(`.cap-figure[data-cam="${cam}"]`);
  if (!fig) return;
  const gal = fig.querySelector(".shot-gallery");
  try {
    const r = await getJSON(`/api/p/${project}/shots/intrinsic/${cam}`);
    const blurMin = r.blur_min_var || 80;
    gal.innerHTML =
      `<div class="coverage">${coverageSVG(r.shots)}
         <span class="msg">${r.count}/${r.target} shots · board coverage</span></div>
       <div class="shot-grid">${r.shots.map((s) => thumbHTML(cam, s, blurMin)).join("")}</div>`;
    fig.querySelector(".canvas-wrap").style.display = "none";
    gal.hidden = false;
  } catch { /* keep live view on error */ }
}

function showLive(cam) {
  const fig = document.querySelector(`.cap-figure[data-cam="${cam}"]`);
  if (!fig) return;
  fig.querySelector(".canvas-wrap").style.display = "";
  const gal = fig.querySelector(".shot-gallery");
  gal.hidden = true; gal.innerHTML = "";
}

async function switchCam(cam) {
  document.querySelectorAll(".cam-tab").forEach((b) =>
    b.classList.toggle("active", b.dataset.cam === cam));
  showOnly(cam);
  let done = false;
  try {
    const st = await getJSON(`/api/p/${project}/status`);
    done = (st.intrinsic_counts?.[cam] || 0) >= (st.targets?.intrinsic || Infinity);
  } catch { /* fall through to live */ }
  if (done) { showLive(cam); await showGallery(cam); }
  else { showLive(cam); startCapture(false); }
}
```

Add `import` note: `getJSON` is already imported at the top of `capture.js` (`import { flash, getJSON, sendJSON } from "/static/js/api.js";`).

- [ ] **Step 5: Wire tabs + auto-swap into the existing handlers**

In `capture.js`, replace the `if (camSelect) { ... }` auto-start block (currently calling `showOnly` + `startCapture(false)` and re-starting on change) with tab-aware logic:

```js
// Intrinsic: tabs drive the (hidden) select; switching decides live vs gallery.
if (camSelect) {
  document.querySelectorAll(".cam-tab").forEach((btn) => {
    btn.onclick = async () => {
      if (btn.dataset.cam === camSelect.value && !stopBtn.disabled) return; // already live here
      await stopCapture();
      camSelect.value = btn.dataset.cam;
      switchCam(btn.dataset.cam);
    };
  });
  switchCam(camSelect.value);     // auto-open the first camera on load
}
```

And in `pollStatus()`, swap to the gallery the moment the active camera reaches its target. Replace the loop body:

```js
async function pollStatus() {
  try {
    const s = await getJSON(`/api/p/${project}/capture/status`);
    if (!s.active) return;
    for (const [cam, c] of Object.entries(s.cameras || {})) {
      const el = document.querySelector(`.counts[data-cam="${cam}"]`);
      if (el) el.textContent = `${c.count}/${c.target} · ${c.status} · ${c.detections} det`;
      if (phase === "intrinsic" && cam === activeCam() && c.count >= c.target) {
        await stopCapture();
        showGallery(cam);
      }
    }
  } catch { /* studio busy */ }
}
```

- [ ] **Step 6: Add gallery/tab CSS to `studio.css`**

Append to `isical/static/css/studio.css`:

```css
/* ---- capture: camera tabs + ingested-shot gallery ---- */
.cam-tabs { display: flex; gap: 6px; }
.cam-tab {
  padding: 4px 14px; border-radius: 6px; cursor: pointer;
  background: #ffffff; border: 1px solid #d7dde6; color: #334155; font-weight: 600;
}
.cam-tab.active { background: #e8f1fc; border-color: #1858a8; color: #0f2a4a; }

.shot-gallery .coverage {
  display: flex; align-items: center; gap: 10px; margin-bottom: 8px;
}
.cov-svg { background: #0b1220; border-radius: 4px; }
.shot-grid {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); gap: 8px;
}
.shot { position: relative; margin: 0; }
.shot img { width: 100%; border-radius: 4px; display: block; }
.shot figcaption {
  position: absolute; left: 4px; bottom: 4px; display: flex; align-items: center; gap: 6px;
}
.shot .badge {
  background: rgba(0, 0, 0, 0.65); color: #fff; font-size: 11px;
  padding: 1px 6px; border-radius: 10px;
}
.shot .dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
.shot .dot.ok { background: #1c7a3f; }
.shot .dot.warn { background: #d9a514; }
.shot .dot.bad { background: #b00020; }
```

- [ ] **Step 7: Run the markup test + full route suite**

Run: `/home/aatanda/miniforge3/envs/monitor3d/bin/python -m pytest isical/tests/test_routes.py -v`
Expected: PASS (including `test_intrinsic_capture_page_has_tabs_and_gallery_markup`)

- [ ] **Step 8: Manual verification**

```bash
cd /home/aatanda/isi_monitor3d && /home/aatanda/miniforge3/envs/monitor3d/bin/python -m isical
```

Open `http://localhost:8300/p/c1/capture/intrinsic`. Because cam_a is already 25/25, its tab should open straight into the **gallery** (25 thumbnails with corner-count badges + sharpness dots, and a coverage map of board positions) instead of the live stream. Switch to the **cam_b** tab — also 25/25 → its gallery. (On a fresh camera below target, the live auto-snap view shows and flips to the gallery the instant it reaches 25/25.) Stop the server.

- [ ] **Step 9: Commit**

```bash
git add isical/templates/capture.html isical/static/js/capture.js isical/static/css/studio.css isical/tests/test_routes.py
git commit -m "isical: cam tabs + ingested-shot gallery (badges + coverage map) at target"
```

---

### Task 6: Final verification — full suite + lint

- [ ] **Step 1: Run the entire isical test suite**

Run: `/home/aatanda/miniforge3/envs/monitor3d/bin/python -m pytest isical/tests -v`
Expected: PASS (all, including the new `test_shot_meta.py` and the appended `test_routes.py` cases)

- [ ] **Step 2: Lint**

Run: `/home/aatanda/miniforge3/envs/monitor3d/bin/python -m ruff check isical`
Expected: `All checks passed!` (fix any findings, then re-run)

- [ ] **Step 3: Confirm no regression in the backbone suite touched by Task 1's shared import (sanity)**

Run: `/home/aatanda/miniforge3/envs/monitor3d/bin/python -m pytest isical/tests -q`
Expected: all green. (No backbone files were modified in this plan.)

---

## Self-Review Notes

- **Spec coverage:** (1) `captured` card state → Tasks 2+4; (2) live→gallery swap with cam tabs → Task 5; (3) per-shot badges + coverage map → Tasks 1+3+5; sidecars + lazy backfill → Tasks 1+3. All spec sections map to a task.
- **Deviation from spec (intentional, improves testability):** the `captured` state is computed **server-side** (`*_captured` flags, Task 2) instead of purely in `phases.js`. Reason: isical has no JS test runner; a server flag is hermetically testable in pytest and keeps one source of truth. `phases.js` only reads it.
- **Type consistency:** shot-metadata schema `{corners:int, centroid:[x,y]|null, blur_var:float}` is identical in Task 1 (`_write_shot_meta`), Task 3 (`_shot_meta` + backfill), and Task 5 (`thumbHTML`/`coverageSVG`). The list endpoint's extra fields (`target`, `count`, `blur_min_var`) are consumed by `showGallery` in Task 5.
- **No placeholders:** every step contains real code/commands; all hex colors are valid.
