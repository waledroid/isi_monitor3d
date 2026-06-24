# isical — intrinsic capture-complete state + ingested-shot gallery

**Date:** 2026-06-24
**Component:** `isical/` (calibration Studio)
**Status:** approved design, ready for implementation plan

## Problem

In project `c1` both cameras reached the intrinsic capture target (cam_a 25/25,
cam_b 25/25) but the phase card stays ◐ partial, because the card only turns
green once the **Solve** has run (`work/intrinsic.json` exists). Capture-complete
is invisible. Separately, once a camera is captured there is no way to review the
ingested shots or judge whether they will produce good intrinsics — the live
view just keeps streaming.

This is not a bug; it is missing UX. Three additions:

1. A distinct **captured** state on the phase card (capture done, not yet solved).
2. When a camera hits its target, **replace its live view with a gallery** of the
   ingested shots, in per-camera tabs.
3. Surface **per-shot quality** (corner count + sharpness) and a **board-coverage
   map** in the gallery — the two best predictors of intrinsic quality.

## Non-goals (YAGNI)

- No shot deletion / delete-and-resnap. Gallery is view-only. (Easy follow-up.)
- No change to the solve, gating, or calibration math. The `captured` state is
  presentation-only; next-phase unlock still requires a real solve (green/done).
- No new third-party dependencies (coverage map is plain SVG/canvas).

## Current state (as explored)

- Phase board (`isical/static/js/phases.js`): each phase computes
  `state(s)` ∈ {`todo`,`partial`,`done`} and `intrinsic_done` =
  `work/intrinsic.json` exists. `done` → green and unlocks the next phase.
- `phase_status` (`isical/core/runners.py`) already returns `intrinsic_counts`
  (per-cam jpg counts), `targets.intrinsic` (default 25), `intrinsic_done`,
  `extrinsic_*`, etc.
- Capture page (`isical/templates/capture.html` + `static/js/capture.js`):
  intrinsic captures ONE selected camera via `#cam-select`; each `.cap-figure`
  holds a live MJPEG `<img class="cap-stream">` fed by `/stream/{name}/{cam}`.
  `pollStatus()` updates per-cam `count/target/status/detections`.
- Capture worker (`isical/capture/session.py` `IntrinsicWorker._save`): writes
  **only** the raw jpg to `data/<name>/intrinsic/<cam>/<cam>_NNN.jpg`. The
  `Detection` (with `.n` corners, `.centroid` normalized [0,1], `.blur_var`)
  is in scope at save time but not persisted.
- Detector (`isical/capture/detect.py` `ChArucoDetector.detect`) returns a
  `Detection`; can be reused to backfill metadata for already-captured shots.
- Image serving: only `/stream/{name}/{cam}` (live) exists. No route lists or
  serves individual ingested jpgs. Static mount is `/static` → `isical/static`.
- Project `c1` already has 25 jpgs/cam with **no** sidecars → the design must
  backfill metadata without re-capture.

## Design

### A. Phase card — `captured` state (`phases.js` + `studio.css`)

Add a fourth state between `partial` and `done`. Per phase (existing class
names kept; `captured` is the only new one):

- `todo` (grey) — no shots.
- `partial` (◐ amber) — ≥1 shot, but ≥1 configured camera below target
  (today's "partial", unchanged).
- **`captured`** (✓ blue, label "captured — Solve now") — **every configured
  camera** ≥ its target, and not yet solved. NEW.
- `done` (✓ green, RMS shown) — solved (`intrinsic_done` / `extrinsic_done`).

Implementation: replace the per-phase `state(s)` functions so they compute
"all cams reached target" from `intrinsic_counts`/`extrinsic_counts` vs
`targets`. The board's `prevDone`/lock logic and Solve/Export buttons are
unchanged — unlock still keys off `st === "done"`. Add a `.phase-card.captured`
CSS rule (blue accent) mirroring the existing `.done`/`.partial` rules. The
glyph map gains `captured → "✓"` with a "Solve now" hint.

Applies to **both** intrinsic and extrinsic cards (same capture→solve shape).
For extrinsic, "reached target" = pairs ≥ `targets.extrinsic` (floor shots are
not part of this gate; they remain a separate readout as today).

### B. Capture page — live → gallery swap + tabs (`capture.html` + `capture.js`)

- Render the camera chooser as **tabs** (cam_a / cam_b) instead of a bare
  `<select>` (the `<select>` may remain as the underlying state holder, or be
  replaced by tab buttons writing the same `activeCam()` value — implementation
  detail for the plan; behaviour is "one active cam at a time," unchanged).
- For the active camera:
  - `count < target` → live auto-snap stream (today's behaviour).
  - `count ≥ target` → swap the live `<img>` for that camera's **gallery**
    (Section D). Capture already auto-stops at target, so no frames are lost.
- `pollStatus()` already knows each cam's `count`/`target`; when a cam crosses
  its target, stop its stream (`img.src=""`) and load its gallery. Switching
  tabs to a not-yet-complete camera shows the live stream again.

### C. Shot metadata — sidecars + lazy backfill (`session.py` + new endpoint)

- **On new capture:** `IntrinsicWorker._save` writes a sidecar JSON next to each
  jpg: `data/<name>/intrinsic/<cam>/<cam>_NNN.json` =
  `{"corners": det.n, "centroid": [x, y] | null, "blur_var": det.blur_var}`.
  Near-zero cost; `det` already in hand. (Extrinsic saves pass a stub detection
  with no corners — sidecars there are optional; intrinsic is the target.)
- **Backfill for existing shots:** the list endpoint, for any jpg lacking a
  sidecar, runs `ChArucoDetector.detect` once on the saved image and writes the
  sidecar (cache). This makes `c1`'s pre-existing 25/cam work with no re-capture.
  Detection uses the project's board spec (`charuco_spec(cfg.board)`).

### Endpoints (`isical/api/routes_capture.py` or a small new router)

- `GET /api/p/{name}/shots/{phase}/{cam}` →
  `{"target": int, "count": int,
    "shots": [{"file": "<cam>_000.jpg", "corners": int,
               "centroid": [x, y] | null, "blur_var": float}, ...]}`
  Sorted by filename. Reads sidecars, backfilling as needed. `phase` ∈
  {`intrinsic`, `extrinsic`}; `cam` validated against configured cameras.
- `GET /shots/{name}/{phase}/{cam}/{file}` → `FileResponse` of the jpg.
  **Path-guarded:** resolve the final path and assert it is inside
  `data/<name>/<phase>/<cam>/`; reject `..`/absolute/traversal with 404.
  `{file}` constrained to a safe pattern (e.g. `^[A-Za-z0-9_\-]+\.jpg$`).

### D. Gallery rendering (`capture.js` + `studio.css`)

- **Thumbnail grid** of the shots from the list endpoint, each `<img>` =
  `/shots/{name}/{phase}/{cam}/{file}`.
- **Per-shot badges** overlaid on each thumb:
  - corner count, e.g. `18 ⌗`.
  - sharpness dot: green/amber/red from `blur_var` relative to the project's
    `blur_min_var` (e.g. ≥1.5× → green, ≥1× → amber, else red).
- **Coverage map**: one small frame-aspect box above the grid; one dot per shot
  at its normalized `centroid`, dot colour by corner count. Shots with
  `centroid == null` are omitted (or shown as a muted "?"). Plain SVG/canvas.
  Purpose: reveal spatial spread — corners-and-edges coverage = good intrinsics;
  center clump = weak. No axes/labels needed beyond a frame outline.

## Error handling

- List endpoint: missing project / unconfigured cam → 404. Unreadable/corrupt
  jpg during backfill → that shot reported with `corners: 0, centroid: null`
  (never 500 the whole list).
- Serve route: traversal or unknown file → 404 (path-guard above).
- Gallery JS: a failed list fetch leaves the live view in place and logs to the
  capture message line; it does not blank the panel.
- Backfill detection runs at most once per jpg (sidecar cached), so repeated
  polls are cheap.

## Testing (`isical/tests/`)

- `IntrinsicWorker._save` writes a sidecar with `corners`/`centroid`/`blur_var`.
- `GET /shots/...` list: returns metadata from existing sidecars; **lazily
  backfills** a sidecar-less jpg (assert the sidecar file appears) and reports
  its corners.
- `captured` state logic lives client-side in `phases.js` (computed from
  `intrinsic_counts`/`extrinsic_counts` vs `targets`); it is presentation-only,
  so verify by exercising the small pure state function (extract it so it is
  importable/testable) — all cams at target & unsolved ⇒ `captured`; with
  `intrinsic_done` ⇒ `done`. No server change for the state itself.
- Image-serve route is path-guarded: `..`/absolute/other-project paths → 404.

## Files touched (estimate)

- `isical/static/js/phases.js` — `captured` state + glyph/label.
- `isical/static/css/studio.css` — `.phase-card.captured`, gallery + coverage styles.
- `isical/templates/capture.html` — cam tabs + gallery container.
- `isical/static/js/capture.js` — tab logic, live→gallery swap, gallery render, coverage map.
- `isical/capture/session.py` — sidecar write in `_save`.
- `isical/api/routes_capture.py` (+ maybe a small new module) — list + serve endpoints, backfill.
- `isical/tests/…` — the cases above.
