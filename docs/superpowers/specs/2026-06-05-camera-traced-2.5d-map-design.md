# Camera-Traced 2.5D Warehouse Map — Design

**Date:** 2026-06-05
**Status:** Approved design → ready for implementation plan
**Owner:** monitor_web (operator dashboard)

## 1. Goal

Give the operator floor map a **static spatial twin of the warehouse** — racks, walls, and
fixed obstacles — rendered as a **styled 2.5D top-down** backdrop that the live metric
`Track2D`/`Track3D` objects move across. The layout is **traced from the calibrated
camera(s)** (no LiDAR, no tape measure required), using the existing rectified bird's-eye
("warp") view as a **metric-aligned tracing underlay** with an opacity slider.

## 2. Scope & process boundary

- **Entirely in `monitor_web`** (consumer side). It reads the existing calibration/warp and
  renders into the existing Pixi map. **Zero changes to the Backbone** (`backbone.runtime`,
  `homography`, `triangulation`).
- `config/warehouse_map.yaml` is dashboard config, exactly like `config/zones.yaml`.

## 3. Key geometric facts (design rationale)

1. **The trapezoid is coverage, not distortion.** Homography rectification removes perspective
   on the floor plane, so a real floor rectangle (a rack base) rectifies to a **true rectangle**.
   The trapezoid/black-wedge shape of the rectified image is only the camera's **FOV footprint**
   on the floor — i.e. which floor is visible — not a warp of the content.
2. **The map is rectangular by construction.** Tracing happens in **world metres** (the Pixi
   map's coordinate space). The trapezoid/black never appears in the output map; it is only the
   underlay reference photo.
3. **The real limitation is accuracy with distance**: the far field is stretched + low-res in
   the rectified view, and a poorly-conditioned `H` (calibrated on one small pallet) is only
   accurate near that pallet. Mitigated by (a) rectangle-snap authoring and (b) the calibration
   prerequisite below.

### Prerequisite (not part of this build, but required for accuracy)
Re-run Mode-1 calibration with the 4 points **spread across a large floor rectangle** (using the
painted floor lines), not bunched on one pallet, so `H` is well-conditioned across the whole
visible floor. With a single-pallet calibration the underlay is only trustworthy near the pallet.

## 4. Data model — `config/warehouse_map.yaml`

Separate from `zones.yaml` (zones = rules/severity; layout = physical structure).

```yaml
elements:
  - id: rack_a1
    type: rack            # rack | wall | obstacle
    shape: rectangle      # rectangle | polygon
    footprint: [[2.10, 0.40], [3.60, 0.40], [3.60, 1.20], [2.10, 1.20]]  # metres, FLOOR base
    height_m: 2.5         # drives the 2.5D extrusion height
    label: "Rack A1"
outline:                  # optional warehouse floor rectangle (set once)
  footprint: [[0,0], [12.0,0], [12.0,8.0], [0,8.0]]   # metres
```

- `footprint` is always the **floor-contact base** polygon (trace the base, not the rack top —
  elevated structure shears in the warp).
- `shape: rectangle` elements are stored as their 4 corners but **authored** via the 2-corner
  rectangle-snap tool (Section 6).

## 5. Components

| Piece | New / reuse | Location |
|---|---|---|
| Rectified snapshot + metric bounds | **new** `GET /api/warp-snapshot/{cam}` | `api/routes_video.py` (or new `routes_map.py`) |
| Layout load/save | **new** `warehouse_map.py` + `GET/POST /api/warehouse-map` | `monitor_web/` + `api/routes_map.py` |
| Underlay sprite + opacity slider | **new** small addition | `static/js/floor_map.js` |
| Tracing corners → world metres | **reuse** `draw_mode.js` **map target** (already returns world `(X,Y)`) | `static/js/draw_mode.js` |
| Rectangle-snap + grid-snap tracing aid | **new** authoring mode | `static/js/draw_mode.js` (option) |
| Authoring UI (type/height, opacity, save) | **extend** the `+` modal pattern | `static/js/zone_manager.js` / new `layout_manager.js` |
| 2.5D rendering layer | **new** | `static/js/floor_map.js` |

## 6. Authoring flow

1. `+` modal → new **"Layout"** section → pick a camera.
2. Dashboard calls `GET /api/warp-snapshot/{cam}` → one rectified floor JPEG **plus its metric
   bounds** `{x_min, y_min, px_per_m, out_wh}`. The frontend places it as a Pixi sprite at that
   exact **world rectangle** under the draw layer. An **opacity slider** fades it from 1.0
   (trace against the real floor) → 0 (clean styled map).
3. Pick **type** (rack/wall/obstacle) + **height_m**. Trace on the **map**:
   - **Rectangle-snap (default for rack/wall):** click **2 opposite corners** → auto-form a clean
     rectangle; optional snap to a **0.1 m grid** and to right angles.
   - **Free polygon:** available for irregular obstacles.
4. **Save** → `POST /api/warehouse-map` → writes `warehouse_map.yaml` atomically (mirror the
   zones write path).
5. **Warehouse outline:** a one-time "set floor rectangle" action (4 clicks or width×length) →
   `outline` in the YAML, so the map frame is a clean rectangle even though one camera covers
   only a trapezoidal patch.
6. **Multi-camera (Mode 2):** each camera contributes its own underlay snapshot; since both share
   one world frame, snapshots drop into place automatically — trace each camera's visible region.

## 7. Endpoints

- `GET /api/warp-snapshot/{cam}` → `{image: <base64 jpeg>, x_min, y_min, px_per_m, out_wh}`.
  Computed identically to the warp stream — requires a small refactor of
  `floor_rectify.build_fit_rectify_matrix` to **also expose `S`** (or `x_min/y_min/px_per_m`) so
  the snapshot and the metric placement use one source of truth. Best-effort: 404/empty if the
  camera isn't calibrated.
- `GET /api/warehouse-map` → the parsed `warehouse_map.yaml`.
- `POST /api/warehouse-map` → validate + atomic write.

## 8. 2.5D rendering (procedural, no art assets)

In `floor_map.js`, a new layer rendered **below** the live track sprites. Per element, bottom→top:
1. **Shadow** — the footprint polygon, dark, slightly offset.
2. **Side faces** — extrude the footprint up the screen by an isometric vector ∝ `height_m`,
   shaded darker.
3. **Top face** — the footprint at the extruded offset, lighter, themed per type
   (rack = slatted grey-blue, wall = solid, obstacle = hazard stripes).

Procedural shaded boxes look "digital-twin" with **zero art assets**; swapping in textured
sprites is a later upgrade, not in scope.

## 9. Axis-convention detail

The rectified image is **+Y-down** (pixel `v` increases downward); the Pixi map world `Y` must be
aligned so the underlay sprite sits true over the traced metres. The snapshot placement maps
rectified-pixel `(0,0)` → world `(x_min, y_min)` and `(out_w, out_h)` → `(x_min + out_w/px,
y_min + out_h/px)`, consistent with `build_fit_rectify_matrix`'s +Y-down render.

## 10. Testing (hermetic, consumer-side)

- `warehouse_map.py`: YAML parse / serialize round-trip; schema validation (types, footprint
  shape, height).
- `/api/warehouse-map`: POST → GET round-trip; rejects malformed payloads.
- Warp-bounds math: `build_fit_rectify_matrix`'s exposed `S` inverts a known world point ↔
  rectified pixel correctly.
- `/api/warp-snapshot/{cam}`: returns 404/empty for an uncalibrated camera; returns bounds for a
  calibrated one (synthetic Mode-1 calibration fixture, as in `test_floor_rectify.py`).
- Pixi rendering is visual → covered by the data/endpoint tests + manual verification.

## 11. Out of scope / future

- Textured/sprite rack art (procedural shading first).
- Auto edge-detection of racks from the rectified image (manual tracing first).
- 3D (`Track3D` height) extrusion of moving objects.
- Editing existing elements via drag handles (author = click-to-create + delete first; drag-edit
  later if needed).

## 12. Open decisions resolved

- **Look:** 2.5D styled (procedural extrusion).
- **Source:** camera-traced, on the rectified underlay.
- **Underlay:** rectified photo as a metric-aligned Pixi sprite with an opacity slider.
- **Rectangle accuracy:** guaranteed by world-metre map + rectangle/grid-snap authoring; the
  trapezoid is only the reference photo.
