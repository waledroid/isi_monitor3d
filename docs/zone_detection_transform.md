# Zone detection — the background worker & the coordinate transform

How zone detections are produced by the **background `ZoneDetectionWorker`** (one thread per camera, one
coherent snapshot per frame — the no-shaky / no-dual-detection architecture) and how a detection found in a
**small, compressed zone slice** is mapped **back onto the full cam1 frame** — while cam1's full-frame
inference does **only pose**.

> TL;DR — a zone is a polygon on cam1. A **single background worker per camera** crops every zone's bounding
> rectangle out of the **same frame**, shrinks each crop to ~320 px (`INTER_AREA`), detects on the small crop,
> maps each box/mask back with `source = (x0 + det·rx, y0 + det·ry)`, resolves cross-zone overlaps, and
> publishes **one atomic snapshot**. The ZONE panels and cam1 are **pure renderers** of that snapshot — no
> detection happens in any HTTP stream. Pose runs once on the whole frame; pose never runs inside a zone.

---

## 1. Why we do this

Running the full 640-px object detector on every cam1 frame is the single heaviest GPU consumer (it's what
kept exhausting the 12 GB card and triggering `CUDA 700`). Instead:

- **Detect only inside zones**, on a **small 320-px crop** of each — far less VRAM/compute, and *more accurate*
  for the objects in that region (the crop fills the model input instead of being a few pixels of a 640 frame).
- **cam1's full-frame inference is pose-only** — people are tracked anywhere on the frame.
- **Pose never runs in a zone** (a zone is a pallet/object ROI; running pose there is wasted GPU).
- All zones + cam1 **share one `(model, infer_size)` detector session** — no duplication.

**Tradeoff:** only objects *inside a zone polygon* get a box on cam1. Cover the areas you want watched with
zones. Pose (people) is unaffected — it is always whole-frame.

---

## 2. The worker — why detections no longer shake or double

### What was wrong before (the duplicate/shaky-mask bug)

Detection used to be **HTTP-driven**: each ZONE panel's `/stream/zone/{id}` connection ran its own detect loop
in its own server thread and wrote a shared cache **with its own timestamp**. cam1 merged entries up to
**1.5 s apart** — so for a moving object it drew zone 1's *fresh* detection **and** zone 2's *stale* one at an
offset position (the "second" box/mask), and the mask was upscaled with `INTER_NEAREST` (blocky → "shaky").
Other defects: zones without an open panel never detected at all (only 2 of up to 6 zones ever ran), and two
connections to the same zone raced on the cache.

### The architecture now (`monitor_web/zone_worker.py`)

```
                    ┌──────────────────────────────────────────────────────┐
 camera (hub) ───►  │  ZoneDetectionWorker  — ONE daemon thread per camera │
                    │  every tick (display_fps):                           │
                    │   1. grab the LATEST real frame (skip if unchanged)  │
                    │   2. for EACH zone (sequentially, SAME frame):       │
                    │        crop → INTER_AREA ≤320 → detect → conf filter │
                    │        → drop persons → polygon clip → remap to      │
                    │          full-frame coords                           │
                    │   3. resolve cross-zone overlaps (one object ⇒       │
                    │      exactly ONE zone)                               │
                    │   4. publish ONE atomic snapshot                     │
                    │        {frame_ts, zones: {zone_id: [dets]}}          │
                    └──────────────────────────────────────────────────────┘
                                   │ (read-only, lock-free)
              ┌────────────────────┼─────────────────────┐
              ▼                    ▼                     ▼
      /stream/zone/{id}    /stream/video/cam_a     (future consumers)
      panel = crop +       cam1 = full frame +
      draw snapshot        draw snapshot + POSE
      (NO detection)       (NO object detection)
```

Key properties:

- **One frame, one timestamp.** All zones are detected on the *same* frame and published together — a moving
  object can never appear twice at two moments in time. This is the dual-detection fix.
- **Single writer.** Only the worker thread writes; it publishes by assigning a *fresh* dict in one statement
  (atomic under the GIL). Readers never see a half-written snapshot, and there is nothing to race on.
- **Cross-zone overlap resolution at publish time.** Same-class boxes that describe one physical object
  (IoU / centre-containment / clipped-intersection test) collapse to a single detection, owned by the zone
  whose polygon contains the box centre **deepest** (`cv2.pointPolygonTest(..., measureDist=True)`); ties go
  to the higher confidence. Each object lands in **exactly one zone**.
- **Polygon sandbox.** Each zone's detections are clipped to its drawn **polygon** (bbox-centre
  point-in-polygon), not just its bounding rectangle — overlapping bounding rects no longer double-report.
- **Smooth masks.** The mask upscale back to source resolution uses **`INTER_LINEAR` + 0.5 threshold**
  (was `INTER_NEAREST`) — soft edges instead of blocky shake.
- **Renderers, not detectors.** `/stream/zone/{id}` and the cam view just *draw* the snapshot. Opening,
  closing, or duplicating panel connections cannot change detection behaviour or load.
- **All zones detect** — including zones 3–6 that have no panel.
- **Idle before START.** While the Backbone isn't running the worker releases the camera stream, publishes an
  empty snapshot, and burns no GPU — the panels/cam show the raw feed (the pre-START state).
- **Staleness guard.** A snapshot older than 1 s (camera down, worker stopped) yields nothing — consumers draw
  no ghosts.
- **Lifecycle.** Started in the app lifespan; `ZoneWorkerManager.reload()` runs on every zone save / config
  save (new camera ⇒ new worker, last zone deleted ⇒ worker stopped, camera source changed ⇒ stream
  re-acquired).

### Does each zone need its own process? No.

A separate process per zone would cost a full CUDA context (~0.5–0.8 GB **each**) on the 12 GB card and fix
nothing: the detector session is **stateless** per call, so sharing it was never the problem — the duplication
was timing + geometry. The sandboxing that matters is **data ownership**: one writer thread per camera, zones
isolated by polygon, consumers read-only.

---

## 3. The coordinate spaces

A detection travels through five frames of reference. Knowing which space a number lives in is the whole game.

```
 ① cam1 SOURCE frame            native sensor px, e.g. 1280 × 720
        │  (the frame the worker pulled from the camera hub)
        │
        │  patch_rect()  +  patch_pixel_box()   ← polygon → axis-aligned box, rescaled to the LIVE size
        ▼
 ② zone BOUNDING RECT           source px:  (x0, y0) … (x1, y1)     size cw × ch
        │
        │  cv2.resize(INTER_AREA),  s = infer_size / max(cw, ch)    (only when the crop is larger than infer)
        ▼
 ③ FED crop                     ≤ infer_size px:  fw × fh   (e.g. 320 × ~180)
        │
        │  detector.detect()    ← letterboxes fed → 320×320 internally, then INVERTS back to fw × fh
        ▼
 ④ model INPUT (320²)           internal only — the Detection comes back in ③'s fw × fh frame
        │
        │  _remap_det():   ×(rx = cw/fw, ry = ch/fh)   then  +(x0, y0)      ← THE INVERSE MAP
        ▼
 ① cam1 SOURCE frame again      box/mask now in 1280 × 720 px  →  published in the snapshot
        │
        │  panel renderer: _to_crop() (−x0, −y0 shift + mask slice)   ← back to the crop for the ZONE panel
        │  cam overlay:    sourceToDisplay()  (object-fit: contain)   ← browser display only
        ▼
 ⑤ DISPLAY px                   the ZONE panel crop, or the cam1 <canvas> overlay
```

The key insight: the detector returns coordinates in the **fed-crop frame (③, `fw×fh`)**, because each detector
inverts its own letterbox back to the image it was handed. So the inverse map only has to undo **two** steps:
the `INTER_AREA` downscale (③→②) and the crop offset (②→①).

---

## 4. Forward path — draw → detect (inside the worker)

| Step | What | Where |
|---|---|---|
| **Draw** | Operator clicks polygon vertices on cam1. Stored as `polygon: [[u,v]…]` in **source px**, with `frame_wh = [W,H]` (the natural size it was drawn at). | `static/js/zone_patch.js` → `startPatchDraw` |
| **Bounding rect** | Polygon → axis-aligned bounding box `[x0,y0,x1,y1]` (the polygon is the display boundary; the *crop* is its rect). | `routes_zone_patches.py::patch_rect()` |
| **Frame-size guard** | Rescale the rect from `frame_wh` (drawn-at) to the **live** frame's natural size, clamp, → integer pixel box. | `routes_zone_patches.py::patch_pixel_box()` |
| **Crop** | `crop = frame[y0:y1, x0:x1]` → size `cw × ch` (source px). | `zone_worker.py::ZoneDetectionWorker._detect_zone()` |
| **Compress** | If `max(cw,ch) > infer_size`: `s = infer_size/max(cw,ch)`; `fed = cv2.resize(crop, …, INTER_AREA)`. `INTER_AREA` is the correct **high-quality downscale** filter; smaller crops are left to the detector's letterbox. | `zone_worker.py::_detect_zone()` |
| **Detect** | `get_zone_detector(model, cfg, infer_size)` → a detector built at `input_size = infer_size` (default **320**), cached per `(model, size)` and shared by all zones. Returns `Detection`s in the **fed frame** (`fw×fh`). | `detection_overlay.py::get_zone_detector()` |
| **Filter** | Per-zone confidence post-filter (the session is built at a low floor, so changing a zone's conf is instant — no session rebuild) → drop person classes → **clip to the polygon** (bbox-centre point-in-polygon in fed coords). | `zone_worker.py::_detect_zone()` |
| **Remap + publish** | `_remap_det()` to full-frame coords; after all zones: cross-zone overlap resolution; one atomic snapshot. | `zone_worker.py::_detect_all_zones()` / `_resolve_overlaps()` |

---

## 5. Inverse path — detect → cam1 frame  *(the core)*

`zone_worker.py::_remap_det()` maps one `Detection` from the fed-crop frame back to full-frame source pixels.

Given a detection at `det = (dx, dy)` in the fed crop:

```
rx = cw / fw          # undo the INTER_AREA downscale (x)
ry = ch / fh          # undo the INTER_AREA downscale (y)

source_x = x0 + dx * rx
source_y = y0 + dy * ry
```

- **Bounding box** `[bx0,by0,bx1,by1]` → `[x0+bx0·rx, y0+by0·ry, x0+bx1·rx, y0+by1·ry]` (affine).
- **Foot point** `(fu,fv)` → `(x0+fu·rx, y0+fv·ry)` (affine).
- **Mask** (a `fh×fw` bool array in the fed frame):
  1. **`INTER_LINEAR`-resize on float32 + `>= 0.5` threshold** to the crop size `cw×ch` (undo the downscale —
     smooth edges, no nearest-neighbour shake),
  2. paste into a fresh full-frame `ih×iw` bool array at offset `(x0, y0)`.

A new `Detection` is returned carrying the full-frame `bbox_xyxy`, `foot_uv`, and `mask` (class/confidence
unchanged) — this is what the snapshot holds. The renderers consume it:

- **cam1** — `annotate_frame(image, detector=None, detections=<snapshot>, pose_detector=…)` draws the supplied
  detections then runs pose on the whole frame (`routes_video.py::_detect_iter`).
- **ZONE panel** — `_to_crop()` shifts each detection by `(−x0, −y0)` and slices the mask back to the crop, so
  the panel shows exactly the same result on its slice (`routes_video.py::_zone_render_iter`).
- **Browser overlay** (display only) — `live_overlay.js::sourceToDisplay()` converts source px → display px
  (object-fit: contain), so boxes ride the CAM panel's letterbox correctly.

---

## 6. Worked example (verified end-to-end)

Full frame **1280 × 720**, one zone covering `(300,20) … (1150,710)`, `infer_size = 320`:

1. Bounding rect `cw×ch = 850×690`. `max = 850 > 320` → `s = 320/850 ≈ 0.376` → `fed ≈ 320×260` (`fw=320`).
2. Detect on the 320 crop → `polybag`, conf **0.99**.
3. Map back: `rx = 850/320 ≈ 2.66`, `ry = 690/260 ≈ 2.65`, offset `(300,20)`.
4. Result on the full frame: **box `(362, 36) – (1080, 707)`**, mask shape **`(720, 1280)`**.

Compare to detecting the *same object on the full frame directly*: ground-truth box `(362, 34) – (1078, 719)`.
**Near-identical** — the round trip is faithful, at a fraction of the GPU cost.

---

## 7. Code map

| Symbol | File | Role |
|---|---|---|
| `ZoneDetectionWorker` | `zone_worker.py` | **The detection driver** — one thread per camera; all zones on one frame; atomic snapshot. |
| `ZoneWorkerManager` | `zone_worker.py` | Worker per camera-with-zones; `reload()` on zone/config save; lifespan start/stop. |
| `_detect_zone()` | `zone_worker.py` | Crop → `INTER_AREA` → detect → conf/person/polygon filters → remap (one zone, one frame). |
| `_resolve_overlaps()` | `zone_worker.py` | Cross-zone dedupe: one object ⇒ one zone (deepest polygon centre wins). |
| `_remap_det()` | `zone_worker.py` | **The inverse map** — fed→crop scale + crop→source offset; mask `INTER_LINEAR`+0.5. |
| `_same_object()` / `_zone_objects()` | `zone_worker.py` | Same-object test (IoU ∨ centre-containment ∨ clipped-intersection) + dedupe utility. |
| `startPatchDraw` | `static/js/zone_patch.js` | Draw the zone polygon in source px (+ `frame_wh`). |
| `patch_rect()` / `patch_pixel_box()` | `api/routes_zone_patches.py` | Polygon → bounding rect; drawn-size → live-size integer pixel box. |
| `_zone_render_iter()` / `_to_crop()` | `api/routes_video.py` | ZONE panel **renderer** (crop + draw snapshot — no detection). |
| `_detect_iter()` | `api/routes_video.py` | cam1 stream: draw snapshot + pose when zones exist; full-frame fallback otherwise. |
| `get_zone_detector()` | `detection_overlay.py` | Detector built at `input_size = infer_size`, cached per `(model, size)`, build-locked. |
| `annotate_frame(…, detections=…)` | `detection_overlay.py` | Draw supplied detections (no full-frame detect) + pose + distance + occupancy. |
| `gpu_inference_safe()` | `detection_overlay.py` | VRAM guard — skip a frame's inference if the card is nearly full. |
| `sourceToDisplay()` | `static/js/live_overlay.js` | Source px → cam1 display px (object-fit: contain) for the overlay. |

---

## 8. Notes & tradeoffs

- **Coverage:** only objects *inside a zone polygon* are detected on cam1 — draw zones over the racks/lanes you
  want watched. Pose (people) is whole-frame regardless.
- **`infer_size` is per-zone** (default 320). Raise it for a large zone with small/far objects (downscaling to
  320 shrinks them); a pallet-sized region at 320 is ideal.
- **`confidence` is per-zone** (blank = global) and applied as a **post-filter** — changing it never rebuilds
  the CUDA session, so edits are instant.
- **`INTER_AREA` is downscale-only.** When a crop is *smaller* than `infer_size` we don't pre-resize — the
  detector's own letterbox upscales it (linear). The high-quality `INTER_AREA` path only kicks in when shrinking.
- **Session sharing:** all zones + cam1 share one `(model, infer_size)` ONNX Runtime session (double-check-locked
  build). A 320 session is much lighter than the old 640 full-frame one — the whole point.
- **Worker cadence:** throttled to the Settings `display_fps` (default 10); unchanged frames are skipped, so a
  slow camera never causes redundant inference.
- **Masks** map exactly like boxes (resize + paste), so seg overlays and pallet occupancy keep working on cam1.
