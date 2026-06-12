# ISI Monitor 3D — Project Task Reference

> Heavy reference / running to-do. Phase-by-phase status of the warehouse-vision system.
> **Last updated:** 2026-06-12 · **Tests:** 370 backbone + 235 monitor_web (all green).

## What this is
A single **Backbone** process turns RTSP video into **metric, identity-stable metadata** published over
**UDP/JSON**, an operator **dashboard** (`monitor_web`) consumes that contract, and an external **trainer**
(`isidet`) produces the detection `.onnx`. Two operating modes:

| Mode | Cameras | Calibration | Output |
|---|---|---|---|
| **Mode 1** `single_cam_homography` | 1 (cam_a) | `calibrate single-cam` (4-pt floor fit) | `Track2D` only (no 3D) |
| **Mode 2** `dual_cam_homography_triangulation` | 2 (cam_a + cam_b) | `calibrate-all` (Multical joint BA) | `Track2D` always + `Track3D` (subscribed) |

Mode is chosen at build time from `len(cameras)`; the pipeline is **deliberately unified** — the same
solo-emit code covers Mode-1 startup and Mode-2 single-camera degradation.

**Legend:** ✅ done & tested · 🔄 in progress · ⬜ to do · 🔬 needs the physical rig · 🅼1 Mode 1 · 🅼2 Mode 2 · ⏸ deferred

---

## Part A — Backbone core (S0–S8) — ✅ COMPLETE

| Sprint | Area | Status | Where |
|---|---|---|---|
| S0 | Skeleton & core: registry, types, 5 ABC seams | ✅ | `backbone/core/` |
| S1 | Shared utils + calibration schema (camera_rig, geometry, zones, timestamps) | ✅ | `backbone/shared/`, `calibration/schema.py` |
| S1.5 | Multical calibration backend (isolated venv) | ✅ 🅼2 | `calibration/calibrate.py`, `multical_io.py`, `setup_multical.sh` |
| S2 | Ingestion: RTSP (GStreamer/PyGObject) + replay + V4L2 + sync + bus | ✅ | `backbone/ingestion/` |
| S3 | Detection: YOLO11 ONNX (CUDA), RF-DETR-seg, OpenVINO, pose | ✅ | `backbone/detection/` |
| S4 | Homography: foot-proj → cross-cam fusion → disagreement gate → ByteTrack-in-m → temporal vote | ✅ | `backbone/homography/` |
| S5 | Triangulation: 2-cam DLT, reprojection gate, 3D Kalman, subscriptions | ✅ 🅼2 | `backbone/triangulation/` |
| S6 | Metadata (UDP/JSON schemas), orchestrator, latency probe | ✅ | `backbone/metadata/`, `backbone/runtime/`, `tools/latency_probe.py` |
| S7 | Simplification pass | ✅ | — |
| S8 | Operational modes + runtime degradation | ✅ 🅼1🅼2 | `orchestrator.py`, `frame_sync._try_emit_solo` |

KPI gates (reprojection ≤2 px, triangulation ≤5–8 px, latency p95 <200 ms) are wired + exercised with
**synthetic** ground truth. Real-rig numbers are pending (Part D/E).

---

## Part B — Detection model & training — 🔄

| Step | Status | Notes |
|---|---|---|
| isidet trainer (YOLO11/26, RF-DETR), `isi-train` env | ✅ | `trainer/isidet/`, run via `scripts/run_train.py` |
| RF-DETR-seg medium trained (0.938 mask mAP) | ✅ | `models/rfdetr/…` |
| RF-DETR re-export @ 672 + 840 (higher-res far objects) | ✅ | `rfdetr_seg_medium_{672,840}.sim.onnx` |
| **yolo26l-seg @ 640, 200 ep, patience 20** (3-class palette/carton/polybag) | 🔄 | epoch ~104; `best.pt` saved; far/small aug applied |
| Export winner `.onnx` + wire into `backbone.yaml detection:` | ⬜ | currently **RF-DETR-672 stopgap** is configured |
| Production training that meets mAP ≥ 0.90 / occupancy P-R ≥ 0.95/0.93 | ⬜ 🔬 | held-out + on-site eval |

---

## Part C — Dashboard (monitor_web) — ✅ feature-complete (consumer-side)

| Feature | Status | Notes |
|---|---|---|
| Big panel MAP / CAM1 / CAM2 / UNIFIED / MP4 toggle | ✅ | `big_panel.js` Alpine store |
| **Single-WebSocket video transport** (all panels over one `/ws/video`; MJPEG kept for debug/MP4) | ✅ | `routes_ws_video.py`, `video_ws.js` |
| One shared RTSP session per cam via CameraHub | ✅ | `camera_hub.py` |
| **Cam views POSE-ONLY** (no full-frame detector ever; objects solely from zone workers) | ✅ | `_detect_iter` (routes_video.py) |
| Detection overlay (boxes/masks/foot-nodes) + pose + occupancy badge | ✅ | `detection_overlay.py` |
| **Zone isolation guards**: VRAM admission + per-zone circuit breaker + per-zone `max_fps` | ✅ | `zone_worker.py`, `get_zone_detector` |
| Pallet occupancy `palette_vide / palette_carton / palette_polybag / …` | ✅ | image-overlap (A) + metric (B) hybrid |
| Person↔pallet distance lines (elastic + distance badge) | ✅ | needs calibration |
| Pixi floor map (digital twin: sprites, zones, danger rings, arrows) | ✅ | `floor_map.js` |
| Zones (world polygons, Settings draw, palette/étagère/danger) | ✅ | carry over Mode 1 ↔ Mode 2, no redraw |
| **Zone-patch SAHI monitor** (pixel ROI → crop → detect in ZONE panels) | ✅ | `routes_zone_patches.py`, `/stream/zone/{id}` |
| Per-zone detection model picker | ✅ | dropdown per ZONE panel; cached per model |
| Settings modal (cameras, model, zones, **display-FPS cap**) | ✅ | `zone_manager.js`, `routes_config.py` |
| Calibration in-dashboard (single-cam 4-pt) | ✅ 🅼1 | `routes_calibrate.py` |
| 3-state status light + **degraded (amber)** state | ✅ | green = running + ALL cams live |
| Status panel **live KPIs** (latency p50/p95, reproj) + accordion expand | ✅ | `bus_subscriber` latency window |
| Logs: terminal access-noise filter + formatted LOGS panel (collapse repeats) | ✅ | `main.py`, `log_format.py` |
| Cam-view overscan (~120% normal, 100% on expand) | ✅ | overlays stay aligned |
| START/STOP supervisor, GB/FR i18n, hidden MP4 dev viewer | ✅ | — |
| Hidden MP4 viewer occupancy overlay | ✅ | parity with CAM preview |

**Runtime perf:** ✅ no pose inside zone patches · ✅ configurable display-FPS cap (default 10)
on all detect/patch/unified streams · ✅ per-zone `max_fps` budget (heavy zone stops dragging light ones) ·
✅ VRAM admission before zone-session builds (no more CUDA-700 from over-admission) ·
⬜ detector **idle-release** (reclaim a stopped session's GPU memory).

---

## Part D — 🅼1 Mode 1 completion (single camera) — 🔄

Mode 1 is **code-complete and calibrated**; remaining items are validation/hardening.

| Step | Status | Notes |
|---|---|---|
| Single-cam pipeline (Track2D over UDP) | ✅ | |
| cam_a calibrated (4-pt pallet, 1280×720) | ✅ | `config/mode1/calibration.json` |
| **Wire the trained production model** (B) | ⬜ | swap the RF-DETR-672 stopgap for the finished yolo26l-seg/RF-DETR-840 |
| Calibration hardening: re-do with **≥5 points** (tape-measured) so the residual gate validates | ⬜ 🔬 | current 4-pt fit is exactly-determined (residual ≈ 0) |
| **End-to-end KPI validation on the rig** | ⬜ 🔬 | latency p95 <200 ms; reproj ≤2 px; mAP ≥0.90; occupancy P-R |
| **Sustained live RTSP run** (not replay) + reconnect resilience | ⬜ 🔬 | `tools/rtsp_smoke.py` then full Backbone |
| systemd unit (industrial: supervised, deterministic restart) | ⬜ | only START/STOP dev supervisor exists today |

---

## Part E — 🅼2 Mode 2 build (two cameras + 3D) — 🔄 (code-ready, hardware-gated)

Backbone Mode 2 + the dashboard groundwork are built & tested **hermetically**; they light up when cam_b is
mounted + jointly calibrated.

| Step | Status | Notes |
|---|---|---|
| Mode detection from camera count; triangulation stack built only in Mode 2 | ✅ | `orchestrator.py` |
| Cross-cam fusion + disagreement gate (handles N=1 too) | ✅ | `backbone/homography/` |
| 2-cam DLT + reprojection gate + 3D Kalman | ✅ | `backbone/triangulation/` |
| Runtime degradation (cam_b drops → solo Track2D, 3D halts, ids persist, no shutdown) | ✅ | `frame_sync._try_emit_solo` |
| **Unified bird's-eye view** (both cams warped to one floor canvas, blended) | ✅ | `floor_rectify.shared_bev_layout`, `/stream/unified`, UNIFIED tab |
| **Live-feed gating** (UNIFIED tab only when both cams actually stream) | ✅ | per-cam liveness in `/api/status` |
| **Degraded indicator** (amber + "running on one camera") | ✅ | Phase D |
| **Single-view 3D floor fallback** (occluded in one cam → flagged Z=0 3D) | ✅ | schema v3 `single_view`/`confidence`; `allow_single_view` subscriptions |
| Single-view tracks rendered distinct on the map (dimmed, `·1cam`) | ✅ | `floor_map.js` |
| Mode-separated calibration files + auto-repoint by camera count | ✅ | `config/mode1/`, `config/mode2/` |
| **Mount physical cam_b** + configure in Settings | ⬜ 🔬 | empty Cam 2 ⇒ Mode 1; filled ⇒ Mode 2 |
| **Joint calibration** `calibrate-all` (Multical) → `config/mode2/calibration.json` | ⬜ 🔬 | prerequisite for fusion + 3D + unified view |
| Validate unified view + triangulation + single-view fallback **on the rig** | ⬜ 🔬 | walk a person behind an occluder → 3D stays continuous |
| Camera-placement guide (overlap critical zones + complementary angles) | ⬜ | doc; system already supports union-2D / overlap-3D |
| Coast-through 3D (short prediction through total gaps, flagged) | ⏸ | Phase C v2 |

---

## Part F — Project hygiene & infra — ✅/🔄

| Item | Status | Notes |
|---|---|---|
| Shared ORT session helper + arena cap (`kSameAsRequested`) everywhere | ✅ | `backbone/shared/ort_session.py` |
| GPU memory probe in heartbeat (`gpu=used/total MB`) | ✅ | `hardware.gpu_memory_mb` |
| Repo reorg: `archive/` (media/scratch), `docs/specs/` (PDFs), `config/mode{1,2}/` | ✅ | root 14 items |
| **trainer/ prune** (22 GB old `runs/`, dup weights, `mytest_*`) | ⬜ | **blocked until training finishes** (~10–15 GB reclaimable) |
| Version control | ✅ | repo on GitHub `waledroid/isi_monitor3d`, branch `main` (2026-06-12) |

---

## Part G — Version control, deployment & health — ⬜ (all to do)

Reproducible, version-controlled, supervised deployment. None of this exists yet; the repo has **no git**
and the system runs from the `monitor3d` conda env + the dashboard's dev START/STOP supervisor.
**Do the git/repo steps FIRST — Docker + ops build on a committed repo.**

### G.1 — Repo & version control — ✅ DONE (2026-06-12)
| Item | Status | Notes |
|---|---|---|
| **`git init` + initial commit** | ✅ | 386 files / ~11 MB; identity `waledroid <22135232+waledroid@users.noreply.github.com>` |
| **`.gitignore`** | ✅ | trainer `runs/data/logs/models` (~28 GB), `archive/` (2.1 GB), all `*.pt`, media, caches, `.venv-multical`, `*:Zone.Identifier`, `.claude/` local files |
| **`.gitattributes` / Git LFS** for weights | ✅ superseded | weights/datasets are **gitignored**, not LFS-tracked — the repo stays code-only; models ship via the deploy volume |
| Remote + branch model | ✅ | pushed to **github.com/waledroid/isi_monitor3d** `main` (public — note: live `config/backbone.yaml` incl. LAN rtsp creds committed by explicit choice); gh CLI 2.94 in `~/.local/bin`, auth via browser flow |

### G.2 — Containerization, deployment & health (build on the committed repo)
| Item | Status | Notes |
|---|---|---|
| **Backbone Dockerfile** (CUDA 12.9 + ONNX Runtime-GPU sm_120 + GStreamer 1.28 + OpenCV 4.13, py3.10) | ⬜ | mirror the `monitor3d` env; needs **NVIDIA Container Toolkit** (`--gpus all`) |
| **Dashboard Dockerfile** (`monitor_web`, FastAPI/uvicorn) | ⬜ | consumes the UDP bus + a shared `config/` + `models/` volume |
| **`docker-compose.yml`** — Backbone + dashboard on a shared UDP network; volumes for `config/`, `config/mode{1,2}/`, models, calibration | ⬜ | one-command bring-up; `.env` for RTSP creds/ports (never bake secrets) |
| Container `HEALTHCHECK` + entrypoints with graceful SIGTERM (deterministic restart) | ⬜ | aligns with the "industrial defaults" principle (supersedes/houses the systemd option) |
| **Jetson Orin NX image variant** (arm64, JetPack ORT) | ⬜ ⏸ | same `.onnx` + contract; only the ORT wheel differs |
| Trainer image (`isi-train`: torch cu128 + ultralytics + rfdetr) | ⬜ ⏸ | optional; training is external |
| **Ops / launch scripts** (`ops/`): `build.sh`, `up.sh`, `down.sh`, `restart.sh`, `logs.sh` (wrap compose; take config path / mode) | ⬜ | |
| **`ops/system_health.sh`** — go/no-go readiness probe | ⬜ | GPU + ORT `CUDAExecutionProvider`, RTSP reachable (per cam), UDP bus alive + fresh, dashboard returns 200, Backbone process up, latency p95 < 200 ms, calibration present for the mode, free VRAM/RAM/disk |
| `system_health` test (hermetic subset in CI + an opt-in `--live` target) to gate deploys | ⬜ | consolidates `tools/rtsp_smoke.py` · `detection_smoke.py` · `latency_probe.py` |
| Image-build CI + version tags | ⬜ ⏸ | |

---

## Part H — Deferred / out of v1 scope — ⏸

| Item | Notes |
|---|---|
| **S5.5 pose-mode 3D** (`yolo_onnx_pose` + `triangulate_keypoints`, `keypoints_xyz`) | for fall-detection; ABC seam exists |
| **N-camera (≥3) triangulation** via aniposelib `CameraGroup` | seams stubbed in `opencv_dlt`/`cross_cam_fusion` |
| Full-frame **SAHI tiling** (vs the targeted zone-patch we have) | generalization of zone patches |
| **Jetson Orin NX port** | same `.onnx`/contract; only the conda `onnxruntime` variant differs — deploy job, not code |
| Consuming **modules** (Sécurité, Palettes, Rayonnages, PLC/WMS gateway) | separate processes; consume the UDP/JSON contract only |

---

## 📓 Session log — 2026-06-10 → 11 (faced & achieved)

**3D map: Pixi → Three.js** ✅ core
- Replaced the Pixi floor map with `static/js/floor_map_3d.js` — vendored `three.module.min.js` + `OrbitControls` via an air-gapped **import map** (`base.html`). Tilted perspective + OrbitControls (zoom / pan / constrained orbit), extruded translucent zones (danger pulse), per-class 3D track primitives (capsule / box / cylinder) + billboard labels + lerp motion + `single_view` dim, **render loop pauses when hidden**, raycast `screenToWorld` shim. Keeps the `window.__floor_map` contract.
- Default framing enlarged on request: `FILL` 0.8 → **1.5** (floor fills the panel like the CAM view) + tilt 58°.
- **Richer 3D `warehouse_map`:** real **3-level white racks** (corner uprights + shelf planes; parametric footprint + `height_m` + `levels` + `rotation_deg`), optional wall slabs, cyan work-area outline. Schema gained `levels`/`rotation_deg` (`warehouse_map.py`). Pallets/people are *tracked objects*, not map structures. Seeded 2 demo racks.

**Training & model (yolo26l-seg)** ✅
- Post-reboot recovery: checkpoints safe; training had reached **epoch 172/200** (converged, ~natural early-stop). Ran a **finalize pass** from `best.pt` (no retrain) → val plots + `results.png` + confusion matrix + PR/F1 curves + `report.md` + **ONNX (opset 17) + OpenVINO**. Metrics: **box mAP50 0.977** / 50-95 0.948, mask 0.972 / 0.921.
- Verified YOLO26 NMS-free **end-to-end seg head** `(1,300,38)` decodes via `yolo_onnx_seg` (Backbone-compatible) — CUDA smoke test.
- **imgsz-slider fix:** re-exported `best.onnx` with `dynamic=True` (was static 640 → slider inert). See memory `imgsz-slider-needs-dynamic-onnx`.
- **Model-folder rename:** all run dirs → `model-type_e<ep>_<imgsz>px_<time>` (RF-DETR without time); fixed config refs + a latent **relative-path bug** in `config.py`.

**Crash & performance — the CUDA-700 firefight** ✅
- Diagnosed the app freeze: stacked CUDA sessions (Backbone + cam preview + pose + per-zone) exhausted the 12 GB card → `CUDA 700 illegal memory access` corrupts the context (unrecoverable). See memory `gpu-vram-budget-12gb`.
- **GPU-memory guard** (`detection_overlay.gpu_inference_safe`): preview skips a frame when free VRAM < 900 MB, yielding to the Backbone.
- **`capture_fps`** input cap (`rtsp.py` `videorate drop-only`; cam_a = 12) — throttles both the Backbone's inference rate and the dashboard decode.
- **Zone-based detection on cam1:** cam1 now detects from the light **320 zone crops** mapped back to the full frame + runs **pose-only** full-frame — drops the heavy 640 full-frame detector (big VRAM cut). Object detection is the sum of cheap zone crops; cam1 shares the `(model, 320)` session with the zone panels.

**Config unification (Phase 1–3)** ✅
- Merged the 4 dashboard config files (`link_lines`, `warehouse_map`, `zone_patches`, `danger_zones_object`) into one sectioned **`monitor_web_ui.yaml`** via `dashboard_config.py` (auto-migration + back-compat). Backbone-contract files (`backbone.yaml` / `zones.yaml` / `calibration.json`) stay separate.
- Added `tests/conftest.py` so tests never touch the real config (after a one-time pollution I cleaned).
- **Per-zone config (Phase 3):** metric zones carry name + kind/severity + **model + confidence**, editable in Settings, persisted to `zones.yaml` (Backbone's `ZoneRegistry` ignores the extra keys).

**Zone-patch redefinition** ✅
- Cam-space zone patches are now **polygons** (red overlay on cam1) instead of rects; the polygon's **bounding rect** is cropped → **resized to 320 with `INTER_AREA`** → detected at a per-zone `infer_size` (320 default). Zone detectors built per `(model, size)`.

**Pixi.js fully removed** ✅
- Deleted `floor_map.js`, `floor_map_avatars.js`, `vendor/pixi.min.js`; dropped the Pixi `<script>` from `base.html` (kept the three.js import map). Three.js is now the only map engine.
- Rewrote `draw_mode.js`'s **MAP** branch Pixi → Three.js: raycast `transform.screenToWorld`, `previewPolygon`/`clearPreview` on a Three.js `previewGroup`, OrbitControls disabled while drawing (re-enabled on cleanup). Restores zone/rack drawing on the 3D map.

**Warehouse-layout editor redesigned (3D racks, simple, map-only)** ✅
- New `layout_manager.js`: draw a rack/wall footprint on the 3D floor + a minimal form (height / levels / rotation / label); no camera/opacity/underlay. The editor only shows in **MAP view** (Alpine effect on `$store.bigPanel.view`).
- Removed the 2 **demo racks** + shrank 3D track primitives ~40% (`geometryForClass()`) per the map plan.

**Person ↔ pallet distance overlay** ✅
- Each pallet bbox → **4 edge-midpoint nodes**; distance to the person **foot node**; only the **single nearest** line per pallet is drawn. Object foot-nodes removed — only the person foot node is shown.
- Pose runs full-frame; **confirmed pose never runs inside zone crops**.
- Line is **UI-configurable** (opacity / color / thickness) in Settings → `distance_line_*` in `monitor_web_ui.yaml`; `annotate_frame`/`draw_person_pallet_distances` honor the style.

**Memory-safety: orphan reaping + STATUS resources** ✅
- Diagnosed **"Killed" = host-RAM OOM** (12 GB WSL cap; CUDA pinned memory is unswappable). Cause: orphaned `backbone.runtime` subprocesses (~1.5 GB each) reparented to `/init` accumulating across restarts. See memory `killed-is-host-ram-oom`.
- `BackboneSupervisor._reap_orphans()` (`pgrep -f backbone.runtime … → SIGKILL`) runs **before** each START; `_free_memory()` (`reset_detector` + `gc.collect` + `malloc_trim`) on both reap and **STOP**.
- **STATUS panel** now shows **GPU/CPU/VRAM "used / total MB"** + GPU util (`hardware.host_memory_mb`/`gpu_utilization_pct` → `/api/status.resources` → `ws_tracks.renderStatus`, red >90 %).

**Detection gating + cam1↔zone de-duplication** ✅
- Before **START**: cam views show **raw feed only** (no preview, no pose) — AMBER; after START the Backbone + all detection/pose/distance run. Per-frame `is_running` gate in `routes_video.py` iterators.
- **De-dup:** cam1 stops its own object detection once a zone is declared — it **reuses the zone's cached detections** (`_ZONE_DET_CACHE`, one inference per zone, remapped to the full frame). Pose still runs.
- **STRICT zone-only rule (no double-draw):** the instant a camera has ANY zone, its main view runs **no** full-frame object detector — objects come **solely** from the zones, and the only full-frame inference is **human pose**. `_zone_objects()` drops person-class zone boxes (humans = pose only, so no box-AND-skeleton) and **dedupes overlapping-zone duplicates** by class+IoU (highest conf wins). No zones ⇒ full-frame detection returns.
- **Humans NEVER detected in zones:** `_drop_persons()` strips person-class right at the zone detector (`_zone_patch_iter`), so neither the zone panel nor the `_ZONE_DET_CACHE` (→ cam reuse) ever carries a human box. Humans appear only on the cam views, via pose. +4 tests (`_zone_objects` / `_drop_persons`).

**Zone-patch UX polish** ✅
- ZONE slice shrunk to **85 %** (`.zone-patch-img`); zone polygon on cam1 is a **dashed red outline, no fill**.
- Zone **syncs to Settings**: a "Camera zones" list (`#zm-cam-zones`) shows each zone's **name + position `@(x0,y0)` + `W×H px · Npts`** + per-zone model + detect size; the ZONE **panel turns light-green** (`.zone-synced`) once declared; **auto-saves** on creation.
- **Zone panel/Settings refinements:** slice now **centred** in the panel body (flex-center); model + detect-size selectors **removed from the panels** (they live only in Settings now); Settings row gained a **per-zone outline colour** picker (`PatchRect.color`; `live_overlay` strokes `p.color || red`) and shows the **full saved polygon vertices** beneath each row (`.zm-coords`).
- **Per-zone confidence box:** each Settings zone row has a **conf input** (`PatchRect.confidence`, blank = global). Plumbed end-to-end through `_zone_patch_iter` → `get_zone_detector`.
- **Per-zone delete, no clear-all:** removed the "clear all zone patches" trash icon from the big-view toolbar (function retired with it); each Settings zone row now ends with its own **delete icon** (`deletePatch(id)` → filters + auto-saves → panels/worker resync via the save hook).
- **3 zone panels + add/delete UX:** zones grid is now **3 panels** (`zones-grid-3`, `zone3` tall bindings); an undeclared slot shows a clickable **"+ Add zone"** placeholder (switches to CAM 1 if needed, then starts the draw). Zone-panel tall toggle got a **scale pop animation** (`zone-pop`/`zone-pop-tall` keyframes — restarts on class swap so it pops both up and down; `prefers-reduced-motion` respected). **Delete compaction:** deleting any zone shifts the rest up to fill its slot (panels render by index) and **default names ("Zone N") renumber** to their new position — custom names untouched.
- **3D map = realtime detection twin** (`/api/map/twin` + map twin layer): the map now mirrors the cam view's OWN detections instead of (only) Backbone tracks. The zone worker snapshot gained **full-frame pose people + frame_wh**; the new endpoint projects objects' foot points, people, and the **zone-patch polygons** pixel→floor through the current calibration (points rescaled to calib size → undistort → H). **Mode 1 = cam_a mirror; Mode 2 unions both cameras in the shared world frame (the unified view)**. `floor_map_3d.js` polls it at 300 ms (paused when hidden), renders objects/people via a greedy nearest-neighbour id matcher (stable ids → smooth lerp), draws zone outlines as **dashed floor lines in each zone's colour**, folds them into the camera framing, and **suppresses the Backbone-track layer while the twin is fresh** (no double objects). 3D bodies shrunk a further ~25% + smaller labels (small + spacious). Degrades to `available:false` without calibration (+1 test). **Live-verified:** 2 palettes at metric positions + both zone outlines in the twin feed, 0 errors; suite 217 green.
- **Map framing/labels re-fix (screenshot review):** `recomputeBounds` no longer pins a ±5 m floor — bounds **hug the content** (zones/twin/layout + 0.8 m pad, min 3.5 m span): workspace 12×12 m → ~4×3.5 m, content fills the panel (FILL 1.5→1.15 accordingly). **Stale Backbone-track ghosts purged** when the twin takes over (the early-return had skipped GC → "#1 · empty" lingered next to twin pallets). Labels shrunk 0.4→0.22 m with tighter offsets, twin tags are **class-only** (conf % jittered the texture every poll).
- **Panel expand rework + sidebar swap:** **STATUS now above LOGS** in the sidebar (same expand-one/collapse-other logic, icons switched to `<>`/`><` chevrons). New **in-column tall expand** (`$store.bigPanel.tallPanel` + `toggleTall`): the big panel and each ZONE panel have a **`<>` button** that grows that panel to the FULL left-column height while hiding its column siblings (sidebar untouched; tall zone spans both grid columns, its slice goes 85%→100%). **Only the big panel keeps the `[]` full-width overlay** in addition to `<>`; the zones' old centered-overlay pop (`expandable` + teleport backdrops) was removed. Doc `docs/zone_detection_transform.md` rewritten for the worker architecture (snapshot diagram, no-shaky/no-dual section, "no per-zone processes", updated code map).
- **Conf is a POST-FILTER, not a cache key** (fixed a freeze): keying the zone detector by conf rebuilt a multi-second yolo26l-seg CUDA session on every conf change (and leaked the old) → UI froze. Now **one session per `(model, size)`** built at a low floor; each zone's conf filters the returned dets. **Verified live**: conf change = 0.035s, session builds stay at 1, GPU flat. Added a **build lock** (double-checked) so concurrent first-access (both zone panels + cam reuse) builds once, not N racing sessions. Heartbeat now reports "zone detection active: N session(s)" instead of the misleading "no preview run yet".
- **Retired the metric-zone editor:** removed the Settings "Zones (up to 6)" map-click panel + `zone-row-template` + all of `zone_manager.js`'s zone-row code. Operator zones are now **only** the CAM-drawn patches under "Camera zones (drawn on CAM)". `/api/config`'s `zones` is now **optional** — the dashboard omits it, so Save **leaves `zones.yaml` untouched** (no accidental wipe). **Cap of 6** enforced client-side (draw blocked + button disabled at 6) and server-side (`PatchesBody.patches max_length=6`, `ConfigPayload.zones max_length=6`). +1 test (omit-zones preserves zones.yaml).

**Whole-app freeze (GPU fine) — threadpool starvation** ✅ (3 parallel agents + live probe)
- Every MJPEG stream is a **sync** generator → Starlette pumps each in the AnyIO worker pool, **one thread held per open stream**, default pool **40**, shared with every `run_in_threadpool` (`/api/status`, saves, control). Accumulated stream connections saturated it → all requests blocked until the tab closed (also why the last setting didn't save). **Fix:** raise the limiter to **256** in the lifespan (`app.py`). **Verified live**: 24 concurrent streams open, `/api/status` stayed 37–43 ms.
- **Zone duplicates were geometric, not the shared session.** The shared CUDA session is stateless (sandboxing it changes nothing). Each zone detected everything in its bounding **rect** (dets never clipped to the **polygon**) so overlapping zones double-reported; and cam1's `_same_object` used `a_in_b AND b_in_a` (failed on offset/clipped twins). Fixed: polygon clip + strengthened merge (IoU>0.5 OR either-centre-contained OR intersection/min-area>0.6). +2 tests.

**Zone-detection architecture revamp → background `ZoneDetectionWorker`** ✅ (fixes the duplicate/shaky seg mask)
- **Root cause of the second shaky mask:** detection was HTTP-driven — each `/stream/zone/{id}` connection detected independently and cached with its **own timestamp** (cam1 merged entries up to 1.5 s apart → fresh + stale copies of one moving object at offset positions); INTER_NEAREST mask upscale from the 320px crop = the shake.
- **New (`monitor_web/zone_worker.py`):** one daemon thread per camera runs ALL zones on the SAME frame, resolves cross-zone overlaps at publish time (deepest-polygon-centre wins, `pointPolygonTest(measureDist)`; each object lands in exactly ONE zone), publishes ONE atomic snapshot `{frame_ts, zones}`. `/stream/zone/{id}` panels + cam1 are now **pure renderers** of that snapshot — zero detection in any HTTP path. Masks remap **INTER_LINEAR + 0.5 threshold**.
- Worker **idles before START** (no GPU, camera stream released, raw feed preserved); `ZoneWorkerManager.reload()` hooks on zone-save + config-save (topology: new cam ⇒ new worker, last zone gone ⇒ worker stopped, source change ⇒ stream re-acquired); **zones 3–6 now detect** (previously only the 2 panelled ones ever ran); stale snapshot (>1 s) ⇒ consumers draw nothing, no ghosts.
- **"Does each zone need its own process?" — No.** A CUDA context costs ~0.5–0.8 GB each on the 12 GB card and the session is stateless; sandboxing is by data + single ownership (one writer thread), not by process.
- +11 tests (`tests/test_zone_worker.py`); **suite 216 green**; live-verified: 0 detector loads while Backbone stopped, exactly **1** after START with both panels + cam1 open simultaneously, zone-save reload 0.05 s, 0 errors, GPU steady 2.1 GB.

**Pending / paused from this session:**
- ✅ **favicon 404 silenced** — `app.py` serves `GET /favicon.ico` → `204 No Content` (no more log clutter).
- ✅ **Orphan reap at dashboard boot** — `BackboneSupervisor.reap_orphans_on_boot()` called in the lifespan startup, so a previous session's OOM-killed Backbone (confirmed PID 230561, 1.5 GB, 3 h) is killed before the new dashboard can be OOM-killed by it. (START still reaps before every spawn.)
- ✅ **Hardened reaper (`/proc` scan, kill on START *and* STOP)** — replaced the fragile config-scoped `pgrep` with `_find_backbone_pids()` reading `/proc/<pid>/cmdline` directly: finds **every** `backbone.runtime` host-wide (any config), immune to pgrep regex-escaping / arg-length truncation, always excludes the dashboard's own PID. `_kill_backbones()` SIGKILLs them (instant, never blocks the event loop). START reaps before spawn; **STOP now also sweeps strays** after terminating its tracked child, so STOP leaves a truly clean slate. +2 regression tests (decoy under a *different* config is found+killed; self never targeted).
- ⬜ `danger_zones_object` has no Settings editor (defaults only).
- ⏸ Walls in the warehouse editor (optional); STOP-time state snapshot (deferred by user).

All suites green throughout: **backbone 370**, **monitor_web 216** (as of the 06-11 zone-worker revamp).

---

## 📓 Session log — 2026-06-11 (isiGen)

**`trainer/isiGen` — synthetic dataset generator (SD 3.5 Large + ControlNet), foundation + phases 1–3** ✅
- New trainer-world package mirroring isidet's conventions: **8 ABC seams, each with its own Registry** (copy of the Backbone's canonical `Registry`): ControlMapExtractor (`canny` ✅, `depth_anything_v2` ✅) · Masker (`sam2` ✅) · Captioner (`template` ✅) · LoraTrainer (`diffusers_sd3` stub) · ScaffoldSource (`depth_remix`/`box3d_procedural` stubs) · ImageGenerator (`sd35_large_controlnet` stub, NF4+offload recipe locked in docstring) · QualityFilter (`clip_score` stub) · DatasetExporter (`yolo_seg`/`labelme` stubs). Stubs raise `NotImplementedError("…lands in the next session")`; their `project.yaml` config keys are reserved & final.
- **Project model:** `data/<project>/project.yaml` (classes = name+trigger+color, unique-validated; per-phase params) + **`manifest.jsonl`** (pydantic, `extra=allow`, atomic save) — every phase reads/extends it; Studio renders from it.
- **Phases working:** curate (sha256 dedupe, EXIF-strip, idempotent) → maps (canny pure-cv2; DepthAnythingV2 via transformers pipeline; SAM2 prompted-from-Studio + auto-fallback, color-composited GT mask) → template captions (deterministic per id, trigger+phrase+background bank, edited captions never overwritten).
- **isiGen Studio** (FastAPI **:8200**, `ISIGEN_` env prefix, monitor_web patterns): Projects / Phase board (8 cards + live job log) / Curate gallery (retag+exclude) / Maps 4-up viewer with **SAM2 prompt canvas** (click=+pt, shift=−pt, drag=box) / Caption editor. **JobRunner** = 1 worker thread (one GPU job at a time), per-job log ring buffer teed to `runs/jobs/`; job bodies are the same `core/runners.py` functions the CLIs call.
- **Env (user's call — reuse `isi-train`):** it already carries torch 2.10+cu128 (Blackwell sm_120 verified) + transformers/accelerate/peft/fastapi; only **diffusers 0.38, bitsandbytes 0.49, sentencepiece, pydantic-settings, pytest + SAM2 (no-build-isolation git pip)** were added — ultralytics still imports fine. The duplicate `isi-gen` env + its yml were **deleted**; instead the repo root now has the **recreate spec for isi-train itself**: `isi-train.yml` (conda layer = python 3.11 + CUDA env vars) + `requirements-isi-train.txt` (exact pip freeze, 193 pkgs, cu128 extra index, SAM2 pinned to a git sha). SD3.5 gated → HF login documented.
- **Verified:** 32 hermetic tests green under isi-train; ruff clean; demo project from real isidet pallet images — **phase 1–3 fully live**: 10 curated (rerun=0) → 10 canny + **10 DepthAnythingV2 depth maps** + **10 SAM2 masks** (auto path → `needs_review`; then a Studio box prompt → **prompted mask painted in the exact class color**, resumable runner redid only that record) → 10 anti-bleed captions. Studio smoke: 5 pages 200, status counts, JobRunner job with live log, thumb/canny media. Stub CLIs fail with the clear next-session message. One live-run bug fixed: SAM2 rejected negative-stride BGR→RGB views (`np.ascontiguousarray`).
**isiGen phases 4–8 implemented (same session, after HF token landed)** ✅ code-complete
- **P6 scaffolds (live-verified):** `box3d_procedural` — analytic depth (bright=near, DepthAnythingV2 convention) + perfect class-color masks, back-to-front stacks (pallet slab + cartons + polybag-wrapped loads); `depth_remix` — paired affine jitter of the REAL phase-2 depth+mask artifacts (depth LINEAR/replicate, mask NEAREST/black so class colors stay exact). Runner materializes `scaffolds/<id>_{control,mask}.png` + `index.jsonl` (status pending→generated, resumable). Demo: 6 pairs, both sources, multi-class metas.
- **P5+7 generator (`sd35_large_controlnet`):** real implementation of the locked recipe — SD3ControlNet + NF4 transformer (`diffusers.BitsAndBytesConfig`), T5-XXL nf4/none escape hatch, `load_lora_weights`, `enable_model_cpu_offload`; `generate()` = depth map → RGB PIL → pipe → BGR. `run_generation` builds anti-bleed prompts from scaffold classes (triggers + phrases + background bank), mints per pending scaffold, saves `generated/`, adds **synthetic manifest records whose mask is the scaffold GT** (aligned by construction), index+manifest saved after EVERY image (resumable).
- **P4 LoRA (`diffusers_sd3` QLoRA):** memory-disciplined custom loop for 12 GB — precompute prompt embeds (pipeline `encode_prompt`, T5 nf4) → free encoders; precompute VAE latents → free VAE; NF4 transformer + PEFT LoRA (to_q/k/v/out.0) + grad checkpointing + 8-bit AdamW; flow-matching objective per diffusers' SD3 script (logit-normal sampling, `target = noise − latents`); saves `pytorch_lora_weights.safetensors` (pipe-loadable) + report.md.
- **P8 (live-verified):** `clip_score` filter (CLIP ViT-B/32 cosine; runner scores synthetics, excludes < min_score); `yolo_seg` exporter — color mask → per-class contours → simplified normalized polygons → images/labels/{train,val} + data.yaml (stable hash split) — **demo: 10 real records exported, labels valid (cls + coords∈[0,1])**; `labelme` exporter — polygons → X-AnyLabeling-ready JSONs (10 written).
- Studio: all 8 phase cards now runnable (jobs router + status counts: scaffolds pending/generated, synthetic, clip_scored, exported). CLIs rewired to `core/runners`. Suite **34 green**, ruff clean.
- **Pending GPU verification:** SD3.5-Large + ControlNet weights downloading in background (~25 GB, `/tmp/isigen_weights.log`) → then smoke-mint 1 image (`run_generate --limit 1`) and a short LoRA run.

---

## 📓 Session log — 2026-06-12 (git + dashboard architecture rework)

**Version control — repo live on GitHub** ✅ *(closes G.1 / priority item 4)*
- gh CLI 2.94 installed user-local (`~/.local/bin`, no sudo), browser-flow auth as **waledroid**, git identity set to the GitHub noreply form. `git init -b main`, `.gitignore` extended (trainer `runs/data/logs/models` ≈28 GB, `archive/` 2.1 GB, all `*.pt`, test caches, `.venv-multical`, WSL `:Zone.Identifier`, `.claude/` local files) — **386 files / ~11 MB** committed, rebased onto GitHub's auto-README, pushed to **github.com/waledroid/isi_monitor3d** (`main`, public; spec PDFs + live configs incl. the LAN rtsp URL included by explicit choice). LFS skipped deliberately: weights are ignored, repo stays code-only.

**Cam views POSE-ONLY — "no zones → full detection" removed** ✅
- `_detect_iter` (routes_video.py) lost its fallback branch: the big CAM views **never** build the full-frame detector now. Once the Backbone runs, the only full-frame model is **pose**; object boxes come solely from the zone-worker snapshot (no zones ⇒ skeletons only). `get_detector`'s remaining consumers: MP4 dev viewer + warp verification. +3 tests (`test_detect_iter.py`) pin it (monkeypatched `get_detector` raises if touched).

**Zone isolation guards (in-process — subprocess sandboxing explicitly rejected)** ✅
- Decision discussed & locked: per-model subprocesses would cost ~0.5 GB CUDA context EACH on the 12 GB card + IPC + the orphan-reaping problem class; the dominant failure (CUDA-700) is *preventable over-admission*, not a random fault. Escalation path if it ever still bites: ONE supervisor-restarted inference subprocess, never per-model.
- **VRAM admission:** `get_zone_detector` raises `ZoneModelUnavailable("no_vram")` instead of building a session when free VRAM < 1.5 GB (ORT OOM mid-build = CUDA-700 = every session in the process dies).
- **Per-zone circuit breaker** (`zone_worker.py`): a zone whose detector fails (refused / build / inference) is disabled 30 s then retried — **other zones keep detecting**; status `ok|no_vram|error` published in the snapshot and drawn as a red banner on the zone panel (`_zone_render_iter`). Zone-save clears the breaker.
- **Per-zone `max_fps`** (`PatchRect.max_fps`, config-only): budgeted zone re-infers only when due, last dets carry forward — a heavy RF-DETR zone at 2–4 fps no longer drags the light zones below display FPS. +5 tests (`test_zone_isolation.py`).

**Settings-save freeze — root-caused & fixed structurally (MJPEG → one WebSocket)** ✅
- **Root cause:** up to 7 simultaneous MJPEG `<img>` streams (hidden big-panel tabs kept streaming under `x-show` + 3 zone panels) exhausted the browser's ~6-connection HTTP/1.1 cap → the settings POST/response + reloads starved → "frozen until a new tab". Secondary: the POST handler blocked the event loop (gc.collect 0.5–2 s, onnx.load, fsync, worker reload).
- **New transport:** all dashboard video over ONE multiplexed **`/ws/video`** (`routes_ws_video.py`): client subs/unsubs JSON, frames binary `uint8 idLen | stream-id | JPEG`; ids `cam:<id>` / `cam:<id>:warp` / `zone:<patch_id>` / `unified`. Server reuses the SAME sync pipelines (extracted `build_cam_stream`/`build_zone_stream`/`build_unified_stream`) in per-subscription daemon threads with a one-slot drop-oldest holder. MJPEG endpoints kept for curl debug + MP4 dev tab.
- **Client:** `static/js/video_ws.js` (`window.__videoWS`, first deferred script in head) renders blob URLs into the existing `<img>`s — CSS/expand untouched; auto-reconnect w/ backoff; **resubscribes on `config:saved`** so model/camera changes apply live (replaces the `streamNonce` URL juggling; only the visible panel holds a stream). `big_panel.js` effect + `zone_patch.js` rewired.
- **Event loop unblocked:** `post_config` + `post_zone_patches` are now deliberately **sync `def`** handlers → FastAPI runs their blocking tails in the threadpool.
- **Verified live (:8100, sandboxed config):** one socket carried cam + zone simultaneously, unknown-stream JSON errors, unsub stops one stream while the other flows; `POST /api/config` → **200 in 23 ms with streams open**, frames kept flowing after the save; `/api/status` responsive throughout; server log clean. +4 WS tests (`test_ws_video.py`).
- One pre-existing test-pollution gotcha fixed: `test_detector_selection` reloads `detection_overlay` mid-suite, rebinding exception classes — the new isolation test references `overlay.ZoneModelUnavailable` through the module.

**Docs:** CLAUDE.md dashboard section rewritten (WS transport, pose-only cam views, zone isolation, script-order constraints, MP4 viewer now `get_detector`'s main consumer).

**Settings cleanup (same day, follow-up)** ✅
- **"Modèle de détection" is pose-only now:** removed the ONNX model picker, OpenVINO .xml field, auto-detected classes, confidence threshold and the imgsz slider from the modal (object models are per-zone). Save sends a new lightweight `pose` payload — `post_config` splices ONLY `pose_onnx_path`/`pose_confidence_threshold` into the detection block, never the object-model keys (+2 tests).
- **Display prefs auto-save:** show nodes/masks/boxes toggles + distance-line opacity/colour/thickness now POST `/api/ui-settings` on change (overlay reads prefs per frame ⇒ instant effect, no Save button). `post_ui_settings` made sync `def` (threadpool).
- **Per-zone FPS in the UI:** each Settings ▸ Zones row gained an `fps` box (`PatchRect.max_fps`; blank = global). First attempt was invisible — the row's CSS grid had exactly 8 column slots, so the 9th input wrapped off-grid; fixed by moving the dims text (`@(x,y) W×Hpx · Npts`) into the small `.zm-coords` line as `slice: … polygon: …`, freeing its column.
- **General FPS box (Cameras tab):** new auto-synced `display_fps` field (1–30, default 10) — caps cam streams, unified view + the zone worker tick; per-zone fps can only go BELOW it. EN/FR labels updated.
- **Zone label on cam view** now anchors to the polygon's most-top-right VERTEX (max x−y), hugging the actual zone lines — not the floating bounding-box corner (was: first-drawn vertex).

**Blinking detections — root-caused & fixed (anti-blink)** ✅
- The zone snapshot expired after a hard 1.0 s (`SNAPSHOT_MAX_AGE_S`); two trips made the overlay boxes flicker: (a) cam1 delivering no NEW frame >1 s (RTSP jitter @ capture_fps 12) while the worker stayed silent instead of re-confirming; (b) at zone fps 25, full re-infer passes + GPU contention with the Backbone pushed the publish gap past 1 s.
- Fix: worker ts-bumps the snapshot while the frame is unchanged (the panels show that same held frame, so its detections remain correct), and each snapshot carries `valid_s = max(1.0, 2.5×pass_duration)` so slow inference can't expire its own result. Dead worker still clears in ~seconds (no ghosts). +2 tests.
- **Advised defaults:** General FPS 10 (camera captures at 12); light zones blank (= global); heavy zones (RF-DETR) 2–4. Per-zone values above the general cap have no effect.

**STOP memory — measured + quick win** ✅ (full fix planned)
- Measured the STOP-path release (real CUDA sessions, `reset_detector`+gc+malloc_trim): session VRAM DOES free (283 of 329 MB), but **~1.24 GB host RSS never returns** — CUDA context + cuDNN/cuBLAS/ORT library mappings are unreleasable in-process by design.
- **Generator-pinning leak fixed:** suspended cam-view generators kept their last running iteration's locals — `pose` (CUDA session) + dets' full-frame masks in `_detect_iter`, `detector` in `_warp_detect_iter` — pinning them after STOP for as long as the panel stayed open, defeating `reset_detector()`. Now the stopped branch drops the refs (+1 test inspecting `gen.gi_frame.f_locals`).
- **Agreed full fix (to schedule):** ONE dashboard-supervised **inference subprocess** owning all GPU work (zones + pose + MP4 preview) — STOP/idle kills it ⇒ OS-guaranteed 100% VRAM+RAM release while the UI stays open; also delivers the CUDA-700 isolation escalation + the long-open "detector idle-release". Costs: ~0.5 GB extra context while running, crop IPC, ~3–5 s warmup on START.

Suites green: **backbone 370**, **monitor_web 235** (was 216 at session start). On-rig items unchanged (pose overlay quality + VRAM banner under real GPU pressure → Part D).

**isiGen generation stack: SD3.5-Large → SDXL + depth ControlNet** ✅ *(same day)*
- The stalled ~20 GB SD3.5-Large download was cancelled; SD3.5 **Medium** was requested but has **no depth ControlNet** (Stability ships ControlNets for Large only) — user chose **SDXL + `diffusers/controlnet-depth-sdxl-1.0`** instead (~9.5 GB fp16, ungated, faster, cheaper LoRA on the 12 GB card). Design: `docs/superpowers/specs/2026-06-12-isigen-sdxl-design.md`.
- New `sdxl_controlnet` generator (fp16, `madebyollin/sdxl-vae-fp16-fix` VAE, no quantization knobs, cpu-offload default) + `diffusers_sdxl` LoRA trainer (same 3-stage memory discipline; DDPM epsilon objective + SDXL `add_time_ids`; UNet-only LoRA fp32 via `cast_training_params`; default res 768). SD3.5 modules **deleted** (never GPU-verified; in git history). Configs/template/demo project/runner defaults/Studio labels/README updated. Suite **35 green**, ruff clean. SDXL weights downloaded (fp16-only includes). CLIP ViT-B/32 weight download completed too (was a stale fragment).
- **GPU smoke mint ✅ live-verified:** `run_generate --project pallets_demo --limit 1` → pipeline loads from the fp16 cache, 30 steps in **29 s** (~2.5 it/s on the 5070), coherent 1024² warehouse/pallet image (depth-forced geometry, no NaN). **Phase-4 LoRA live-verified too** (100-step smoke on the demo's 10 images): embeds/latents precompute + free, 23.2M trainable (r16 @768), loss 0.143→0.10, ~7 s/step (2000 steps ≈ 4 h), peak 7.6 GB VRAM; weights + report saved to `runs/lora/`, and a second mint with `lora_weights` set exercised `pipe.load_lora_weights` end-to-end. **All 8 isiGen phases now GPU-verified.** Production LoRA (2000 steps on a real curated set) still to run.

---

## ▶ What's actually left, prioritized

1. **✅ yolo26l-seg trained + exported + wired** (2026-06-10): finalize pass → dynamic ONNX + OpenVINO; `backbone.yaml` detection now points at `yolo26l-seg .../best.onnx`. Remaining is the on-rig KPI check (item 2). *(Part B/D)*
2. **🔬 Mode 1 on-rig validation:** ≥5-point recalibration + KPI check (latency/reproj/mAP/occupancy) + sustained live run. *(Part D)*
3. **🔬 Mode 2 bring-up:** mount cam_b → `calibrate-all` joint calibration → validate unified view + 3D + single-view fallback on the rig. *(Part E)*
4. **✅ Version control & repo** (2026-06-12): repo live at github.com/waledroid/isi_monitor3d (`main`); weights gitignored instead of LFS. *(Part G.1)*
5. **⬜ Single inference subprocess** (agreed design): all dashboard GPU work in one supervised worker process — STOP/idle kill ⇒ total VRAM+RAM release with the UI open; subsumes detector idle-release + CUDA-700 isolation escalation. *(Part C/F)*
6. **⬜ Deployment & ops:** Docker images (Backbone + dashboard) + `docker-compose` + `ops/` launch scripts + **`ops/system_health.sh`** go/no-go probe; container `HEALTHCHECK` / graceful restart; trainer prune (post-training). *(Part D / F / G.2)*
7. **⏸ Later:** S5.5 pose-3D, N-cam aniposelib, full SAHI tiling, Jetson image. *(Part H)*

**Bottom line:** the software for **Mode 1 is complete** (pending model swap + on-rig KPI sign-off), and **Mode 2 is
code-ready** (pending the physical second camera + joint calibration). The remaining work is mostly **validation on
real hardware**, not new code.
