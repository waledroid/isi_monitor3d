# CLAUDE.md

Guidance for Claude Code in this repository.

## Source of truth

- **Spec (French):** `docs/specs/Cahier des Charges-Système de Vision Industrielle.pdf` — customer requirements, KPIs, sprint plan.
- **Approved build plan:** `~/.claude/plans/starry-stirring-fairy.md` — sprints, file structure, plugin seams. When spec and plan disagree, ask before deviating.
- This repo is the **Backbone**. Modules (Sécurité, Palettes, Rayonnages, Dashboard, PLC/WMS gateway) live in separate processes and consume only the UDP/JSON + MQTT contract.

**Status.** Backbone v1 is feature-complete (S0–S8: core, shared utils + calibration schema, Multical backend, conda env, ingestion, ONNX detection, homography, triangulation, comms + orchestrator, simplification pass, operational modes). Remaining: on-site validation with the real rig, Isitec-specific training, deferred S5.5 pose mode. Jetson Orin NX port needs no code change (`.onnx` and Python are portable; only the `onnxruntime` wheel differs).

## Module reuse

Every module app (isical, isistream, isiGen, isidet, **isicomms** = MQTT broker + gateway) exports to a self-contained folder via `scripts/export_module.sh <module> <dest>` — see `docs/REUSE.md`. The shared core travels as a wheel built at export time. isicomms' REST/MQTT surface and `ISI_GATEWAY_*` env prefix are frozen interface; it serves a probe UI at `:8080/ui` and Swagger at `/docs`.

## Commands

```bash
# installer (stage-by-stage, resumable; installs Miniforge itself on a clean PC)
./install.sh gpu                                  # main branch; `cpu` on the cpu branch
./install.sh gpu --env mysite                     # use (or create) conda env "mysite" instead of monitor3d / monitor3d-cpu;
                                                  #   the 3d alias + systemd units follow the name; ENV_NAME=mysite also works
./install.sh gpu --dry-run | --list | --only STAGE | --skip STAGE | --systemd
# (the `3d` / `3d_cpu` alias runs only from inside the repo — cd there first)

# environment by hand
conda env create -f environment.yml -n monitor3d
conda activate monitor3d                          # before every command below
conda env update -f environment.yml -n monitor3d --prune   # after env edits
pip install -e ".[dev,geometry,schemas]"          # pip-only fallback

# calibration
bash calibration/setup_multical.sh                # one-time Multical venv (Mode 2); AprilGrid needs: sudo apt install -y cmake libopencv-dev libeigen3-dev
python -m calibration.calibrate gen-boards --output-dir boards/ --n-boards 6     # A4 ChArUco + 6 AprilGrids (print 100 %, 0 margins)
python -m calibration.calibrate calibrate-all ...                                # Mode 2: joint BA, single ChArUco
python -m calibration.calibrate calibrate-2cam \                                 # Mode 2 two-stage (recommended): ChArUco intrinsics → AprilGrid extrinsics (K fixed)
    --intrinsic-dir cam_a=shots/intr_a --intrinsic-dir cam_b=shots/intr_b \
    --extrinsic-dir cam_a=shots/extr_a --extrinsic-dir cam_b=shots/extr_b \
    --floor-shot cam_a=shots/floor_a.jpg --floor-shot cam_b=shots/floor_b.jpg \
    --work-dir /tmp/cal --output calibration.json      # board sizes: --tag-length/--tag-spacing/--n-boards, --squares-x/-y/--square-length/--marker-length
python -m calibration.calibrate calibrate-2cam ... --vis          # Multical 3D viewer (needs a display); `vis --workspace /tmp/cal` re-opens a run
python -m calibration.calibrate single-cam --camera-id cam_a --image-size 1920 1080 \   # Mode 1: floor fit, use ≥5 --pair u,v,X,Y (4 = exactly determined, no residual gate)
    --pair u1,v1,X1,Y1 ... --output calibration.json

# tests / lint
pytest                                            # backbone + calibration
pytest tests/test_registry.py::test_register_and_create ; pytest -k registry
pytest --cov=backbone --cov=calibration
ruff check --fix backbone calibration tests && ruff format backbone calibration tests

# run (Direction 1: two processes, one config)
python -m backbone.runtime --config config/backbone.yaml   # metric engine (points mode: no CUDA)
python -m isistream --config config/backbone.yaml          # perception producer
python tools/latency_probe.py online --config config/backbone.yaml --seconds 60   # KPI probe
# dashboard START spawns/reaps both; headless = two systemd units. Rollback to single-process: ingestion.mode: frames + restart.
```

**Python 3.10** (JetPack 6.x). No 3.12+ syntax.

**Two environments, on purpose.**
- `monitor3d` conda env: Python 3.10, OpenCV 4.13 headless (FFmpeg-only, no GStreamer backend), GStreamer 1.28 + PyGObject, FilterPy, scipy, numpy, PyYAML, pytest/ruff, pydantic 2, CUDA 12.9 + ONNX Runtime 1.23.2 (pip `onnxruntime-gpu`) + TensorRT 10.16 (pip `tensorrt-cu12`) on the RTX 5070 (sm_120), OpenVINO (lazy import). Model backend is suffix-dispatched (`build_onnx_session`): `.onnx` → ORT CUDA fp16; `.engine` → native TensorRT (`backbone/shared/trt_session.py`, build with `tools/onnx_to_engine.py`, ~2.1–2.3× over CUDA EP, per-machine artifact + JSON sidecar). The lazy TRT-EP (`detection.trt_enabled`, `models/.trt_cache`) was retired 2026-07-23.
- `calibration/.venv-multical/`: Multical only (pins `opencv-contrib-python <=4.7`); `calibrate.py` calls its binary by absolute path. The Backbone never imports Multical.
- `isi-train` conda env: training only (ultralytics pulls `opencv-python`, which breaks monitor3d's OpenCV).

**RTSP ingest** (`backbone/ingestion/rtsp.py`): GStreamer via PyGObject (not `gst-python`, not `cv2.CAP_GSTREAMER`). Pipeline: `rtspsrc latency=100 drop-on-latency=true protocols=tcp ntp-sync=true ! {depay} ! {decoder} ! videoconvert ! video/x-raw,format=BGR ! appsink sync=false max-buffers=1 drop=true`. Codec probed once via `ffprobe` (H.264 → `rtph264depay`/`avdec_h264`, H.265 → `rtph265depay`/`avdec_h265`, default H.264). Explicit depay, never `decodebin` (races `not-linked` ~1 in 4 starts, builds a dead audio branch). One `Gst.Pipeline` + `GLib.MainLoop` per source in a daemon thread; appsink pushes into a `Queue(maxsize=1, drop-old)`.

**Capture timestamp.** `Frame.capture_ts = time.time()` at the appsink callback (lags the shutter by ~100 ms + decode, but is consistent across cameras). All latency is measured against it. Shutter-NTP alignment (`GstNetClientClock`) is a documented future seam, not needed for the KPI.

## Architecture — the non-negotiables

**Direction 1 (July 2026), `ingestion.mode` in `backbone.yaml`:**
- **`points` (deployed default):** the Backbone is a pure metric engine — no RTSP, no detectors, no CUDA, no onnxruntime import (~190 MB RSS). **isistream** (`python -m isistream`, spawned by the dashboard's `IsistreamHost`) owns capture → zone-scoped detection → pose and publishes per-camera `DetectionSetMessage`s (`detection_set`; explicit-empty heartbeat; `seq` gap counting; `config_fingerprint` drift warning) to `ingestion.points.listen_port` (9010). `backbone/ingestion/points_in.py` feeds the unchanged `FrameSynchronizer`. Persons ride the same stream (`cls="person"` + `keypoints_uv`) and echo back on observations, so the dashboard draws skeletons with zero inference (`WirePoseSource`). `metadata.images` is refused in points mode.
- **`frames`:** pre-split behaviour, Backbone owns all perception. Rollback + hermetic-test path.
- The producer is **always its own process** (in-process: ~2,200 ms/tick vs ~55 ms standalone — GIL + ORT-pool contention). `isistream/` is FastAPI-free; monitor_web only supervises it (config-save hot-restarts it).
- **Shared frame bus:** the producer writes every decoded frame to `/dev/shm/isi3d_frame_<cam>` (`backbone/shared/frame_shm.py`, double-buffered seqlock). The dashboard prefers the bus (re-checks every 5 s) and falls back to its own RTSP session when absent/stale. One RTSP session + one decode per camera system-wide. Readers: `FrameShmReader`.
- `tests/test_points_mode.py` pins that both modes produce identical Track2D streams. Points-mode KPIs: capture→publish p50 ≈ 77 ms / p95 ≈ 126 ms; VRAM ≈ 2.5 GB (was 5.2).

1. **One calibration, two queries.** `calibration.json` (per-camera `K, D, R, t` + derived `H`, `P`) feeds homography and triangulation.
2. **One identity space.** The homography tracker owns `track_id`; triangulation augments, never re-IDs.
3. **Subscription, not polling.** Triangulation runs only for tracks matching the rules at `subscriptions_path` in `backbone.yaml` (default `config/mode2/subscriptions.yaml`). Default output is `Track2D`; `Track3D` on demand.
4. **Plugin where multiplicity is real.** Exactly **five** ABC seams (`tests/test_registry.py::test_five_seams_present`).
5. **Process boundaries are contractual.** Zero Backbone imports from modules; `backbone/comms/schemas.py` is the only contract. Expand the schema, never share code.
6. **Fail honestly.** Every geometric output is gated (reprojection error, cross-camera disagreement). Bad input ⇒ no output or flagged output, never silent-bad.
7. **Industrial defaults.** systemd-supervised, no cloud, deterministic restart; latency against `capture_ts`, never publish-time `time.time()`.

### Two methods, one Backbone

**Zone-scoped detection** (`detection.scope: zones`, default): each zone polygon is projected into each camera (z=0 and z=2 m, distortion-aware) once at build, cropped, batched at `detection.zone_imgsz` (384, needs a dynamic export) and remapped (`backbone/detection/zone_scope.py`). **No zones ⇒ no object detector, pose-only Backbone.** `scope: full_frame` restores whole-frame detection. Dashboard zone patches are a display layer: `zone_detection_source: backbone` (default) renders the Backbone's per-camera `ObservationsMessage` (schema v6: boxes + occupancy + optional masks via `detection.decode_masks`); `local` runs in-dashboard per-zone inference (dev/fallback). COMMUNICATION zone cards read `/api/zone-patches/state`.

**Étagère zones** (`config/etagere.yaml`, dashboard-authored, isistream-consumed): per-camera image-space 3×3 cell grids for bin racks. isistream crops each cell (+8 % margin), batches through the 2-class `yolo26n@320` (`empty_box`/`filled_box`, end-to-end head — `is_end2end_detect_output`) and emits raw `etagere_state`; the Backbone stabilises per cell (`EtagereStateTracker`: flip ≥ 70 % of a 15-vote window, unknown decay 5 s, heartbeat 5 s) and publishes on UDP + MQTT (`{prefix}/etagere/{zone_id}`, retained). Never in `zones.yaml`. Optional `etagere:` block in `backbone.yaml`: `config_path`, `stabilize_window` (15), `stabilize_flip_ratio` (0.7), `unknown_after_s` (5.0), `heartbeat_s` (5.0). Tools: `trainer/isidet/scripts/{grid_click,etagere_dataset}.py`, config `configs/train_etagere.yaml`.

- **Homography** — always, per detection: foot point → undistort → `H` → `(X, Y)` m → cross-camera fusion + disagreement gate → ByteTrack in meters → temporal vote → `Track2D`.
- **Triangulation** — Mode 2, subscribed tracks only: 2-cam DLT (`cv2.triangulatePoints`), aniposelib for ≥3 cams (S5.5); reprojection-gated; 3D Kalman; `Track3D` with the same `track_id`.

### Operational modes

| Mode | Cameras | Calibration | Output |
|---|---|---|---|
| **1** `single_cam_homography` | 1 | `calibrate single-cam` (`K=I, D=0, R=I, t=0` + real `H`) | `Track2D`; triangulation stack not instantiated |
| **2** `dual_cam_homography_triangulation` | 2 | `calibrate-all` / `calibrate-2cam` | `Track2D` + `Track3D` for matched subscriptions |

**Degradation.** A Mode 2 node that loses a camera keeps serving `Track2D` from the survivor: the synchronizer emits solo `FramePair`s after `ingestion.frame_sync.degraded_emit_after_ms` (100), `cameras_seeing_min: 2` subscriptions stop matching so `Track3D` halts cleanly, `Orchestrator.source_status[cam]` flips to `"exited"`/`"crashed"`, the global `stop_event` is not set, and Mahalanobis matching keeps `track_id`s across the transition. The same `_try_emit_solo` code covers Mode 1 startup. Solo emission is latest-frame-only via a sticky per-camera degraded flag (Mode 1 pays no wait; Mode 2 pays it once); if all cameras are degraded, a probe un-degrades one every 10× the threshold so alignment can re-form (fixed 2026-08-06).

### The five plugin seams (and only these)

| Seam | v1 implementations | Why a plugin |
|---|---|---|
| `FrameSource` | `rtsp`, `replay` (`backbone/ingestion/`) | recorded MP4 for tests; future USB/ROS |
| `Detector` | `yolo_onnx`, `yolo_openvino` (`backbone/detection/`) | ORT CUDA vs Intel CPU/iGPU; `yolo_onnx_pose` in S5.5 |
| `Tracker` | `bytetrack` (`backbone/homography/`) | SORT/OC-SORT swap |
| `Triangulator` | `opencv_dlt` (`backbone/triangulation/`) | aniposelib for ≥3 cams |
| `MetadataSink` | `udp`, `mqtt` (`backbone/comms/`) | ROS, PLC later |

ABCs live in `backbone/core/interfaces.py`; implementations self-register with `@<seam>_registry.register("name")`, and each package `__init__.py` imports its modules so the decorators fire on import. `backbone/runtime/orchestrator.py` is the **only** caller of `registry.create()` and imports every layer package at module top.

**No ABCs for:** `FootProjector`, `CrossCamFusion`, `DisagreementGate`, `SubscriptionManager`, `ReprojectionGate`, `KeypointAssociator`, `TemporalStabilizer` and similar single-implementation utilities.

### Hardware

- **Dev:** RTX 5070 12 GB (Blackwell sm_120), Linux/WSL2, ORT `CUDAExecutionProvider` on CUDA 12.9.
- **Production (later):** Jetson Orin NX 16 GB — same plugin contract, `calibration.json`, UDP schema and `.onnx`. Avoid x86-only deps. Training is external; the Backbone is inference-only.
- ONNX export (training env): `yolo export model=yolo11n.pt format=onnx imgsz=640 dynamic=False simplify=True opset=17` → `models/`, point `detection.onnx_path` at it, inspect with `python tools/onnx_inspect.py`.

## KPIs (Backbone v1 acceptance)

| Indicator | Target |
|---|---|
| Capture → publish latency, p95 | < 200 ms |
| Homography reprojection error | ≤ 2 px |
| Triangulation reprojection error per view | ≤ 5–8 px (gate) |
| Detection mAP@0.5 | ≥ 0.90 |
| Pallet empty/full precision / recall | ≥ 0.95 / ≥ 0.93 |

Latency: `tools/latency_probe.py` + the orchestrator's `LatencyMeter`. Geometry: `tests/test_e2e_homography_synthetic.py` (≤1 mm zero-noise, <10 cm under 2 px noise) and `tests/test_e2e_triangulation_synthetic.py` (≤1 mm zero-noise). On-site tape-measure verification pending.

## Orchestrator

`backbone.runtime.Orchestrator(config_path)`: loads `backbone.yaml`; picks Mode 1/2 from `len(cameras)` (Mode 1 leaves `Triangulator`, `KeypointAssociator`, `ReprojectionGate`, `Tracker3D` as `None`); builds `CameraRig`, `ZoneRegistry`, `SubscriptionManager`, sources, bus + synchronizer, detector, homography stack, triangulation stack, `Publisher`; exposes sync `step(framepair)` (tests) and async `run()` (per-source threads + one pipeline thread, SIGINT/SIGTERM via `install_signal_handlers()`); surfaces `mode`, `source_status`, `latency_meter`, `frame_count`. Refuses to start with no `metadata.sinks`.

## Test suite

`pytest` runs backbone + calibration; `cd monitor_web && pytest` the dashboard. Hermetic: no real RTSP, cameras or weights. `test_orchestrator.py` composes the whole pipeline from YAML with a stub ONNX + `ReplayFrameSource`s and asserts `Track2D` + `Track3D` on a loopback socket. Files map to sprints: registry (S0); calibration_schema, camera_rig, geometry, timestamps (S1); calibrate_cli, multical_io (S1.5); frame_bus, frame_sync, ingestion_* (S2); detection_*, yolo_onnx (S3); foot_projector … e2e_homography_synthetic (S4); zones … e2e_triangulation_synthetic (S5); metadata_schemas, publisher, udp_sink, orchestrator (S6/S8); calibrate_single_cam (S8).

**2-camera gotcha** (`test_two_camera_disagreement_manifests_as_z_offset`): with exactly 2 cameras the DLT is exactly determined, so the reprojection gate cannot catch cross-cam disagreement — it shows as a Z offset. The S4 `DisagreementGate` catches it before triangulation; the reprojection gate matters from 3 cameras.

Live checks: `tools/rtsp_smoke.py rtsp://…`, `tools/detection_smoke.py --onnx <model> --image <jpg> --keep person`, `tools/latency_probe.py online|listen`.

## Operator dashboard (`monitor_web/`)

Separate FastAPI process; consumes the Backbone over UDP/JSON + shared YAML. Imports only `backbone.comms.schemas`, `backbone.shared.zones`, `backbone.ingestion`, never `backbone.runtime/homography/triangulation`; `backbone.detection` only in `monitor_web/detection_overlay.py`.

- **Run:** `cd monitor_web && pip install -e ".[dev]"` once, then `python -m monitor_web` (:8000, open `localhost` not `0.0.0.0`). Alias `3d`.
- **Stack:** FastAPI + Jinja2 + HTMX + Material Web + Pixi.js (floor map) + Alpine.js (`static/js/big_panel.js` = `Alpine.store('bigPanel')`). Script order matters: `video_ws.js` is the first deferred script in `<head>`; `big_panel.js` is a classic deferred script immediately before Alpine. No Node toolchain.
- **Layout:** MAP / CAM 1 / CAM 2 big panel, LOGS + STATUS sidebar, START/STOP (spawns/kills Backbone + isistream), GB/FR i18n. Every panel has an expand `[]` button (centered ~1080p overlay, Esc closes; stacking: panel 1000 < backdrop 999 < Settings 2000 < draw toolbar 2100).
- **Video:** all panel video over one multiplexed `/ws/video` socket (`api/routes_ws_video.py`, `static/js/video_ws.js`): binary `uint8 idLen | stream-id | JPEG`; ids `cam:<id>`, `cam:<id>:warp`, `zone:<patch_id>`, `unified`; client subscribes visible panels only and resubscribes on `config:saved`. MJPEG endpoints (`/stream/video|zone|unified|mp4`) remain for debugging and share the same pipelines (`build_*_stream` in `routes_video.py`). Config-save handlers are sync `def` on purpose (blocking tail runs in the threadpool).
- **Floor map:** per-class sprites, Type-1 proximity rings (`config/danger_zones_object.yaml`), Type-2 polygon danger zones (`Zone.kind`/`severity`), proximity arrows.
- **Settings modal (`+`):** cameras (two slots `cam_a`/`cam_b`, RTSP URL or USB `/dev/video*`; empty Cam 2 ⇒ Mode 1), detection model (`backbone.yaml` `detection` block; backend auto-picked by `backbone.shared.hardware.gpu_available()`: NVIDIA → `yolo_onnx`, else `yolo_openvino`), up to 6 zones drawn on the map (palette / étagère / danger), Étagères (4 corners → auto-split → drag). Writes `backbone.yaml` + `zones.yaml` atomically (YAML comments do not survive).
- **Cam views run no models in points mode:** skeletons from wire persons (`WirePoseSource`), boxes from wire observations (`WireObjectSource`); frames mode falls back to the local `AsyncPoseRunner`. `detection_overlay.get_detector` serves only the MP4 dev viewer.
- **Zone workers:** one `ZoneDetectionWorker` thread per camera, one CUDA session per `(model, input_size)`, no subprocesses (each CUDA context ≈ 0.5 GB). Guards: VRAM admission (`ZoneModelUnavailable` below 1.5 GB free — an ORT OOM mid-build throws CUDA 700 and corrupts every session), per-zone circuit breaker (30 s), optional `max_fps` per patch. Escalation if CUDA 700 recurs: one supervised inference subprocess, never per-model.
- **Detector plugin auto-select (`detection_overlay.select_plugin`):** by output names — `dets`/`labels`/`masks` → `rfdetr_onnx_seg` (built with `onnx_path`, `class_names` default `[palette, carton, polybag]`, `confidence_threshold`, optional `mask_threshold`; NMS-free, fixed square input); 2 outputs → `*_seg`; 1 → detect. RF-DETR exports live under `trainer/isidet/models/rfdetr/<ts>/`; `list_trained_onnx()` scans `_MODEL_ROOTS`.
- **Endpoints:** `/api/status`, `/api/config`, `/api/zones`, `/api/zone-patches`, `/api/danger-zones-object`, `/api/cameras/available`, `/api/ui-settings`, `/api/logs`, `/api/control/{start,stop}`, `/ws/video`, `/ws/tracks`, `/stream/*`.
- **Hidden MP4 dev viewer:** double-click the logo → password (`Settings.mp4_unlock_password`, default `isitec`, env `MONITOR_WEB_MP4_UNLOCK_PASSWORD`, `POST /api/unlock`) → MP4 tab replays a `media_dir` file with the configured detector in-process. Obscurity for a localhost tool, not auth.
- **Tests:** `cd monitor_web && pytest` (hermetic).

## Training (`trainer/isidet/`) — external

YOLOv11 detection (RF-DETR path exists, unused). `conda activate isi-train && python scripts/run_train.py --config configs/train_pallet.yaml` from `trainer/isidet/`. Config: `weights`, `imgsz`, `epochs` (authoritative), `batch_size`, `workers`, warehouse augmentations incl. `camera_aug`, export `pt` + `onnx` + `openvino` with `export_nms: false`, `export_opset: 17`. LR/scheduler in `configs/optimizers/yolo_optim.yaml`. Runs: `runs/detect/models/yolo/<model>_e<epochs>_<ts>/` with `report.md`. Data prep: `scripts/prepare_labelme_dataset.py`, `scripts/labelme_to_yolo.py --preserve-splits`, `tools/merge_pallet_dataset.py`. Sanity: `trainer/isidet/run_test.sh`. WSL2: heavy models (yolo11l/yolo26l) can swap-thrash the 12 GB VM into an EIO crash — use yolo11m, lower `batch_size`/`workers`.
