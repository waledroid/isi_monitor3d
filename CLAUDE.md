# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Source of truth

- **Spec (French):** `docs/specs/Cahier des Charges-Système de Vision Industrielle.pdf` — the customer-facing requirements, KPIs, and 3-month sprint plan from Isitec.
- **Approved build plan:** `~/.claude/plans/starry-stirring-fairy.md` — defines sprints S0–S6, file structure, plugin seams, and verification. When the spec and the plan agree, the plan is the executable form; when they disagree, ask before deviating.
- The codebase implements the **Backbone** described in the plan. Modules (Sécurité, Palettes, Rayonnages, Dashboard, PLC/WMS gateway) are explicitly out of scope here — they live in separate processes and consume only the UDP/JSON contract.

**Sprint progress** (so you know what exists vs what doesn't before reading code):

| Sprint | Status | What lives where |
|---|---|---|
| S0 — skeleton & core | ✅ | `backbone/core/{registry,types,interfaces}.py` |
| S1 — shared utilities & calibration schema | ✅ | `backbone/shared/{camera_rig,geometry,timestamps,zones}.py`, `calibration/schema.py` |
| S1.5 — Multical calibration backend | ✅ | `calibration/{calibrate,multical_io}.py`, `calibration/setup_multical.sh` |
| S1.75 — `monitor3d` conda env | ✅ | `environment.yml` |
| S2 — ingestion (RTSP + replay + sync + bus) | ✅ | `backbone/ingestion/`, `tools/rtsp_smoke.py` |
| S3 — detection (YOLO11 via **ONNX Runtime**, detect-only) | ✅ | `backbone/detection/`, `tools/{detection_smoke,onnx_inspect}.py` |
| S4 — homography layer (foot proj → fusion → ByteTrack-in-meters → stabilizer) | ✅ | `backbone/homography/` |
| S5 — triangulation layer (subscription-driven, foot-centroid; pose deferred to S5.5) | ✅ | `backbone/triangulation/` |
| S6 — comms (metadata) + orchestrator + latency probe | ✅ | `backbone/{comms,runtime}/`, `tools/latency_probe.py` |
| S7 — simplification pass (drop dead/speculative paths) | ✅ | -113 LOC, -7 tests; `FrameBus` single-subscriber, `Tracker3D.predict_only` removed, pose stubs gone, `UdpSink` multicast TTL gone |
| S8 — operational modes (1-cam / 2-cam) + runtime degradation | ✅ | `calibration/calibrate_single_cam.py`, `frame_sync.degraded_emit_after_ms`, `Orchestrator.mode` / `.source_status` |

**Backbone v1 is feature-complete.** What remains is on-site validation with a real Hikvision rig (calibrate via `calibrate-all` for Mode 2 or `calibrate single-cam` for Mode 1, swap the synthetic stub ONNX for a trained `yolo11n.onnx`, verify the KPIs end-to-end), Isitec-specific training that produces the production `.onnx`, and the deferred S5.5 pose-mode extension. The Jetson Orin NX port is out of v1 scope and requires no code changes — the `.onnx` and Python are portable, only the conda env's `onnxruntime` variant differs.

## Commands

```bash
# environment (conda — the primary path)
conda env create -f environment.yml -n monitor3d
conda activate monitor3d                          # always do this before anything below
conda env update -f environment.yml -n monitor3d --prune   # refresh after env edits

# calibration — paths depend on operational mode
# (AprilGrid extrinsics need system build deps first — apt, one-time, need root:
#    sudo apt install -y cmake libopencv-dev libeigen3-dev
#  ChArUco-only calibration needs none of these.)
bash calibration/setup_multical.sh           # one-time: bootstrap Multical's isolated venv (Mode 2)
python -m calibration.calibrate calibrate-all ...                   # Mode 2: 2-cam Multical joint BA (single ChArUco)

# Mode 2, two-stage (recommended for 2 cams): ChArUco intrinsics → multi-AprilGrid extrinsics (K fixed).
#   1) print the boards (one PNG per board; print at 100% scale, 0 margins):
python -m calibration.calibrate gen-boards --output-dir boards/ --n-boards 6   # A4 ChArUco + 6 AprilGrids
#   2) shoot: per-camera ChArUco from many angles (intrinsics) + both cams viewing the 6-AprilGrid
#      target spread over the shared volume (extrinsics) + one ChArUco floor shot per cam (world anchor):
python -m calibration.calibrate calibrate-2cam \
    --intrinsic-dir cam_a=shots/intr_a --intrinsic-dir cam_b=shots/intr_b \
    --extrinsic-dir cam_a=shots/extr_a --extrinsic-dir cam_b=shots/extr_b \
    --floor-shot cam_a=shots/floor_a.jpg --floor-shot cam_b=shots/floor_b.jpg \
    --work-dir /tmp/cal --output calibration.json
# (calibrate-2cam runs `multical intrinsic` then `multical calibrate --calibration intrinsic.json
#  --fix_intrinsic`; AprilGrid board sizes via --tag-length/--tag-spacing/--n-boards; ChArUco via
#  --squares-x/-y/--square-length/--marker-length. Same calibration.json schema as calibrate-all.)

# Visualize the result in Multical's built-in 3D viewer (camera + board poses, reprojection).
# Needs a display (WSLg/$DISPLAY) + viewer deps (setup_multical.sh installs them best-effort).
python -m calibration.calibrate calibrate-all  ... --vis            # auto-open after the BA solve
python -m calibration.calibrate calibrate-2cam ... --vis            # (interactive: blocks until closed)
python -m calibration.calibrate vis --workspace /tmp/cal           # re-open a saved run (keep --work-dir!)

python -m calibration.calibrate single-cam \                        # Mode 1: 1-cam 4-point floor fit
    --camera-id cam_a --image-size 1920 1080 \
    --pair u1,v1,X1,Y1 --pair u2,v2,X2,Y2 \
    --pair u3,v3,X3,Y3 --pair u4,v4,X4,Y4 \
    --output calibration.json
# (Use ≥5 pairs for Mode 1 — with exactly 4 the fit is exactly-determined and the residual gate can't fire.)

# tests
pytest                                            # whole suite (covers backbone + calibration)
pytest tests/test_registry.py                     # one file
pytest tests/test_registry.py::test_register_and_create   # one test
pytest -k "registry"                              # by keyword
pytest --cov=backbone --cov=calibration           # with coverage

# lint / format
ruff check backbone calibration tests
ruff check --fix backbone calibration tests
ruff format backbone calibration tests

# run the system end-to-end (Direction 1 — the deployed default: two processes)
python -m backbone.runtime --config config/backbone.yaml   # the METRIC ENGINE (points mode: no CUDA; SIGINT/SIGTERM to stop)
python -m perception --config config/backbone.yaml         # the perception producer (capture+detect+pose → detection sets)
# (the dashboard's START button spawns/reaps BOTH; headless = two systemd units, same config file.
#  Rollback to the single-process pre-split Backbone: ingestion.mode: frames + restart.)
python tools/latency_probe.py online --config config/backbone.yaml --seconds 60   # KPI probe

# pip-only fallback (no conda)
pip install -e ".[dev,geometry,schemas]"          # full deps; conda is still the primary path
```

**Python target is 3.10** — pinned in the conda env to match JetPack 6.x for the eventual Jetson port. Don't use 3.12+ syntax (e.g. `type` alias keyword).

**Two environments, on purpose:**
- `monitor3d` conda env — everything the Backbone runtime and tests need. Activated for every dev / test / lint command above. Includes Python 3.10, OpenCV 4.13 (headless py310), GStreamer 1.28 + plugins, PyGObject, FilterPy, scipy, numpy, PyYAML, pytest/ruff, **CUDA 12.9 + ONNX Runtime 1.25 (with CUDAExecutionProvider on the RTX 5070 Blackwell sm_120)**, **OpenVINO** (S13 `yolo_openvino` detector — CPU/Intel-iGPU only, imported lazily), **pydantic 2.x** (S6 metadata schemas).
- `calibration/.venv-multical/` venv — Multical only, isolated because it pins `opencv-contrib-python <=4.7.0` and would corrupt the runtime env's OpenCV 4.13. `calibrate.py` invokes its binary by absolute path; the Backbone never imports Multical.

**RTSP ingest backend (S2, implemented).** S2's RTSP path uses GStreamer driven from Python via **PyGObject** (`import gi; gi.require_version("Gst","1.0"); from gi.repository import Gst, GLib`), not `gst-python` and not `cv2.CAP_GSTREAMER` (the conda-forge OpenCV build is FFmpeg-only — no GStreamer backend compiled in). We avoided `gst-python` because its only py3.10 conda-forge build is 1.21.x, which transitively pins FFmpeg 5 and conflicts with OpenCV 4.13's FFmpeg 8 dep. PyGObject is gstreamer-version-agnostic and binds to whatever GStreamer's GIR files describe at runtime. The low-latency pipeline (in `backbone/ingestion/rtsp.py:PIPELINE_TEMPLATE`): `rtspsrc latency=100 drop-on-latency=true protocols=tcp ntp-sync=true ! {depay} ! {decoder} ! videoconvert ! video/x-raw,format=BGR ! appsink emit-signals=true sync=false max-buffers=1 drop=true`. The depay/decode pair is **codec-aware** — `RtspFrameSource` probes the stream's codec once at start (`_probe_rtsp_codec` via `ffprobe`) and picks `rtph264depay`/`avdec_h264` for H.264 or `rtph265depay`/`avdec_h265` for H.265/HEVC (defaults to H.264 if the probe fails). This supports a mixed-codec rig (e.g. an H.264 Hikvision + an H.265 Dahua). We deliberately use an **explicit depay, not `decodebin`**: a depayloader's static sink pad makes rtspsrc's delayed link race-free, whereas `decodebin`'s autoplug intermittently trips `streaming stopped, reason not-linked (-1)` (~1-in-4 starts) and would also build a dead audio branch on cameras that carry an audio track. Each `RtspFrameSource` owns one `Gst.Pipeline` + `GLib.MainLoop` in a daemon thread; the appsink callback decodes BGR and pushes into an internal `Queue(maxsize=1, drop-old)`.

**Capture timestamp policy.** `Frame.capture_ts` is set to `time.time()` **at the appsink callback** in `RtspFrameSource._on_sample`. This lags the camera shutter by the rtspsrc latency buffer (~100 ms) + decode time, but it is the earliest moment the Backbone observes the frame, and using it consistently across cameras keeps cross-camera pairing stable. End-to-end latency probes (`tools/rtsp_smoke.py` now, `tools/latency_probe.py` in S6) measure `time.time() - capture_ts` and correctly reflect the time the Backbone *holds* the frame. True shutter-NTP alignment would require a `GstNetClientClock` against an NTP server and reading `buf.pts + pipeline.base_time` — not needed for the < 200 ms KPI, but the future-work seam is documented in `rtsp.py`'s docstring.

## Architecture — the non-negotiables

The Backbone is a single process that turns RTSP video into metric, identity-stable metadata. Seven operating principles drive every design decision; deviating from them silently corrupts modularity, identity, or reliability:

**Direction 1 (July 2026) — the shipped topology.** `ingestion.mode` in `backbone.yaml` picks who owns the pixels:

- **`points` (the deployed default):** the Backbone is a pure **metric engine** — no RTSP sources, no detectors, **no CUDA, no onnxruntime import** (~190 MB RSS). A separate perception producer (`python -m perception --config config/backbone.yaml`, spawned/reaped by the dashboard's `PerceptionHost` alongside the Backbone on START/STOP) owns capture → zone-scoped detection → pose and publishes per-camera `DetectionSetMessage`s (schema type `detection_set`; explicit-empty heartbeat; `seq` gap counting; `config_fingerprint` drift warning) to the engine's loopback ingest port (`ingestion.points.listen_port`, default 9010). `backbone/ingestion/points_in.py` feeds them into the **unchanged** `FrameSynchronizer`, so pairing, Mode 1/2, and runtime degradation work exactly as with frames. Persons ride the same stream (`cls="person"` + `keypoints_uv`) and also ride the observations echo back out, so the dashboard renders skeletons with **zero inference** (`WirePoseSource`). `metadata.images` is refused in points mode (no frames ⇒ no JPEGs).
- **`frames` (rollback + hermetic-test path):** today's pre-split behavior, byte-identical — the Backbone owns all perception. One YAML line + restart to go back.
- The producer runs as **its own process, never in-process**: measured on the rig, the same tick costs ~55 ms standalone vs ~2,200 ms inside the dashboard (GIL + ORT-pool contention). `perception/` is FastAPI-free by construction; monitor_web only supervises it.
- **Shared frame bus (decode once, fan out):** the producer publishes every decoded frame to `/dev/shm/isi3d_frame_<cam>` (`backbone/shared/frame_shm.py` — double-buffered seqlock, lock-free readers); the dashboard's camera hub PREFERS the bus (zero RTSP session of its own while the producer runs, re-checks every 5 s) and falls back to its own source when the bus is absent/stale (backbone stopped, frames mode, pre-START preview). ONE RTSP session + ONE decode per camera system-wide; panels show the exact pixels the models saw. A `shm` FrameSource plugin (`backbone/ingestion/shm_source.py`) exposes the bus to any future consumer.
- The differential test (`tests/test_points_mode.py`) pins that both modes produce identical Track2D streams from the same detections. Measured KPIs in points mode: capture→publish p50 ≈ 77 ms / p95 ≈ 126 ms; total VRAM ≈ 2.5 GB (one CUDA process) vs ≈ 5.2 GB before.

1. **One calibration, two queries.** `calibration.json` (per-camera `K, D, R, t` plus derived `H` and `P`) feeds both geometric methods. They cannot drift independently.
2. **One identity space.** The homography tracker owns `track_id`. The triangulation layer augments tracks with 3D and *never* re-IDs.
3. **Subscription, not polling.** Triangulation runs only for tracks matching rules in `config/subscriptions.yaml`. The default output is `Track2D` from homography; `Track3D` is on-demand.
4. **Plugin where multiplicity is real — concrete everywhere else.** There are exactly **five** ABC seams; `tests/test_registry.py::test_five_seams_present` pins this. Adding ABCs anywhere else is ceremony, not modularity.
5. **Process boundaries are contractual.** Backbone has **zero imports** from modules. The UDP/JSON schema (later in `backbone/comms/schemas.py`) is the only contract. Expand the schema rather than sharing code.
6. **Fail honestly.** Every geometric output is gated — reprojection error for triangulation, cross-camera disagreement for fusion. Bad input must produce no output or flagged-uncertain output, never silent-bad output.
7. **Industrial defaults.** systemd-supervised, no cloud, deterministic restart. Latency is measured against `frame.capture_ts` (the single capture-time clock propagated through every downstream message), never against `time.time()` at publish.

### Two methods, one Backbone

**Zone-scoped detection (default).** The system is zone-based: with `detection.scope: zones` (the default) the Backbone's object detector sees only the configured floor zones — each zone polygon is projected into each camera (z=0 and z=2 m, distortion-aware) once at build, cropped per pair, batched through the detector at `detection.zone_imgsz` (default 384, needs a dynamic export), and remapped (`backbone/detection/zone_scope.py`). **No zones configured ⇒ the object detector is not built at all — pose-only Backbone** (person tracks continue; the pose model is the stated global exception). `scope: full_frame` restores the everything-visible behaviour (existing tests pin it). The dashboard's zone *patches* are a display layer; by default (`zone_detection_source: backbone`) the zone workers RENDER the Backbone's per-camera `ObservationsMessage` (UDP, schema v6 — boxes + occupancy + optional mask polygons via `detection.decode_masks`) instead of running their own models — ONE perception; `zone_detection_source: local` restores in-dashboard per-zone inference (models/SAHI/ENH) as the dev/fallback path. The COMMUNICATION zone cards read `/api/zone-patches/state` either way.

- **Homography** runs always, per frame, per detection. Foot point → undistort → `H` → `(X, Y) m` → cross-camera fusion + disagreement gate → ByteTrack in meters → temporal vote → publish `Track2D`.
- **Triangulation** runs only in **Mode 2** for subscribed tracks. Identity inherited from the 2D tracker; 2-cam DLT (`cv2.triangulatePoints`) by default, N-cam via `aniposelib.CameraGroup` when ≥3 cameras saw the track (S5.5); reprojection-gated; 3D Kalman; publish `Track3D` with the same `track_id`.

### Operational modes (S8)

The orchestrator chooses an operational mode at build time from `len(cfg["cameras"])`:

| Mode | Cameras | Calibration | Pipeline output |
|---|---|---|---|
| **Mode 1** — `single_cam_homography` | 1 | `calibrate single-cam` (4+ point floor-plane fit; `K=I, D=0, R=I, t=0` placeholders + real `H`) | `Track2D` only. Triangulation stack never instantiated. |
| **Mode 2** — `dual_cam_homography_triangulation` | 2 | `calibrate-all` (Multical joint BA → `K, D, R, t, H, P` per camera) | `Track2D` always + `Track3D` for matched subscriptions. |

**Runtime degradation.** A Mode 2 deployment that loses one camera at runtime (RTSP drop, dead PoE, unplugged) continues serving `Track2D` from the surviving camera. The synchronizer emits solo `FramePair`s after `ingestion.frame_sync.degraded_emit_after_ms` (default 100 ms). Subscriptions with `cameras_seeing_min: 2` naturally stop matching ⇒ `Track3D` halts cleanly. `Orchestrator.source_status[cam_id]` flips from `"alive"` to `"exited"` or `"crashed"`. The global `stop_event` is **not** set — a single-source crash never kills the pipeline. ByteTrack matches on Mahalanobis distance, so `track_id`s persist across the degradation/recovery transition.

**One mechanism, both cases.** The same solo-emit code in `FrameSynchronizer._try_emit_solo` covers Mode 1 startup (only 1 camera ever configured) and Mode 2 degradation (one of 2 cameras dies mid-run). Downstream (`CrossCamFusion`, `DisagreementGate`, `Orchestrator.step()`) handles N=1 input naturally.

**Latency note.** Solo emission is LATEST-FRAME-ONLY via a sticky per-camera degraded flag: Mode 1 (single camera configured) emits every frame immediately (no `degraded_emit_after_ms` tax); Mode 2 degradation pays the wait once at entry — the NEWEST buffered frame emits, older ones are discarded — then streams the survivor at full input fps with zero buffering. The flag clears automatically when aligned pairing resumes.

### The five plugin seams (and only these)

| Seam | ABC location | v1 implementations | Why a plugin |
|---|---|---|---|
| `FrameSource` | `backbone/core/interfaces.py` | `rtsp` ✅, `replay` ✅ — `backbone/ingestion/` | RTSP today, recorded MP4 for dev/tests, future USB/ROS bags |
| `Detector` | same | `yolo_onnx` ✅, `yolo_openvino` ✅ — `backbone/detection/` | ONNX Runtime + CUDAExecutionProvider (NVIDIA); `yolo_openvino` for Intel CPU/iGPU edge nodes (same raw head, reuses the decode — CPU-only on the RTX 5070); pose-mode `yolo_onnx_pose` lands in S5.5 |
| `Tracker` | same | `bytetrack` ✅ — `backbone/homography/` | ByteTrack-in-meters; SORT/OC-SORT/Kalman-only easy to swap |
| `Triangulator` | same | `opencv_dlt` ✅ — `backbone/triangulation/` | 2-cam DLT now; aniposelib for ≥3 cams in S5.5 |
| `MetadataSink` | same | `udp` ✅ — `backbone/comms/` | UDP/JSON now; future MQTT, ROS, S7 PLC |

Implementations register themselves via decorator:

```python
@detector_registry.register("yolo_onnx")
class YoloOnnxDetector(Detector):
    ...
```

The runtime orchestrator (`backbone/runtime/orchestrator.py`) is **the only place** that calls `registry.create()`. Plugins never instantiate each other. The orchestrator explicitly imports every layer package (`backbone.comms`, `backbone.detection`, `backbone.homography`, `backbone.ingestion`, `backbone.triangulation`) at module top to fire `@register` before any plugin lookup.

**Auto-registration pattern.** Each plugin's package `__init__.py` imports the implementation modules so the `@register` decorators run on `import` — e.g., `import backbone.ingestion` makes `frame_source_registry.names()` return `['replay', 'rtsp']`. Replicate this when adding a new plugin (e.g., S5.5's `yolo_onnx_pose`).

### Where ABCs do NOT belong

`FootProjector`, `CrossCamFusion`, `DisagreementGate`, `SubscriptionManager`, `ReprojectionGate`, `KeypointAssociator`, `TemporalStabilizer`, and similar utilities are concrete, single-implementation modules. Each has one sensible way to be implemented. Do not wrap them in ABCs — that adds ceremony without enabling real substitution.

### Hardware targeting

- **Dev/v1:** NVIDIA RTX 5070 12 GB workstation (Linux/WSL2), Blackwell sm_120. ONNX Runtime with `CUDAExecutionProvider` on CUDA 12.9 runs `yolo_onnx` natively.
- **Production (out of current scope):** Jetson Orin NX 16 GB (Seeed reComputer J4012). Same plugin contract, same `calibration.json`, same UDP schema, **same `.onnx` artifact** — ONNX Runtime has a Jetson build; the `.onnx` is portable, so the port is a deploy/env job not a code change. Per the user's directive: training is **external** to this repo; the Backbone is inference-only and consumes `.onnx` files.

Avoid x86-only deps. The user redirected S3 away from TensorRT to ONNX Runtime explicitly so the artifact stays portable across the dev/Jetson split.

**ONNX export pipeline** (lives outside this repo, in your training env):
```bash
yolo export model=yolo11n.pt format=onnx imgsz=640 dynamic=False simplify=True opset=17
```
Drop the resulting `.onnx` into `models/` and point `config/backbone.yaml`'s `detection.onnx_path` at it. Inspect with `python tools/onnx_inspect.py models/yolo11n.onnx` before plugging it in.

## KPIs (acceptance for Backbone v1)

| Indicator | Target |
|---|---|
| End-to-end latency (capture → publish, p95) | < 200 ms |
| Homography reprojection error | ≤ 2 px |
| Triangulation reprojection error per view | ≤ 5–8 px (gate threshold) |
| Detection mAP@0.5 | ≥ 0.90 |
| Classification precision / recall (pallet empty/full) | ≥ 0.95 / ≥ 0.93 |

Latency is measured against `frame.capture_ts`, instrumented by `tools/latency_probe.py` (S6) and the orchestrator's `LatencyMeter` (`capture_to_publish`). Geometric error is exercised end-to-end with synthetic ground truth in `tests/test_e2e_homography_synthetic.py` (S4, ≤1 mm zero-noise, <10 cm under 2 px pixel noise) and `tests/test_e2e_triangulation_synthetic.py` (S5, ≤1 mm zero-noise foot-centroid 3D). On-site verification against tape-measured truth is deferred until a Hikvision rig is reachable.

## Orchestrator entry point

`backbone.runtime.Orchestrator(config_path)` is the production entry point. It:

1. Loads `backbone.yaml` (calibration path, cameras, ingestion tunables, detection plugin + onnx path, homography thresholds, triangulation subscriptions, metadata sinks).
2. Detects operational mode from `len(cameras)` (1 → Mode 1, 2 → Mode 2). In Mode 1 it **does not instantiate** the triangulation stack (`Triangulator`, `KeypointAssociator`, `ReprojectionGate`, `Tracker3D` all stay `None`).
3. Builds `CameraRig`, `ZoneRegistry`, `SubscriptionManager`, all `FrameSource`s, the bus + synchronizer, the detector, the homography stack, the (mode-dependent) triangulation stack, and the `Publisher` (fan-out to UDP).
4. Exposes a synchronous `step(framepair)` (used by tests) and an async `run()` that spawns per-source ingestion threads + a single pipeline thread, with `SIGINT`/`SIGTERM` graceful shutdown via `install_signal_handlers()`.
5. Surfaces operator-visible state: `Orchestrator.mode`, `Orchestrator.source_status` (per-camera `"alive"`/`"exited"`/`"crashed"`), `Orchestrator.latency_meter`, `Orchestrator.frame_count`.

Refuses to start with no `metadata.sinks` configured — bad config fails fast.

## Test suite

`pytest` runs the full suite — **290 tests as of S8**, all green. Mapping by sprint:

- `test_registry.py` — S0 (5 ABC seams pinned by `test_five_seams_present`)
- `test_calibration_schema.py`, `test_camera_rig.py`, `test_geometry.py`, `test_timestamps.py` — S1
- `test_calibrate_cli.py`, `test_multical_io.py` — S1.5
- `test_frame_bus.py`, `test_frame_sync.py`, `test_ingestion_replay.py`, `test_ingestion_rtsp.py` — S2
- `test_detection_preprocess.py`, `test_detection_postprocess.py`, `test_yolo_onnx.py` — S3
- `test_foot_projector.py`, `test_cross_cam_fusion.py`, `test_disagreement_gate.py`, `test_track.py`, `test_bytetrack.py`, `test_temporal_stabilizer.py`, `test_e2e_homography_synthetic.py` — S4
- `test_zones.py`, `test_subscription_manager.py`, `test_keypoint_associator.py`, `test_opencv_dlt.py`, `test_reprojection_gate.py`, `test_tracker_3d.py`, `test_e2e_triangulation_synthetic.py` — S5
- `test_metadata_schemas.py`, `test_publisher.py`, `test_udp_sink.py`, `test_orchestrator.py` — S6 (Mode-1/Mode-2/degradation tests added in S8 also live here)
- `test_calibrate_single_cam.py` — S8 (4-point Mode 1 fit + back-compat for pre-S8 calibration files)

**Hermetic strategy.** Real RTSP, real cameras, and real ONNX weights aren't unit-tested (they would need a live rig). `test_orchestrator.py` proves the full pipeline composes from YAML by feeding a hand-built batched stub ONNX through real `ReplayFrameSource`s and asserting `Track2D` + `Track3D` arrive on a loopback UDP socket — the on-site rig swaps the stub for the trained model and the synthetic rig for actual cameras.

**Live verification entry points:**
- `python tools/rtsp_smoke.py rtsp://<hikvision>` — RTSP capture sanity.
- `python tools/detection_smoke.py --onnx <model> --image <jpg> --keep person` — single-frame detection + per-stage timing.
- `python tools/latency_probe.py online --config config/backbone.yaml --seconds 60` — full pipeline p50/p95/p99 latency.
- `python tools/latency_probe.py listen --port 50001 --seconds 60` — listen on the UDP bus and measure `now - envelope.ts` (works against any running Backbone, not just a local one).

**One important S5 gotcha** (documented in `tests/test_e2e_triangulation_synthetic.py::test_two_camera_disagreement_manifests_as_z_offset`): with exactly 2 cameras, the linear-DLT triangulation system is exactly determined, so the reprojection gate **cannot** catch cross-cam disagreement at the 3D stage — disagreement instead manifests as a Z offset. The S4 `DisagreementGate` is what catches cross-cam metric mismatches before triangulation. The reprojection gate becomes genuinely useful with 3+ cameras (S5.5 aniposelib path).

---

## Operator dashboard (`monitor_web/`)

A **separate FastAPI process** (sibling project under `monitor_web/`) — the operator UI. Consumes the Backbone over the UDP/JSON bus + shared YAML; per the process-boundary rule it imports only consumer-side helpers (`backbone.comms.schemas`, `backbone.shared.zones`, `backbone.ingestion`), **never** `backbone.runtime/homography/triangulation`, and imports `backbone.detection` only in **one documented place** — `monitor_web/detection_overlay.py`, used by the zone workers, the cam-view pose overlay, and the hidden MP4 viewer (all below).

- **Run:** `conda activate monitor3d && cd monitor_web && pip install -e ".[dev]"` (one-time), then `python -m monitor_web` (uvicorn on :8000). Shell alias `3d` does activate + run. Open `http://localhost:8000/` (not `0.0.0.0`).
- **Stack:** FastAPI + Jinja2 + HTMX + Material Web (CDN) + **Pixi.js** for the floor map + **Alpine.js** (CDN) driving the big-panel reactive state (`static/js/big_panel.js` = an `Alpine.store('bigPanel')`; markup binds via `x-show`/`:class`/`@click`; the live `<img>`s get their frames imperatively from `video_ws.js` via an Alpine effect in `boot()` — only the MP4 dev tab still uses a `:src` MJPEG URL). Script order matters twice: `video_ws.js` is the **first deferred script in `<head>`** so `window.__videoWS` exists before any module attaches a stream, and `big_panel.js` is a **classic deferred script immediately before the Alpine script** so its `alpine:init` listener registers first (module-vs-defer order is otherwise ambiguous → store never registers → bindings error → views hidden). Session persistence (view + mp4 pick) is plain `sessionStorage` via `Alpine.effect` (no persist plugin). No Node toolchain.
- **Layout:** big panel with a **MAP / CAM 1 / CAM 2** segmented toggle (Pixi 2D digital-twin map + per-camera live video), LOGS + STATUS sidebar, START/STOP (spawns/kills the Backbone subprocess), GB/FR i18n.
- **Video transport (one WebSocket):** ALL panel video (cam views, warp verification, zone panels, unified) flows over a single multiplexed **`/ws/video`** socket (`api/routes_ws_video.py` + `static/js/video_ws.js`, global `window.__videoWS`): binary frames `uint8 idLen | stream-id | JPEG`; ids `cam:<id>` / `cam:<id>:warp` / `zone:<patch_id>` / `unified`; client subscribes only the *visible* panels and **resubscribes on `config:saved`** so settings apply live. This replaced per-panel MJPEG `<img>`s, which exhausted the browser's 6-connection HTTP/1.1 cap and froze the UI on settings saves. The MJPEG endpoints (`/stream/video`, `/stream/zone`, `/stream/unified`, `/stream/mp4`) remain for curl debugging + the MP4 dev tab; both transports share the same frame pipelines (`build_cam_stream` / `build_zone_stream` / `build_unified_stream` in `routes_video.py`). The config-save handlers (`post_config`, `post_zone_patches`) are deliberately **sync `def`** so their blocking tail (onnx introspection, fsync'd writes, `reset_detector()`'s `gc.collect()`, worker reload) runs in the threadpool, never on the event loop.
- **Floor map (S10):** Pixi.js — per-class sprites, smooth motion, **Type-1** object proximity rings (`config/danger_zones_object.yaml`) + **Type-2** polygon danger zones (`Zone.kind` / `severity`), proximity arrows with distance labels.
- **Settings modal (S11/S12/S13):** the `+` icon opens a transparent overlay (titled **Settings**) to edit **cameras**, the **detection model**, and define up to 6 **zones** by clicking points on the map; persists atomically to `backbone.yaml` + `zones.yaml`. Cameras = two fixed slots **Cam 1 (`cam_a`) / Cam 2 (`cam_b`)**, each with a **type selector RTSP | USB (V4L2)** — RTSP → URL, USB → device path (datalist of `/dev/video*`). Empty Cam 2 ⇒ Mode 1. **Detection model** section sets `backbone.yaml`'s `detection` block (model path + class names + conf) — drives both the live Backbone and the MP4 viewer. The **backend is auto-selected from hardware** (`backbone.shared.hardware.gpu_available()`: NVIDIA GPU → `yolo_onnx`/CUDA, CPU-only → `yolo_openvino`); the modal shows only the matching path field (no manual backend/device pickers). The inactive backend's path is remembered in the UI-settings YAML. Zone categories: **palette** (neutral) / **étagère** (light green) / **danger** (light red).
- **Expand control (S13.2):** every panel (big + the two zone panels) has a `[]` button (bottom-**right**; the title/LIVE/ZONE label is bottom-**left**) that pops the panel to a centered ~1080p overlay (not browser fullscreen), dimming the rest with a backdrop (Esc / backdrop-click to close). Two-way animation (`.panel.expanded` pop-in / `.panel.closing` shrink, ~160 ms; `prefers-reduced-motion` respected). Driven by `$store.bigPanel` (big panel) and a shared `Alpine.data('expandable')` (zone panels). Big-panel video is `object-fit: cover` (fills the panel) normally and `contain` (full frame) when expanded. Stacking order: expanded panel `z 1000` < panel backdrops `999`... < **Settings overlay `2000`** < draw toolbar `2100`, so the `+` modal shows even over an expanded panel.
- **Endpoints:** `/api/status`, `/api/config` (GET/POST cameras+detection+zones), `/api/zones`, `/api/zone-patches`, `/api/danger-zones-object`, `/api/cameras/available`, `/api/ui-settings`, `/api/logs`, `/api/control/{start,stop}`, `/ws/video` (all dashboard video — see transport above), `/stream/video/{cam}` + `/stream/zone/{id}` + `/stream/unified` (MJPEG debug equivalents), `/stream/mp4/{name}` (annotated, hidden dev viewer), `/ws/tracks`.
- **Cam views run NO models in points mode:** the big CAM 1/CAM 2 views never run a full-frame object detector, and with `ingestion.mode: points` they run no pose either — skeletons render from the wire's person observations (`WirePoseSource` in `pose_overlay.py`: keypoints ride `ObservationDet.keypoints_uv`), object boxes come from the zone workers' bus-fed snapshots, and the workers' map-people come from the same wire (mode-gated in `_snapshot_from_bus`; a frames-mode Backbone falls back to the local `AsyncPoseRunner`/worker pose as before). `detection_overlay.get_detector` (full-frame, with its latest-`best.onnx` fallback) is used only by the MP4 dev viewer.
- **Zone detection & isolation:** one background `ZoneDetectionWorker` thread per camera runs every zone patch on the same frame and publishes one atomic snapshot (`zone_worker.py`). Per-zone models share one CUDA session per `(model, input_size)`. In-process guards keep one zone's failure from killing the rest (deliberately **no subprocess sandboxing** — each extra CUDA context costs ~0.5 GB of the 12 GB card): **VRAM admission** (`get_zone_detector` raises `ZoneModelUnavailable` instead of building when free VRAM < 1.5 GB — an ORT OOM mid-build throws CUDA 700 and corrupts the context for every session), a **per-zone circuit breaker** (failed zone disabled 30 s then retried; status `no_vram`/`error` published in the snapshot and drawn as a banner on the zone panel), and an optional per-zone **`max_fps`** budget in `zone_patches` (config-only; a heavy RF-DETR zone capped at 2–4 fps stops dragging light zones — its last detections carry forward between runs). Agreed escalation if CUDA-700 ever still bites: ONE supervisor-restarted inference subprocess, never per-model processes.
- **Detector task auto-selection (`detection_overlay.select_plugin`):** `get_detector` introspects the chosen ONNX/IR's **output names**, not just arity, to pick the task plugin: outputs named `dets`/`labels`/`masks` (RF-DETR's 3 outputs) → **`rfdetr_onnx_seg`**; 2 outputs → `{yolo_onnx,yolo_openvino}_seg`; 1 output → detect. RF-DETR is built with only `onnx_path` + `class_names` (default `[palette, carton, polybag]`) + `confidence_threshold` (+ optional `mask_threshold`) — the YOLO-only kwargs (`iou_threshold`, `keep_classes`, slider `input_size`) are dropped because RF-DETR is NMS-free and reads its own fixed square input. Its masks flow through `draw()` unchanged (same `Detection.mask` path as YOLO-seg); occupancy still works (image-overlap), pose/distance overlays are unaffected (RF-DETR has no pose).
- **Swap to RF-DETR** (note: `backbone.yaml` is rewritten atomically by the Settings modal, so YAML comments don't persist — config it via the modal or by editing then not re-saving). The `detection:` block becomes:
  ```yaml
  detection:
    plugin: rfdetr_onnx_seg
    onnx_path: /home/aatanda/isi_monitor3d/trainer/isidet/models/rfdetr/rfdetr-medium-seg_e41_432px/inference_model.sim.onnx
    class_names: [palette, carton, polybag]
    confidence_threshold: 0.3
    # mask_threshold: 0.5   # optional
  ```
  The RF-DETR export lives under `trainer/isidet/models/rfdetr/<ts>/`, **not** `runs/` — `list_trained_onnx()` scans both (`_MODEL_ROOTS`) so it's offered in the Settings model dropdown (label relative to `trainer/isidet/`). Smoke it with `python tools/detection_smoke.py --onnx <rfdetr.onnx> --image <jpg>` (auto-selects the plugin via the registry; `--annotate out.jpg` to save boxes).
- **Tests:** `cd monitor_web && pytest` (consumer-side, hermetic).

**Hidden MP4 dev viewer (S12.2, implemented).** A password-gated diagnostic tab to replay a media-folder MP4 in the big view **with detections overlaid**: double-click the Isitec logo → enter a password → a 4th **MP4** tab appears beside CAM 2; pick a file from the modal's dropdown (lists `*.mp4` under `media_dir`, default the repo root, recursive, hidden dirs pruned). Playback (`/stream/mp4/{name}`) runs the configured detector **in-process** via the shared `monitor_web/detection_overlay.py` (`get_detector` full-frame — the MP4 viewer is now its main consumer since the cam views went pose-only; paced to the file's FPS; raw playback if no model) — a **deliberate, documented exception** to the no-detector-import rule, justified only because it's a dev-only tool. Password: `Settings.mp4_unlock_password`, default `"isitec"`, override via env **`MONITOR_WEB_MP4_UNLOCK_PASSWORD`** (`POST /api/unlock`). It's obscurity for a localhost tool, **not real auth** — don't commit a real secret here.

## Training (`trainer/isidet/`) — external to the Backbone

The Backbone is **inference-only**; model training lives in the separate **isidet** trainer and produces the `.onnx` the `yolo_onnx` detector consumes. Detection-only (no masks) for the pallet task.

- **Env:** the dedicated **`isi-train`** conda env (ultralytics + torch cu128, GPU-ready on the RTX 5070). Do **not** install the training stack into `monitor3d` — ultralytics pulls `opencv-python`, which clashes with monitor3d's conda OpenCV. Train and runtime stay isolated (same reasoning as the Multical venv).
- **Path:** YOLOv11 **detection** (the RF-DETR path exists in the trainer but is unused). Config `configs/train_pallet.yaml`: model via `weights:` (e.g. `yolo11m.pt`; also `yolo26*` etc.), `imgsz`, `epochs`, `batch_size`, `workers`, warehouse augmentations incl. `camera_aug`, and one-shot export **`pt` + `onnx` + `openvino`** with `export_nms: false` / `export_opset: 17` (raw head matching `yolo_onnx`). LR/scheduler live in `configs/optimizers/yolo_optim.yaml`; `epochs` is authoritative in the train config.
- **Run from `trainer/isidet/`:** `conda activate isi-train && python scripts/run_train.py --config configs/train_pallet.yaml`. Run dirs: `runs/detect/models/yolo/<model>_e<epochs>_<timestamp>/`, each with weights, exports, and an auto-generated **`report.md`** (3-line expert overview + metrics/config/plots, via `scripts/make_train_report.py`).
- **Data prep (repo-root `scripts/`):** `prepare_labelme_dataset.py` (clean/rename/split LabelMe), `labelme_to_yolo.py` (LabelMe → YOLO, `--preserve-splits`). Multi-dataset merge: `tools/merge_pallet_dataset.py`.
- **Sanity check:** `trainer/isidet/run_test.sh` — interactive random-half prediction on a folder → annotated outputs + stats in `mytest_<model>_conf<conf>_<ts>/` (imgsz/FP16/CPU-fallback to avoid OOM).
- **WSL2 note:** heavy models (yolo11l / yolo26l) can swap-thrash the 12 GB WSL VM → transient EIO/bus-error crash (not disk-full). Levers: smaller model (yolo11m), `batch_size`, `workers`.
