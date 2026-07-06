# Plan — NVIDIA DeepStream ingestion front-end (dual-camera, GPU decode → batched delivery)

## Why

Replace the Backbone's CPU ingestion front-end (per-camera `rtspsrc→avdec→videoconvert` pipelines + `FrameSynchronizer` + `FrameBus`) with a single DeepStream pipeline that does **acquisition, synchronization, NVDEC decode, GPU preprocessing and batched delivery** in hardware. What it buys:

| Today (CPU path) | DeepStream |
|---|---|
| 2× `avdec_h264` software decode (~15–30 % CPU total, scales with resolution) | 2× **NVDEC** hardware decode (~0 % CPU, dedicated silicon — doesn't even compete with CUDA compute) |
| `videoconvert` I420→BGR on CPU per frame (6 MB copies) | `nvvideoconvert` on GPU |
| Python `FrameSynchronizer` (skew window, sticky-degraded) + `FrameBus` | `nvstreammux` — hardware-grade batching: `batch-size=2`, `sync-inputs=1`, and `batched-push-timeout` gives the **degraded solo-emit behavior for free** (partial batch after timeout when one camera stalls) |
| CPU letterbox ~9 ms/pair (already optimized) + H2D copy into ORT | `nvdspreprocess`/`nvvideoconvert` crop+scale+pad+normalize **on GPU**, zero-copy into inference (Phase 2) |

**Strategic alignment**: the production target is Jetson Orin NX, where DeepStream is the native JetPack stack and the CPU is scarce — this work IS the Jetson port's ingestion layer. On the dev RTX 5070 the win is moderate (frees ~1–2 cores + kills the residual pre-processing); on Jetson it's decisive.

## Architecture

```
nvurisrcbin(rtsp cam_a) ─┐                                     ┌─ Phase 1: nvvideoconvert→RGBA → appsink → numpy BGR FramePair
                          ├─ nvstreammux (batch=2, sync-inputs, ┤
nvurisrcbin(rtsp cam_b) ─┘   batched-push-timeout=100ms)        └─ Phase 2: nvdspreprocess tensors → ORT IOBinding (GPU-resident)
```

- Per-frame `NvDsFrameMeta` carries `source_id` + `buf_pts` → maps to `Frame(camera_id, capture_ts)`; `capture_ts` = arrival wall-clock at the appsink probe (same policy as today — single capture clock) with the RTP/NTP timestamp path documented as the future upgrade.
- The batch replaces `FramePair` assembly: one appsink callback = one ready pair (or solo when a camera is down → the sticky-degraded semantics survive unchanged at the contract level).
- Integration seam: a new **`deepstream` ingestion engine** selected by `ingestion.engine: deepstream` in backbone.yaml (default `gstreamer` = today's path, full rollback by config). It bypasses `FrameSynchronizer`/`FrameBus` and feeds `_pipeline_loop` a `get_latest_pair()` (latest-only slot, same drop-old doctrine as everything else). The 5-seam plugin registry is untouched — this replaces the *composition* of FrameSource+Sync+Bus for the pair case, while `FrameSource` plugins remain for replay/tests/Mode-1.

## Phase 0 — feasibility spike (1 day, throwaway)

The go/no-go gates, in order:

1. **WSL2 + DeepStream**: officially unsupported; NVDEC works in WSL2 (driver ≥ 510) but DeepStream must run from the official container (`nvcr.io/nvidia/deepstream:7.x`) with `--gpus all`. Spike: run `deepstream-app` with both site RTSP URLs inside the container; confirm 2× NVDEC decode (check `nvidia-smi dmon -s u` enc/dec columns) and `nvstreammux` batching at the cameras' real rates (cam_a 18 fps / cam_b 25 fps, mixed).
2. **Blackwell (sm_120) support**: needs DeepStream ≥ 7.1 / CUDA 12.6+. Verify the container's nvv4l2decoder initializes on the 5070.
3. **pyds bindings**: python bindings inside the container for batch-meta access at appsink.
4. **The ABI trap (biggest practical risk)**: our conda env's GStreamer 1.28 (conda-forge) is ABI-incompatible with DeepStream's plugins (built against Ubuntu's system GStreamer). The backbone process must therefore run either (a) **inside the DeepStream container** (recommended — matches the deploy/ compose direction and the Jetson story) or (b) on system python + system GStreamer with the monitor3d deps pip-installed. Spike must prove one of these can `import backbone` AND load nvstreammux in one process.

If WSL2 blocks hard: the fallback spike is `nvh264dec` (GStreamer's plain NVDEC element, no DeepStream) inside the existing pipeline template — decode offload only, none of the batching — still worth having as `decoder: nvdec` option.

### Phase 0 results (2026-07-05, branch `deepstream`)

| Check | Result |
|---|---|
| NVDEC libs in WSL2 | ✅ `libnvcuvid.so` + `libnvidia-encode.so` present (driver 591.86) |
| Docker GPU passthrough + `video` capability | ✅ (`NVIDIA_DRIVER_CAPABILITIES=compute,utility,video` injects nvcuvid) |
| NVDEC functional on live camera | ✅ ffmpeg `h264_cuvid` decoded cam_a RTSP; decoder engine active |
| **conda GStreamer 1.28 has `nvcodec`** | ✅ `nvh264dec`/`nvh265dec`/`nvav1dec` + `cudaconvert(scale)` + `cudadownload` — NOT expected; removes the ABI-trap for the fallback path entirely |
| Full-GPU pipeline in the EXISTING env | ✅ `rtspsrc ! rtph264depay ! h264parse ! nvh264dec ! cudaconvertscale ! CUDAMemory,BGR ! cudadownload` ran live against cam_a (decoder util 2%, no errors) |
| DeepStream 7.1 container on Blackwell | ❌ **hard refusal** — entrypoint: "RTX 5070 … not yet supported in this version"; CUDA 12.6 lacks sm_120 kernels. DS 7.1 is dead for this dev box (fine for Jetson later). |
| DeepStream **8.0** on Blackwell | ✅ GPU accepted; `nvstreammux`/`nvv4l2decoder`/`nvurisrcbin`/`nvvideoconvert`/`nvdspreprocess` all load |
| **LIVE dual-camera batched pipeline** (the money test) | ✅ `2× rtspsrc → nvv4l2decoder → nvstreammux(batch=2, 100ms timeout) → nvvideoconvert → NVMM RGBA` ran 25 s against both site cameras, clean; decoder 1%, GPU 4% |
| pyds in DS 8.0 container | ⚠ not preinstalled (Python 3.12 in container); DS 8 ships bindings separately — a Phase-1 image-build step (`pyds` wheel / `pyservicemaker`), not a blocker |

### PHASE 0 VERDICT: **GO**

Use `nvcr.io/nvidia/deepstream:8.0-triton-multiarch` as the Phase-1 base image (the 7.1 image can be deleted from this box). Two implementation notes learned in the spike: containers need `-e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video` and `--network host` (RTSP to the 192.168.2.x cameras); the container's Python is 3.12 (backbone targets 3.10 — pip-installing `backbone` into the container needs the version bounds checked, or the appsink consumer stays a thin bridge process).

**Consequence for sequencing**: a new "Phase 0.5" is available at near-zero risk — add `decoder: nvdec` (+ GPU colorspace) as a source option in the EXISTING `PIPELINE_TEMPLATE`, no DeepStream, no container, no ABI issue, works in the current conda env for both the Backbone and the dashboard hub. Captures the decode + colorconvert CPU offload immediately; DeepStream proper (nvstreammux batching, Phase 1-2) remains the Jetson-aligned follow-up.

## Phase 1 — DeepStream ingestion engine (CPU-BGR delivery, detector untouched)

New module `backbone/ingestion/deepstream_source.py`:

- Builds the pipeline: `nvurisrcbin uri=rtsp://…` ×2 (built-in RTSP reconnect — replaces our source-crash/exit handling) → `nvstreammux batch-size=2 width=1920 height=1080 batched-push-timeout=100000 sync-inputs=1 attach-sys-ts=1` → `nvvideoconvert` → `capsfilter video/x-raw(memory:NVMM),format=RGBA` → `appsink`.
- Appsink callback: iterate `NvDsBatchMeta.frame_meta_list`; for each frame map its `NvBufSurface` slice to CPU (`pyds.get_nvds_buf_surface` → numpy view → `cv2.cvtColor` RGBA→BGR *once, small* — or request BGRx from nvvideoconvert to skip it), build `Frame(camera_id=source_id_map[source_id], capture_ts=now)` and assemble the `FramePair`; publish into a one-slot latest holder (the FrameBus drop-old doctrine, maxsize 1).
- `Orchestrator._build()`: `ingestion.engine == "deepstream"` → construct `DeepStreamPairSource(cameras, mux_cfg)` instead of sources+sync+bus; `_pipeline_loop` reads `pair_source.get_latest(timeout=0.5)`. `frames_by_camera` counters + `source_status` fed from the frame-meta stream (source present in batches = alive; absent > N s = down) so the STATUS fps rows and degradation reporting keep working unchanged.
- Config:
  ```yaml
  ingestion:
    engine: deepstream          # gstreamer (default) | deepstream
    deepstream:
      mux_width: 1920
      mux_height: 1080
      batched_push_timeout_ms: 100   # = today's degraded_emit_after_ms role
  ```
- **Runtime packaging**: extend `deploy/onprem/docker-compose.yml` with a `backbone` service built FROM the DeepStream base image + `pip install -e .`; the dashboard keeps spawning the backbone locally when engine=gstreamer, and targets the container (or `docker compose run`) when deepstream — exact spawn mechanics decided in Phase 1 (simplest: supervisor command becomes configurable, `backbone_cmd` in monitor_web Settings).

Tests (hermetic where possible): pair assembly + latest-only slot + source-status derivation unit-tested with faked frame-meta structures; the pipeline itself gets a `@pytest.mark.deepstream` live-rig test (skipped without the container). Existing 617 tests untouched (default engine unchanged).

## Phase 2 — GPU-resident preprocessing + zero-copy inference

- Replace appsink-to-CPU with `nvdspreprocess` (or `nvvideoconvert` + custom CUDA): per-source scale+letterbox-pad+FP32-normalize into a batched NCHW tensor **on GPU** — this deletes `batch_letterbox` entirely for the deepstream path.
- Feed ORT via **IOBinding**: wrap the NvBufSurface/tensor as a `__dlpack__` capsule (via cupy) → `io_binding.bind_input(..., device_type='cuda')`. No H2D copy, no CPU float conversion. Detector plugin gets an optional `detect_gpu(batch_tensor, metas)` path; decode/NMS outputs stay as today.
- Expected effect on the 5070: pipeline step drops from ~50 ms to ≈ GPU-inference-only (~37 ms) + ms-level glue → ~25 pairs/s ceiling; on Jetson it's the difference between feasible and not.
- The dashboard preview keeps its own CPU path (it needs JPEG anyway); zone workers unaffected.

## Phase 3 — Jetson notes (deferred)

Same pipeline strings run on JetPack with two renames (`nvv4l2decoder` is default there; NVMM semantics identical). The deepstream engine + container packaging from Phase 1 is the Jetson deployment artifact; only the compose base image tag changes (`-l4t`).

## Risks / honest caveats

- **WSL2 is not a supported DeepStream platform** — Phase 0 exists precisely to fail fast. If it fails, keep the `nvh264dec` decode-offload fallback and defer full DeepStream to the Jetson.
- **Calibration/pixel coordinates**: nvstreammux scales all sources to mux W×H — set to 1920×1080 (calibration frame) so downstream geometry is untouched. (If ever reduced, the deferred detection→calibration scale guard becomes prerequisite.)
- **capture_ts semantics** change subtly (batch arrival vs per-frame appsink arrival) — latency KPI measurement stays consistent because both cameras share the batch clock; document in the module.
- **Ops surface grows**: a CUDA-context-bearing container in the loop; the VRAM budget gains one context but Phase 2 removes the letterbox/H2D — net ≈ neutral on the 5070, positive on Jetson.
- Rollback is always `ingestion.engine: gstreamer`.

## Sequencing

Phase 0 spike (go/no-go) → Phase 1 (engine + container, CPU delivery — shippable, decode/sync/batch wins) → Phase 2 (GPU tensors + IOBinding — the big latency win) → Phase 3 rides the Jetson port.
