# ISI Monitor 3D — Hardware Spec Sheet

**Project:** ISI Monitor 3D (industrial vision Backbone for Isitec)
**Last updated:** 2026-06-24
**Scope:** the physical rig used for development/validation — cameras, calibration targets, and compute host. Camera makes/codecs were confirmed by live probing (RTSP `Server:` header, MAC OUI, HTTP auth banner); board geometry is the project's `calib.yaml` / `tools/boards_print/boards.yaml`.

---

## 1. Cameras (Mode-2 rig: cam_a + cam_b)

Two **white-label IP cameras** (neither is a true Hikvision/Dahua despite the path schemes). The Backbone's RTSP ingest is codec-aware, so the mixed H.264/H.265 rig works without per-camera special-casing.

| | **cam_a** | **cam_b** |
|---|---|---|
| Role | left/primary | right/secondary |
| IP : port | `192.168.1.88 : 554` | `192.168.1.108 : 554` |
| Credentials | `admin` / `admin` | `admin` / `admin123` |
| Make (probed) | Generic **"Hipcam RealServer/V1.0"** firmware (HiP2P / CamHi / iCSee family) | **Dahua-style** firmware (`realmonitor` path scheme) |
| Identifying evidence | RTSP `Server: Hipcam RealServer/V1.0`; HTTP `401 Basic realm="index.html"`; MAC `00:3e:a5:c2:16:d2` (OUI **unregistered**) | RTSP path `/cam/realmonitor?channel=N&subtype=M`; Hikvision/Hipcam paths 404 |
| Video codec | **H.264** (High/Main) + PCM A-law **audio** track | **H.265 / HEVC** (Main), video-only |
| Main-stream resolution | 1920 × 1080 | 2048 × 1536 (was 2880 × 1620; lowered on-camera) |
| Frame rate | ~20 fps | 20 fps |
| Main-stream bitrate | — | ~2.3 Mbps (≈0.036 bits/px — raise to ~5–6 Mbps CBR to kill motion macroblocking) |
| **Main-stream RTSP URL** | `rtsp://admin:admin@192.168.1.88:554/1` | `rtsp://admin:admin123@192.168.1.108/cam/realmonitor?channel=1&subtype=0` |
| Sub-stream RTSP URL | `…/11` (main) / `…/12` (sub) — Hipcam convention | `rtsp://admin:admin123@192.168.1.108/cam/realmonitor?channel=1&subtype=1` (704 × 480) |

**Notes**
- Credentials are **default LAN credentials** on an RFC-1918 segment — not internet-exposed. Change them for production.
- cam_a carries an audio track; the ingest pipeline intentionally selects the video RTP only, so it is unaffected.
- "Works in VLC" does **not** prove the Backbone will link a stream — verify the codec with `ffprobe -rtsp_transport tcp <url>`. The pipeline supports H.264 and H.265 (`rtph264depay/avdec_h264` or `rtph265depay/avdec_h265`, chosen at start from the probed codec).

---

## 2. Calibration targets

Two printed targets, both produced by `python -m calibration.calibrate gen-boards` and stored as PNGs in `tools/boards_print/` (`charuco_main.png`, `april_0.png … april_5.png`). Print at **100 % scale, 0 margins**, mount flat/rigid.

### 2.1 ChArUco board — intrinsics + floor (world) anchor

| Parameter | Value |
|---|---|
| Type | ChArUco (chessboard + ArUco markers) |
| Grid | **5 × 7 squares** |
| Square length | **0.035 m** (35 mm) |
| Marker length | **0.026 m** (26 mm) |
| ArUco dictionary | **DICT_5X5_50** |
| ArUco markers on board | 17 (multical `num_ids=17`) |
| Calibration features | **24 interior corners** = (5−1) × (7−1) |
| Printed board size | ≈ **175 × 245 mm** (5·35 × 7·35) — fits A4 portrait |
| Used by | Stage 1 `multical intrinsic` (per-camera K, D) + the per-camera floor ChArUco shot (world anchor) |
| Auto-snap gate | ≥ 12 ChArUco corners, sharp + steady + novel pose; target **25 shots/camera** |

### 2.2 AprilGrid target — extrinsics (camera-to-camera poses)

A target of **6 disjoint-ID AprilGrid boards** shown together across the shared volume so both cameras see overlapping tags.

| Parameter | Value |
|---|---|
| Type | AprilGrid (×6 boards: `april_0 … april_5`) |
| Tags per board | **1 × 2** (1 wide, 2 tall) |
| Tag family | **t36h11** |
| Tag length | **0.11 m** (110 mm) |
| Tag spacing | **0.2** (ratio → ~22 mm gap between tags) |
| Tag IDs | disjoint, `start_id` = 0, 2, 4, 6, 8, 10 → **12 tags total, IDs 0–11** |
| Printed board size | ≈ **110 × 242 mm** each (1·110 × [2·110 + 0.2·110]) |
| Used by | Stage 2 `multical calibrate --fix_intrinsic` (rig R, t with K held fixed) |
| Auto-snap gate | ≥ 4 AprilTags per camera; target **20 synchronized pairs** |

> AprilGrid extrinsics require the `apriltags2-ethz` build (system deps: `cmake libopencv-dev libeigen3-dev`). The ChArUco intrinsics path needs none of this.

---

## 3. Compute host

| | Development / v1 | Production target (out of v1 scope) |
|---|---|---|
| Platform | RTX 5070 workstation, Linux / **WSL2** | Jetson Orin NX 16 GB (Seeed reComputer J4012) |
| GPU | NVIDIA **RTX 5070 12 GB**, Blackwell **sm_120** | Orin NX integrated GPU |
| CUDA / runtime | CUDA 12.9, **ONNX Runtime 1.25** (CUDAExecutionProvider) | ONNX Runtime (Jetson build) |
| HEVC decode | NVDEC `nvh265dec` available (pipeline uses CPU `avdec_h265`) | NVDEC |
| CPU | 8 logical cores (probed) | — |
| Portability | Same `.onnx`, same `calibration.json`, same UDP schema — the port is a deploy/env job, not a code change | |

**VRAM budget:** 12 GB shared across the Backbone detector + per-zone dashboard sessions; concurrent CUDA sessions are admission-guarded (≥1.5 GB free) to avoid CUDA-700.

---

## 4. Network

- All devices on the LAN `192.168.1.0/24` (RFC-1918, not internet-exposed).
- Link to cam_b measured healthy: **0 % packet loss, ~0.8 ms RTT** — bitrate, not the network, drives any cam_b motion artifacts.
- RTSP transport: **TCP** (`protocols=tcp`, `latency=100 ms`, drop-on-latency) for both cameras.

---

### Provenance
- Camera identities: live probe (`ffprobe`, raw RTSP `OPTIONS`, `curl -I`, MAC/OUI lookup), 2026-06-24.
- Board geometry: `isical/data/<project>/calib.yaml` + `tools/boards_print/boards.yaml`.
- Compute/network: `CLAUDE.md` hardware section + on-host measurement.
