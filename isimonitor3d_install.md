# ISI Monitor 3D — clean-PC installation (GPU, branch `main`)

Installs the full warehouse system from git on a fresh Ubuntu 22.04/24.04 or WSL2 machine
with an NVIDIA GPU: the metric engine (`backbone`), the perception producer (`isistream`),
the operator dashboard (`monitor_web`), the calibration studio (`isical`) and the
communication stack (`isicomms` gateway + Mosquitto in Docker).

Reference rig: RTX 5070 12 GB (Blackwell, sm_120), Linux/WSL2, two RTSP cameras.
Everything below was checked against the repository at commit `f06e4dc` (2026-09-10).

Ports used on the PC:

| Port | Who | Notes |
|---|---|---|
| 8000 | dashboard (`monitor_web`) | operator UI, START/STOP |
| 8080 | isicomms gateway | REST + `/ui` probe page + `/docs` |
| 1883 | Mosquitto | MQTT; 9001 = MQTT over WebSockets |
| 8300 | isical Studio | calibration |
| 9010 | engine ingest (loopback UDP) | isistream → backbone |
| 9001 | engine → dashboard (loopback UDP) | never leaves the host |

---

## 0. Prerequisites

1. **NVIDIA driver** with CUDA 12.x support installed on the host (on WSL2: the Windows
   driver, nothing to install inside WSL). Check: `nvidia-smi` shows the GPU.
2. **Miniforge** (conda + mamba): https://github.com/conda-forge/miniforge — install to
   `~/miniforge3`, run `conda init bash`, open a new shell.
3. **Docker Engine + compose v2** (for the comms stack). On WSL2, Docker Desktop with WSL
   integration is fine. Check: `docker compose version`.
4. **git**.
5. **apt packages** — only needed if you will calibrate with AprilGrid boards (the
   two-camera "calibrate-2cam" path). ChArUco-only calibration needs none.

   ```bash
   sudo apt install -y cmake libopencv-dev libeigen3-dev
   ```

   `cmake` must be 3.x (apt's 3.28 is fine; CMake 4 breaks the AprilTag build).
6. **Network**: the PC must reach both cameras' RTSP streams. Give the PC a static address
   on the camera LAN (see §7); the dashboard's gateway URL is tied to it.

WSL2 users: keep `networkingMode=mirrored` in `.wslconfig` if you use it, the code already
handles its quirks (application-layer UDP fragmentation at 1300 B, loopback ports kept
below the Windows reserved range). Ports 8000/8080/1883/8300 are reachable from Windows at
the host's IP.

---

## 1. Clone

```bash
cd ~
git clone https://github.com/IsitecVision/isi_monitor3d.git
cd isi_monitor3d
git checkout main
```

The clone contains code, tests, example configs and the *site* configs of the reference
rig (`config/backbone.yaml`, `config/zones.yaml`, …). It does **not** contain model
weights, calibration files or trained runs — those are gitignored and are produced or
copied in §5 and §6.

---

## 2. Runtime environment (`monitor3d`)

```bash
conda env create -f environment.yml -n monitor3d
conda activate monitor3d          # do this in every shell that runs or tests the system
```

Python is pinned to 3.10. The env brings OpenCV (headless), GStreamer + plugins,
`ffmpeg`/`ffprobe`, PyGObject, ONNX Runtime, OpenVINO, FastAPI, pytest, ruff, and installs
the repo itself in editable mode (`pip install -e .`), so `backbone`, `isistream`,
`calibration` and `isical` are importable.

**TensorRT engines (the deployed detector format) need the pip ONNX Runtime**, not the conda
one that `environment.yml` installs. After the env is created, switch it exactly as the
reference rig runs:

```bash
conda activate monitor3d
conda remove -n monitor3d --force onnxruntime          # drop the conda build, keep everything else
pip install onnxruntime-gpu==1.23.2 tensorrt-cu12==10.16.1.11
# the tensorrt wheel ships ~2.8 GB of Windows builder resources that Linux never loads:
rm -f ~/miniforge3/envs/monitor3d/lib/python3.10/site-packages/tensorrt_libs/libnvinfer_builder_resource_win_*
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
# expect CUDAExecutionProvider (and TensorrtExecutionProvider) in the list
```

This is exactly what the reference Docker image does (`docker/Dockerfile.app`);
`docker/environment.app.yml` is `environment.yml` minus the conda `onnxruntime` line.

Refresh later with `conda env update -f environment.yml -n monitor3d --prune` (re-run the
pip step afterwards).

Do **not** install the training stack (ultralytics/torch) into this env; it lives in its own
env (§10).

---

## 3. Calibration backend (Multical, isolated venv)

Multical pins an old OpenCV that would corrupt the runtime env, so it lives in its own venv
that the calibration code calls by absolute path.

```bash
conda activate monitor3d
bash calibration/setup_multical.sh
# → calibration/.venv-multical/bin/multical
```

Add `MULTICAL_VIEWER=0` in front of the command to skip the ~1 GB Qt/VTK 3D viewer if the PC
has no display. The AprilTag part of the script needs the apt packages from §0.

---

## 4. Dashboard

```bash
conda activate monitor3d
cd monitor_web && pip install -e ".[dev]" && cd ..
```

Add the launcher alias to `~/.bashrc` (the reference rig uses exactly this):

```bash
alias 3d='conda activate monitor3d && python -m monitor_web'
```

`3d` must be run from the repository root (it has no `cd`). Settings are read from
environment variables prefixed `MONITOR_WEB_` (port, UDP port, paths); the defaults match
this repo's layout, so none are required.

---

## 5. Models

Weights are not in git. You need:

| Model | Config key | Where it comes from |
|---|---|---|
| Pallet segmentation, e.g. `pallet.onnx` | `detection.onnx_path` | trained with `trainer/isidet` (§10) or copied from the reference rig / a release |
| Person pose `yolo11n-pose-dynamic.onnx` | `detection.pose_onnx_path` | exported from ultralytics (`yolo export model=yolo11n-pose.pt format=onnx dynamic=True opset=17`) or copied |
| SuperPoint + LightGlue ONNX (only for the *targetless* calibration path) | — | `models/README.md` has the two `curl` lines |

Put them under `models/` (or keep the trainer's run path) and reference them with
**absolute paths** in `config/backbone.yaml`.

Build the per-machine TensorRT engines (GPU required, do this before starting the system):

```bash
conda activate monitor3d
python tools/onnx_to_engine.py models/pallet.onnx --imgsz 320 --min-batch 1 --opt-batch 8 --max-batch 32
python tools/onnx_to_engine.py models/yolo11n-pose-dynamic.onnx --imgsz 640 --max-batch 4
# each writes MODEL.engine + MODEL.engine.json next to the source
```

An `.engine` is tied to this GPU and TensorRT version; the `.json` sidecar records that and
the runtime refuses a foreign engine. Point `onnx_path` / `pose_onnx_path` at the `.engine`
files (the backend is chosen by file suffix; a `.onnx` path runs ORT-CUDA instead).

Sanity check a model on a still image:

```bash
python tools/onnx_inspect.py models/pallet.onnx
python tools/detection_smoke.py --onnx models/pallet.onnx --image some_frame.jpg --annotate out.jpg
```

---

## 6. Configuration

Start from the tracked `config/backbone.yaml` (the reference rig's file) and change the
site-specific values. The dashboard's Settings modal can edit most of them, but it rewrites
the file atomically and drops YAML comments, so make a copy first if you want to keep notes.

Minimum edits in `config/backbone.yaml`:

- `node_id` — unique per PC (it becomes the MQTT prefix `isiMonitor3D/v1/<node_id>`).
- `cameras` — the two RTSP URLs with credentials; keep `latency_ms: 100`, `decoder: nvdec`.
  One camera only ⇒ Mode 1 (no triangulation), no other change needed.
- `calibration_path` — absolute path of the `calibration.json` produced in §7.
- `detection.onnx_path`, `detection.pose_onnx_path` — absolute paths from §5.
- `ingestion.mode: points` (the deployed topology; `isistream` refuses to start otherwise),
  `ingestion.points.max_skew_ms: 100`.
- `homography.pallet_state` — `enter_after: 5`, `presence_conf_min: 0.6` (site-validated
  2026-09-10).
- `metadata.sinks` — the `udp` sink on `127.0.0.1:9001` (dashboard) and the `mqtt` sink on
  `127.0.0.1:1883` with `prefix: isiMonitor3D/v1/<node_id>`.

Other files:

- `config/zones.yaml` — drawn in the dashboard (Settings → Floor zones). Set **Base height**
  to the platform height (0.304 m on the reference site) when pallets sit on a platform.
- `config/mode2/subscriptions.yaml` — which tracks get 3D; the tracked file is a good default.
- `config/danger_zones_object.yaml` — optional; copy the `.example` to enable proximity rings.
- `config/monitor_web_ui.yaml` — written by the dashboard. Set `gateway_url` in the
  dashboard's Settings (or edit the file) to `http://<this PC's static IP>:8080` once §8 is up.

---

## 7. Calibration

Two cameras → Mode 2, done in the isical Studio:

```bash
conda activate monitor3d
python -m isical            # http://localhost:8300
```

Print the boards from `tools/boards_print/` at 100 % scale. The Studio walks through
Intrinsic (≈25 ChArUco shots per camera, RMS ≤ 2 px) → Extrinsic (≈20 synced AprilGrid pairs
plus one floor ChArUco shot per camera) → Export, which writes `calibration.json` and can
stamp its path into `backbone.yaml`. The same steps exist as a CLI
(`python -m calibration.calibrate calibrate-2cam …`, see `CLAUDE.md`).

One camera → Mode 1: measure ≥5 floor points and run

```bash
python -m calibration.calibrate single-cam --camera-id cam_a --image-size 1920 1080 \
    --pair u1,v1,X1,Y1 --pair u2,v2,X2,Y2 --pair u3,v3,X3,Y3 --pair u4,v4,X4,Y4 --pair u5,v5,X5,Y5 \
    --output config/calibration.json
```

Cameras drift; treat calibration as a maintained artifact and re-run when the residual on
the dashboard's status panel exceeds 2 px.

---

## 8. Communication stack (isicomms gateway + Mosquitto)

```bash
cd ~/isi_monitor3d
docker compose -p on-prem -f isicomms/deploy/onprem/docker-compose.yml up -d --build
docker compose -p on-prem -f isicomms/deploy/onprem/docker-compose.yml ps
curl http://127.0.0.1:8080/healthz       # {"ok":true}
```

The project name `-p on-prem` is mandatory (without it compose creates a second stack that
collides on port 1883). Containers restart automatically after a reboot. If a system
Mosquitto already holds 1883: `sudo systemctl disable --now mosquitto`.

After any change to the `isicomms` package the gateway image must be **rebuilt**, not
restarted: `docker compose -p on-prem -f isicomms/deploy/onprem/docker-compose.yml build gateway && … up -d gateway`.
Never rebuild Mosquitto for code changes (it drops retained messages).

The broker allows anonymous local clients by design; do not expose 1883/8080 outside the
LAN. For a central cloud broker with TLS/auth use `isicomms/deploy/cloud/` and its README.

---

## 9. Run

**Operator mode (normal):**

```bash
cd ~/isi_monitor3d
3d                                 # dashboard on http://localhost:8000
```

Press **START** in the dashboard: it spawns the engine (`python -m backbone.runtime`) and
the producer (`python -m isistream`) and shows their logs; **STOP** reaps both. The status
panel goes green when both cameras are live and the UDP feed is fresh.

**Manual mode (two terminals, same config):**

```bash
conda activate monitor3d
python -m backbone.runtime --config config/backbone.yaml
python -m isistream --config config/backbone.yaml
```

**Headless (site deployment):** the dashboard is optional. Create two systemd units (paths
adjusted; `python` is the env's interpreter, e.g. `~/miniforge3/envs/monitor3d/bin/python`):

```ini
# /etc/systemd/system/isi-backbone.service
[Unit]
Description=ISI Monitor 3D metric engine
After=network-online.target docker.service
[Service]
User=<user>
WorkingDirectory=/home/<user>/isi_monitor3d
ExecStart=/home/<user>/miniforge3/envs/monitor3d/bin/python -m backbone.runtime --config /home/<user>/isi_monitor3d/config/backbone.yaml
Restart=always
RestartSec=3
[Install]
WantedBy=multi-user.target
```

```ini
# /etc/systemd/system/isistream.service
[Unit]
Description=ISI Monitor 3D perception producer
After=isi-backbone.service
[Service]
User=<user>
WorkingDirectory=/home/<user>/isi_monitor3d
ExecStart=/home/<user>/miniforge3/envs/monitor3d/bin/python -m isistream --config /home/<user>/isi_monitor3d/config/backbone.yaml
Restart=always
RestartSec=3
[Install]
WantedBy=multi-user.target
```

`sudo systemctl enable --now isi-backbone isistream`. Use absolute model and calibration
paths in `backbone.yaml` (systemd's working directory rules differ from the dashboard's).
Do not run the systemd units and the dashboard's START at the same time.

---

## 10. Optional: training environment

Only needed to train or re-export detectors (`trainer/isidet`) or generate synthetic data
(`trainer/isiGen`). It must stay separate from `monitor3d`.

```bash
conda env create -f isi-train.yml -n isi-train        # Python 3.11, torch cu128, ultralytics
conda run -n isi-train python -c "import torch; print(torch.cuda.get_device_capability())"
cd trainer/isidet && conda activate isi-train && python scripts/run_train.py --config configs/train_pallet.yaml
```

The run directory contains the `.onnx` export (raw head, opset 17) that §5 consumes.

---

## 11. Verify the installation

```bash
conda activate monitor3d
pytest -q                                   # backbone + calibration suites (~900 tests, no GPU or cameras needed)
cd monitor_web && pytest -q && cd ..        # dashboard suite
cd isicomms && pytest -q && cd ..           # gateway suite

python tools/rtsp_smoke.py rtsp://<camera-url>                 # capture sanity per camera
python tools/latency_probe.py online --config config/backbone.yaml --seconds 60   # p50/p95 < 200 ms
curl -s http://127.0.0.1:8000/api/status | python3 -m json.tool | head -30          # readiness green
curl -s http://127.0.0.1:8080/nodes                              # your node_id alive, both cameras
open http://127.0.0.1:8080/ui                                    # live probe page
```

End-to-end check on the floor: place a pallet in a zone, remove it. On `/zones` the zone
should show `palette_empty` within about half a second and `no_palette` within about two
seconds of removal.

---

## 12. Gotchas collected on the reference rig

- **Static IP.** The dashboard's `gateway_url` and the AGV consumers point at the PC's LAN
  address. A DHCP lease change shows up as "gateway unreachable". Reserve the address on
  the router or set it statically (reference site: 192.168.2.113/22, gateway 192.168.1.254,
  DNS 192.168.1.10).
- **VRAM.** Run one CUDA process for perception. Benchmarks or conversions on the GPU while
  the system runs can crash the live session (CUDA error 700).
- **`Killed`** in the dashboard log means host RAM exhaustion, usually orphaned engine
  processes; the supervisor reaps them on the next START.
- **Two cameras with different frame rates** are normal; pairing tolerance is 100 ms and the
  synchronizer never emits solo pairs from two healthy cameras (fixed 2026-09-10). If one
  camera dies the system continues on the other and recovers on its own.
- **Zone edges.** A pallet parked exactly on a polygon edge is held by a 15 cm exit margin;
  still, draw zones with some clearance.
- **Editing configs by hand** while the dashboard is open: the next Settings save rewrites
  the file. Restart (STOP/START) after any manual change to `backbone.yaml`.
