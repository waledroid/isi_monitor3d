# ISI Monitor 3D CPU — clean-PC installation (branch `cpu`)

Installs the single-camera, CPU-only variant from git on a fresh Ubuntu 22.04/24.04 or WSL2
machine **without a GPU** (reference: 32 GB RAM, one RTSP camera). Inference runs on
OpenVINO IR models on the CPU; there is no ONNX Runtime, TensorRT, isical Studio or trainer
on this branch. Calibration is the dashboard's in-app four-point floor fit.

Checked against branch `cpu` at commit `d809a4d` (2026-09-10). The branch's own `README.md`
is the authoritative short version; the `CLAUDE.md`, `docker/README.md`,
`monitor_web/README.md` and `config/backbone.yaml.example` in the tree are GPU-line
documents left behind and should be ignored for installation.

Ports used on the PC (chosen so a GPU-line install can coexist on the same box):

| Port | Who | Notes |
|---|---|---|
| **8200** | dashboard (`monitor_web`) | must be set with `MONITOR_WEB_PORT=8200`; the code default is 8000 |
| 8080 | isicomms gateway | REST + `/ui` + `/docs` |
| 1883 | Mosquitto | MQTT (no WebSockets listener on this branch) |
| 9012 | engine ingest (loopback UDP) | isistream → backbone |
| 9003 | engine → dashboard (loopback UDP) | offset from the GPU line's 9001 |

---

## 0. Prerequisites

1. **Miniforge** (conda + mamba) installed to `~/miniforge3`, `conda init bash`, new shell.
2. **Docker Engine + compose v2** for the comms stack (Docker Desktop with WSL integration
   is fine on WSL2). No NVIDIA container toolkit needed.
3. **git**. No apt packages are required: OpenCV, GStreamer, ffmpeg/ffprobe and OpenVINO all
   come from the conda environment.
4. **Network**: the PC must reach the camera's RTSP stream. Give the PC a static address on
   the camera LAN; AGV consumers and the dashboard's gateway URL are tied to it.

WSL2: loopback ports are deliberately low (9003/9012) because ports above ~49152 fail to
bind under mirrored networking. Nothing to configure.

---

## 1. Clone

```bash
cd ~
git clone https://github.com/IsitecVision/isi_monitor3d.git isi_monitor3d_cpu
cd isi_monitor3d_cpu
git checkout cpu
```

Unlike the GPU line, this branch **ships everything it needs to run**: the two OpenVINO IR
models under `models/` (pallet segmentation and person pose, about 12 MB, tracked in git),
a starting Mode-1 calibration in `config/mode1/calibration.json`, and the site's
`config/backbone.yaml` and `config/zones.yaml`.

---

## 2. Environment (`monitor3d-cpu`)

```bash
conda env create -f environment.yml -n monitor3d-cpu
conda activate monitor3d-cpu
pip install --no-deps -e monitor_web -e isicomms
```

Python 3.10, OpenVINO ≥ 2024, headless OpenCV 4.13 with GStreamer and FFmpeg, PyGObject,
FastAPI, pytest, ruff. The repo itself is installed editable by the env file. `--no-deps`
matters: the dashboard and gateway packages would otherwise pull `opencv-python` and collide
with the conda OpenCV. Never install Multical, ultralytics, torch or onnxruntime into this env.

Refresh later with `conda env update -f environment.yml -n monitor3d-cpu --prune`.

Launcher alias for `~/.bashrc` (the reference machine uses exactly this):

```bash
alias 3d_cpu='cd ~/isi_monitor3d_cpu && conda activate monitor3d-cpu && MONITOR_WEB_PORT=8200 python -m monitor_web'
```

---

## 3. Models

Nothing to download. `config/backbone.yaml` already points at:

| Model | Config key | Path |
|---|---|---|
| Pallet / carton / polybag segmentation | `detection.model_xml` | `models/pallet_seg_openvino/model.xml` |
| Person pose | `detection.pose_model_xml` | `models/yolo11n_pose_openvino/model.xml` |

Measured on the reference CPU: 24 ms per inference at 320 px for the pallet model, 31 ms at
480 px for pose; the pipeline runs at 11 to 13 pairs per second.

To deploy a newly trained model, convert its ONNX export once in any environment that has
OpenVINO and drop the pair into `models/<name>/`:

```bash
ovc model.onnx --output_model models/<name>/model.xml
```

A directory whose path contains `pose` is offered in the dashboard's pose dropdown;
everything else in the object-model dropdown. Class names are read from the IR itself.

Smoke-test a model on a still image (no GPU, no cameras):

```bash
python tools/detection_smoke.py --xml models/pallet_seg_openvino/model.xml --image shot.jpg --input-size 320 --annotate out.jpg
```

---

## 4. Configuration

Edit `config/backbone.yaml` (the live file is the reference on this branch; the `.example`
is stale). The dashboard's Settings modal can do most of it, but it rewrites the file
without comments, so restart after any manual edit.

- **Absolute paths** — the tracked file is rooted at `/home/aatanda/isi_monitor3d_cpu`.
  If your clone lives elsewhere, replace that prefix in `calibration_path`, `zones_path`,
  `subscriptions_path`, `detection.model_xml`, `detection.pose_model_xml`:

  ```bash
  sed -i "s#/home/aatanda/isi_monitor3d_cpu#$PWD#g" config/backbone.yaml
  ```

- `node_id` — unique per PC; the MQTT prefix becomes `isiMonitor3D/v1/<node_id>`
  (reference: `Sortie_Machine_4_cpu`). Keep the `_cpu` suffix if a GPU node shares the broker.
- `cameras.cam_a` — the RTSP URL with credentials (main stream as `source`, the camera's
  substream as `detect_source`), `decoder: software`, `capture_fps: 25`, `output_wh: [1280, 720]`.
  One camera only; there is no Cam 2 on this branch.
- `ingestion.mode: points`, `ingestion.points.listen_port: 9012`, `max_skew_ms: 100`.
- `metadata.sinks` — `udp` on `127.0.0.1:9003` (dashboard) and `mqtt` on `127.0.0.1:1883`.
- `homography.pallet_state` — `enter_after: 5`, `presence_conf_min: 0.4` on this branch
  (the GPU site moved to 0.6 on 2026-09-10; raise it here if the same behaviour is wanted).
- `detection` — `plugin: yolo_openvino_seg`, `device: CPU`, `zone_imgsz: 320`,
  `pose_imgsz: 480`, `pose_every_n: 3`.

`config/monitor_web_ui.yaml` holds the dashboard's zone patches and `gateway_url`. Set
`gateway_url` to `http://<this PC's static IP>:8080` once §6 is up (or leave it empty for a
muted hint in the status panel).

---

## 5. Camera, calibration and zones (in the dashboard)

```bash
3d_cpu            # or: MONITOR_WEB_PORT=8200 python -m monitor_web  → http://localhost:8200
```

1. **Settings → Cameras**: set the cam_a RTSP URL. The field prefills with the reference
   site's camera when unconfigured.
2. **Calibrate** (ruler button): click the four corners of a reference pallet lying on the
   floor, TL → TR → BR → BL, enter its size (reference: 1.200 × 0.800 m). This writes
   `config/mode1/calibration.json` (`calibration_mode: single_cam_4pt`) with the world
   origin at the pallet's top-left corner. Re-run whenever the camera is moved.

   CLI equivalent with surveyed floor points, at least four pairs of pixel and metre
   coordinates:

   ```bash
   python -m calibration.calibrate single-cam --camera-id cam_a --image-size 1280 720 \
       --pair u1,v1,X1,Y1 --pair u2,v2,X2,Y2 --pair u3,v3,X3,Y3 --pair u4,v4,X4,Y4 \
       --output config/mode1/calibration.json
   ```

3. **Settings → Floor zones**: draw the zones on the camera image. Set **Base height** to
   the platform height (0.304 m on the reference site) when pallets sit on a platform
   support; leave 0 for pallets on the floor.
4. **START**. The dashboard spawns the engine and the producer itself (both with
   `OMP_NUM_THREADS=2`, sized to share the CPU) and shows their logs. **STOP** reaps both.

---

## 6. Communication stack (isicomms gateway + Mosquitto)

```bash
cd ~/isi_monitor3d_cpu
docker compose -p on-prem -f isicomms/deploy/onprem/docker-compose.yml up -d --build
curl http://127.0.0.1:8080/nodes          # your node_id appears once the engine runs
```

Same stack as the GPU line minus the WebSockets listener. Keep `-p on-prem` (a second
project name spins a duplicate broker on 1883). Rebuild the gateway image after any change
to the `isicomms` package; never rebuild Mosquitto for code changes. Only **one** broker per
machine: if a GPU-line install already runs the stack, do not start a second one, just point
this node at it (it publishes under its own `node_id`).

The broker accepts anonymous local clients; keep 1883 and 8080 inside the LAN. For a central
TLS broker use `isicomms/deploy/cloud/`.

---

## 7. Manual and headless operation

Two terminals, same config file:

```bash
conda activate monitor3d-cpu
python -m backbone.runtime --config config/backbone.yaml
python -m isistream --config config/backbone.yaml
```

Headless site deployment: two systemd units, dashboard optional. Replace `<user>`; the
interpreter is `~/miniforge3/envs/monitor3d-cpu/bin/python`.

```ini
# /etc/systemd/system/isi-backbone.service
[Unit]
Description=ISI Monitor 3D CPU metric engine
After=network-online.target docker.service
[Service]
User=<user>
WorkingDirectory=/home/<user>/isi_monitor3d_cpu
Environment=OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
ExecStart=/home/<user>/miniforge3/envs/monitor3d-cpu/bin/python -m backbone.runtime --config /home/<user>/isi_monitor3d_cpu/config/backbone.yaml
Restart=always
RestartSec=3
[Install]
WantedBy=multi-user.target
```

```ini
# /etc/systemd/system/isistream.service
[Unit]
Description=ISI Monitor 3D CPU perception producer
After=isi-backbone.service
[Service]
User=<user>
WorkingDirectory=/home/<user>/isi_monitor3d_cpu
Environment=OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
ExecStart=/home/<user>/miniforge3/envs/monitor3d-cpu/bin/python -m isistream --config /home/<user>/isi_monitor3d_cpu/config/backbone.yaml
Restart=always
RestartSec=3
[Install]
WantedBy=multi-user.target
```

`sudo systemctl enable --now isi-backbone isistream`. Do not run the units and the
dashboard's START at the same time.

Docker alternative for the whole thing: `./up.sh` builds a CPU image and starts app
(:8200), Mosquitto and the gateway; `./up.sh down` stops them.

---

## 8. Verify

```bash
conda activate monitor3d-cpu
pytest -q tests calibration/tests           # ~740 tests, hermetic
cd monitor_web && pytest -q && cd ..        # ~400 tests
cd isicomms && pytest -q && cd ..

python tools/rtsp_smoke.py rtsp://<camera-url>
curl -s http://127.0.0.1:8200/api/status | python3 -m json.tool | head -30   # readiness green, cam_a live
curl -s http://127.0.0.1:8080/nodes                                          # node alive
```

Floor check: push a pallet into a zone and out again. `/zones` should show `palette_empty`
within about half a second and `no_palette` within about two seconds of removal.

---

## 9. Gotchas

- **Port 8200 is not automatic.** Without `MONITOR_WEB_PORT=8200` the dashboard starts on
  8000 and the process reaper keys on the port. The dashboard must listen on UDP 9003 to
  match the engine's sink; a test pins that wiring.
- **Absolute paths** in `config/backbone.yaml` (see §4) are the first thing to check when a
  freshly cloned install cannot find its model or calibration.
- **CPU load.** The engine and producer are caged to two threads each so they share the
  machine with the dashboard and the broker. Expect 11 to 13 pairs per second; do not run
  other heavy inference on the same box while the system is live.
- **Static IP** for the PC on the camera LAN; a DHCP change surfaces as "gateway unreachable"
  in the dashboard.
- **Zone edges.** A 15 cm exit margin keeps a pallet parked on a polygon edge from flapping
  the count; still draw zones with clearance.
- **Keeping the branch current.** Fixes land on `main` first and are cherry-picked onto
  `cpu`; the branch predates the GPU line's class-presence wire (no `cls` on the zone
  message) and the étagère feature.
