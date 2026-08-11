# ISI Monitor 3D — CPU deployment branch (`cpu`)

Single-camera, CPU-only variant of ISI Monitor 3D for a 32 GB RAM machine
without a GPU. One RTSP camera → zone-scoped pallet/carton detection +
person pose → metric `Track2D` over UDP/MQTT → operator dashboard + the
isicomms AGV gateway. **All inference is OpenVINO IR** (`model.xml` +
`model.bin`) on CPU — the ONNX Runtime / TensorRT / training tooling of the
GPU line (`main`) does not exist on this branch.

## What's different from `main`

| | `main` (GPU line) | `cpu` (this branch) |
|---|---|---|
| Cameras | 2 (Mode 2: homography + triangulation/Track3D) | **1 (Mode 1: homography only — no 3D)** |
| Inference | ONNX Runtime CUDA / native TensorRT engines | **OpenVINO IR, CPU** (`yolo_openvino`, `yolo_openvino_seg`, `yolo_openvino_pose`) |
| Models | `.onnx` / `.engine` under `trainer/` | **`.xml` IRs under `models/`** |
| Calibration | isical Studio + Multical (2-cam BA) | **in-app 4-point floor fit** (dashboard ▸ ruler) |
| Dev tools | isical/, trainer/ (isiGen + isidet), thesis/ … | stripped |
| Dashboard port | :8000 | **:8200** |

## Quick start (bare metal)

```bash
conda env create -f environment.yml            # → monitor3d-cpu
conda activate monitor3d-cpu
pip install --no-deps -e monitor_web -e isicomms

# dashboard (spawns backbone + isistream on START):
MONITOR_WEB_PORT=8200 python -m monitor_web    # or the `3d_cpu` alias
```

Open `http://localhost:8200/`:
1. **Settings ▸ Cameras** — set the cam_a RTSP URL (leave Cam 2 empty → Mode 1).
2. **Calibrate** (ruler button) — click the 4 corners of a reference pallet on
   the floor (TL→TR→BR→BL) + its size → writes `config/mode1/calibration.json`.
3. **Settings ▸ Zones** — draw the floor zones (base height for platforms).
4. **START** — detection + pose come up on the OpenVINO models in `models/`.

## Docker

```bash
./up.sh          # app (:8200) + mosquitto (:1883) + isicomms gateway (:8080)
```

## Models

Ship IRs under `models/<name>/model.xml` (+ `model.bin` beside it):

- `models/pallet_seg_openvino/` — pallet/carton/polybag seg (zone detection)
- `models/yolo11n_pose_openvino/` — person pose (skeletons + person tracks)

Convert new models once in any env with openvino:
`ovc model.onnx --output_model model.xml`. A path containing `pose` is
offered in the pose dropdown; everything else in the object-model dropdown.

## Port map / coexistence with a GPU-line install

| Service | Port |
|---|---|
| Dashboard | **8200** |
| Engine UDP sink (dashboard bus) | **9003** |
| isistream → engine points ingest | **9012** |
| MQTT broker (shared) | 1883 — this node publishes as `…_cpu` |
| isicomms gateway | 8080 |

Rules when the GPU line runs on the same machine:
- **One capture stack per camera id** — the shared frame bus lives at
  `/dev/shm/isi3d_frame_<cam>` (name-fixed); never run both stacks' capture
  against the same camera ids at once.
- **One mosquitto on 1883** — reuse the existing broker; don't `up.sh` a
  second stack's broker while the first runs.

## Tests

```bash
pytest tests calibration/tests        # runtime + calibration
cd monitor_web && pytest              # dashboard
cd isicomms && pytest                 # gateway
```
