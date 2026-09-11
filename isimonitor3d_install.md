# ISI Monitor 3D — install (GPU, branch `main`)

Ubuntu 22.04/24.04 or WSL2 · NVIDIA GPU · two RTSP cameras.
Ports: dashboard 8000 · gateway 8080 · MQTT 1883 · isical 8300 · loopback UDP 9010/9001.

## 1. Install

```bash
sudo apt install -y git curl
git clone https://github.com/IsitecVision/isi_monitor3d.git
cd isi_monitor3d && git checkout main
./install.sh gpu                 # add --env NAME to use/create another conda env (default monitor3d)
```

Each stage prints DONE, NEEDS YOU (a manual step, listed above the summary) or FAILED.
Fix the NEEDS YOU items and rerun the same command; done stages are skipped.

```bash
./install.sh gpu --dry-run       # preview
./install.sh gpu --list          # stage ids
./install.sh gpu --only comms    # one stage
./install.sh gpu --skip comms    # skip one (broker on another machine)
./install.sh gpu --systemd       # also write + enable the two service units (sudo)
```

## 2. NEEDS YOU steps

**Models** (not in git) — copy from the reference PC, then build the engines:
```bash
mkdir -p models      # pallet.onnx + yolo11n-pose-dynamic.onnx -> models/
python tools/onnx_to_engine.py models/pallet.onnx --imgsz 320 --min-batch 1 --opt-batch 8 --max-batch 32
python tools/onnx_to_engine.py models/yolo11n-pose-dynamic.onnx --imgsz 640 --max-batch 4
```

**Config** — `nano config/backbone.yaml`: `node_id` (unique per PC) · `cameras` RTSP URLs ·
`calibration_path` and `detection.onnx_path` / `pose_onnx_path` (absolute `.engine` paths) ·
`ingestion.mode: points` · `metadata.sinks` udp `127.0.0.1:9001` + mqtt `127.0.0.1:1883`.

**Calibration** — `python -m isical` → http://localhost:8300 → Intrinsic → Extrinsic → Export.
Boards: print `tools/boards_print/*.png` at 100 %. One camera (Mode 1): `python -m calibration.calibrate single-cam --help`.

**Zones** — dashboard Settings: draw the floor zones (Base height 0.304 on a platform) and set `gateway_url` to `http://<PC-IP>:8080`.

## 3. Run

```bash
3d                               # from the repo folder; http://localhost:8000 → START
```
Manual (two terminals): `python -m backbone.runtime --config config/backbone.yaml` and `python -m isistream --config config/backbone.yaml`.
Headless: `./install.sh gpu --systemd`, then `sudo systemctl start isi-backbone isistream`. Never together with the dashboard's START.

## 4. Verify

```bash
python tools/rtsp_smoke.py rtsp://<camera-url>
python tools/latency_probe.py online --config config/backbone.yaml --seconds 60   # p95 < 200 ms
curl -s http://127.0.0.1:8000/api/status | python3 -m json.tool | head -20        # light: green
curl -s http://127.0.0.1:8080/nodes                                                 # node alive, 2 cams
```
Floor check: pallet in → `palette_empty` < 0.5 s; pallet out → `no_palette` < 2 s (`curl http://127.0.0.1:8080/zones`).

## Notes

- Manual env: `conda env create -f environment.yml -n monitor3d`; refresh with `conda env update -f environment.yml -n monitor3d --prune`.
- Static IP for the PC on the camera LAN (reference 192.168.2.113/22); a lease change shows as "gateway unreachable".
- One CUDA process at a time. Engines are per-machine: rebuild on another GPU.
- Gateway code change: `docker compose -p on-prem -f isicomms/deploy/onprem/docker-compose.yml build gateway && … up -d gateway`. Never rebuild Mosquitto.
- Manual edit of `backbone.yaml` → STOP/START.
- Training env (optional): `conda env create -f isi-train.yml -n isi-train`; run from `trainer/isidet/`.
