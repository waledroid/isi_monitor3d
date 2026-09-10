# ISI Monitor 3D — install from git (GPU, branch `main`)

> Automated: `./install.sh gpu` runs these stages with a progress bar, skips stages already done, and lists what still needs a person (`--dry-run` to preview, `--list` for stage ids, `--skip`/`--only`, `--systemd`). The stages below are the manual equivalent.

Clean Ubuntu 22.04/24.04 or WSL2, NVIDIA GPU, two RTSP cameras.
Ports: dashboard 8000 · gateway 8080 · MQTT 1883 · isical 8300 · loopback UDP 9010/9001.

## Stage 0 — host prerequisites
```bash
nvidia-smi                                   # driver with CUDA 12.x visible
docker compose version                       # Docker Engine + compose v2
sudo apt install -y git cmake libopencv-dev libeigen3-dev   # cmake 3.x (AprilGrid calibration only)
# Miniforge:
curl -L -o Miniforge3.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3.sh -b -p ~/miniforge3 && ~/miniforge3/bin/conda init bash && exec bash
```

## Stage 1 — clone
```bash
cd ~
git clone https://github.com/IsitecVision/isi_monitor3d.git
cd isi_monitor3d
git checkout main
```

## Stage 2 — runtime environment
```bash
conda env create -f environment.yml -n monitor3d
conda activate monitor3d
conda remove -n monitor3d --force onnxruntime
pip install onnxruntime-gpu==1.23.2 tensorrt-cu12==10.16.1.11
rm -f ~/miniforge3/envs/monitor3d/lib/python3.10/site-packages/tensorrt_libs/libnvinfer_builder_resource_win_*
python -c "import onnxruntime as ort; print(ort.get_available_providers())"   # CUDAExecutionProvider present
```

## Stage 3 — calibration backend (Multical, isolated venv)
```bash
bash calibration/setup_multical.sh           # add MULTICAL_VIEWER=0 in front on a headless PC
ls calibration/.venv-multical/bin/multical
```

## Stage 4 — dashboard
```bash
cd monitor_web && pip install -e ".[dev]" && cd ..
echo "alias 3d='conda activate monitor3d && python -m monitor_web'" >> ~/.bashrc && source ~/.bashrc
```

## Stage 5 — models (not in git)
```bash
mkdir -p models
# copy from the reference PC / trainer output:
#   pallet.onnx                 -> models/pallet.onnx
#   yolo11n-pose-dynamic.onnx   -> models/yolo11n-pose-dynamic.onnx
python tools/onnx_inspect.py models/pallet.onnx
python tools/onnx_to_engine.py models/pallet.onnx --imgsz 320 --min-batch 1 --opt-batch 8 --max-batch 32
python tools/onnx_to_engine.py models/yolo11n-pose-dynamic.onnx --imgsz 640 --max-batch 4
ls models/*.engine                            # per-machine artifacts, rebuild on another GPU
```

## Stage 6 — comms stack (gateway + Mosquitto)
```bash
docker compose -p on-prem -f isicomms/deploy/onprem/docker-compose.yml up -d --build
curl http://127.0.0.1:8080/healthz            # {"ok":true}
# port 1883 busy?  sudo systemctl disable --now mosquitto
```

## Stage 7 — config
```bash
cp config/backbone.yaml config/backbone.yaml.site-backup
nano config/backbone.yaml
```
Set: `node_id` (unique per PC) · `cameras` RTSP URLs · `calibration_path` (absolute, from stage 8) ·
`detection.onnx_path` / `pose_onnx_path` (absolute `.engine` paths) · `ingestion.mode: points` ·
`ingestion.points.max_skew_ms: 100` · `homography.pallet_state: {enter_after: 5, presence_conf_min: 0.6}` ·
`metadata.sinks`: udp `127.0.0.1:9001` + mqtt `127.0.0.1:1883` prefix `isiMonitor3D/v1/<node_id>`.
```bash
cp config/danger_zones_object.yaml.example config/danger_zones_object.yaml   # optional proximity rings
```

## Stage 8 — calibration
```bash
python -m isical                              # http://localhost:8300 → Intrinsic → Extrinsic → Export
# boards: print tools/boards_print/*.png at 100 % scale
# one camera only (Mode 1):
python -m calibration.calibrate single-cam --camera-id cam_a --image-size 1920 1080 \
  --pair u1,v1,X1,Y1 --pair u2,v2,X2,Y2 --pair u3,v3,X3,Y3 --pair u4,v4,X4,Y4 --pair u5,v5,X5,Y5 \
  --output config/calibration.json
```

## Stage 9 — run
```bash
3d                                            # http://localhost:8000 → Settings: zones (Base height 0.304 on a platform), gateway_url http://<PC-IP>:8080 → START
```
Manual (two terminals):
```bash
python -m backbone.runtime --config config/backbone.yaml
python -m isistream --config config/backbone.yaml
```

## Stage 10 — verify
```bash
pytest -q
(cd monitor_web && pytest -q); (cd isicomms && pytest -q)
python tools/rtsp_smoke.py rtsp://<camera-url>
python tools/latency_probe.py online --config config/backbone.yaml --seconds 60   # p95 < 200 ms
curl -s http://127.0.0.1:8000/api/status | python3 -m json.tool | head -20        # light: green
curl -s http://127.0.0.1:8080/nodes                                                 # node alive, 2 cams
# floor: pallet in → palette_empty < 0.5 s; pallet out → no_palette < 2 s   (curl http://127.0.0.1:8080/zones)
```

## Stage 11 — headless (optional, instead of the dashboard's START)
```bash
sudo tee /etc/systemd/system/isi-backbone.service >/dev/null <<UNIT
[Unit]
Description=ISI Monitor 3D metric engine
After=network-online.target docker.service
[Service]
User=$USER
WorkingDirectory=$HOME/isi_monitor3d
ExecStart=$HOME/miniforge3/envs/monitor3d/bin/python -m backbone.runtime --config $HOME/isi_monitor3d/config/backbone.yaml
Restart=always
RestartSec=3
[Install]
WantedBy=multi-user.target
UNIT
sudo tee /etc/systemd/system/isistream.service >/dev/null <<UNIT
[Unit]
Description=ISI Monitor 3D perception producer
After=isi-backbone.service
[Service]
User=$USER
WorkingDirectory=$HOME/isi_monitor3d
ExecStart=$HOME/miniforge3/envs/monitor3d/bin/python -m isistream --config $HOME/isi_monitor3d/config/backbone.yaml
Restart=always
RestartSec=3
[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl daemon-reload && sudo systemctl enable --now isi-backbone isistream
```
Never run the units and the dashboard's START at the same time.

## Stage 12 — optional training env
```bash
conda env create -f isi-train.yml -n isi-train
cd trainer/isidet && conda activate isi-train && python scripts/run_train.py --config configs/train_pallet.yaml
```

## Notes
- Static IP for the PC on the camera LAN (reference: 192.168.2.113/22, gw 192.168.1.254, DNS 192.168.1.10). A lease change shows as "gateway unreachable".
- One CUDA process at a time; no benchmarks on the GPU while the system runs.
- Gateway code change → `docker compose -p on-prem -f isicomms/deploy/onprem/docker-compose.yml build gateway && … up -d gateway`. Never rebuild Mosquitto.
- Any manual edit of `backbone.yaml` → STOP/START.
