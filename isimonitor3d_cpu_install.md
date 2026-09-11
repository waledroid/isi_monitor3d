# ISI Monitor 3D CPU — install from git (branch `cpu`)


Clean Ubuntu 22.04/24.04 or WSL2, no GPU, one RTSP camera. OpenVINO on CPU.
Ports: dashboard **8200** · gateway 8080 · MQTT 1883 · loopback UDP 9012/9003.
Ignore `CLAUDE.md`, `docker/README.md`, `monitor_web/README.md`, `config/backbone.yaml.example` on this branch (GPU-line leftovers).

## Quick path: `./install.sh cpu`

```bash
sudo apt install -y git curl
git clone https://github.com/IsitecVision/isi_monitor3d.git isi_monitor3d_cpu
cd isi_monitor3d_cpu && git checkout cpu
./install.sh cpu
```

One line per stage with a progress bar, then a summary:

```
[#########---------------------]  3/10 Conda env monitor3d-cpu
   ✔ done
...
Summary
  prereq     DONE       Host prerequisites
  config     NEEDS YOU  Site configuration
  verify     DONE       Test suite
```

- **DONE** — the stage's check passed (just done, or already there).
- **NEEDS YOU** — a person must act (camera URL, calibration, zones); the lines above say what is missing.
- **FAILED** — the other stages still run; exit code 1.

Fix the NEEDS YOU items and run the same command again: done stages are skipped, so it resumes where it stopped. Flags:

```bash
./install.sh cpu --dry-run       # show what each stage would do, change nothing
./install.sh cpu --list          # stage ids
./install.sh cpu --only comms    # run one stage
./install.sh cpu --skip comms    # skip one (e.g. broker already on a GPU node)
./install.sh cpu --systemd       # also install the two service units (sudo)
./install.sh cpu --env mysite    # use (or create) the conda env "mysite" instead of monitor3d-cpu; alias + units follow it
```
The `3d_cpu` alias only runs from inside the repo (`cd` there first); elsewhere it prints a reminder.

Then open a new shell, run `3d_cpu`, press START. The stages below are the manual equivalent.

## Stage 0 — host prerequisites
```bash
docker compose version                       # Docker Engine + compose v2
sudo apt install -y git
curl -L -o Miniforge3.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3.sh -b -p ~/miniforge3 && ~/miniforge3/bin/conda init bash && exec bash
```

## Stage 1 — clone
```bash
cd ~
git clone https://github.com/IsitecVision/isi_monitor3d.git isi_monitor3d_cpu
cd isi_monitor3d_cpu
git checkout cpu
ls models/pallet_seg_openvino models/yolo11n_pose_openvino config/mode1/calibration.json   # shipped in git
```

## Stage 2 — environment
```bash
conda env create -f environment.yml -n monitor3d-cpu
conda activate monitor3d-cpu
pip install --no-deps -e monitor_web -e isicomms
echo "alias 3d_cpu='conda activate monitor3d-cpu && MONITOR_WEB_PORT=8200 python -m monitor_web'" >> ~/.bashrc && source ~/.bashrc
```
Never install onnxruntime, torch, ultralytics or Multical into this env.

## Stage 3 — config
```bash
sed -i "s#/home/aatanda/isi_monitor3d_cpu#$PWD#g" config/backbone.yaml     # absolute paths → this clone
nano config/backbone.yaml
```
Set: `node_id` (unique, keep a `_cpu` suffix if a GPU node shares the broker) · `cameras.cam_a` RTSP URL (`source` + `detect_source`) ·
`ingestion.mode: points`, `listen_port: 9012`, `max_skew_ms: 100` · `metadata.sinks`: udp `127.0.0.1:9003` + mqtt `127.0.0.1:1883` ·
`homography.pallet_state: {enter_after: 5, presence_conf_min: 0.7}` · `detection.plugin: yolo_openvino_seg`, `device: CPU`, `zone_imgsz: 320`.

## Stage 4 — comms stack (gateway + Mosquitto)
```bash
docker compose -p on-prem -f isicomms/deploy/onprem/docker-compose.yml up -d --build
curl http://127.0.0.1:8080/nodes
```
One broker per machine: if a GPU-line install already runs the stack, skip this stage.

## Stage 5 — dashboard: camera, calibration, zones, START
```bash
3d_cpu                                        # http://localhost:8200
```
1. Settings → Cameras: cam_a RTSP URL.
2. Calibrate (ruler): click the 4 pallet corners TL→TR→BR→BL + pallet size (1.200 × 0.800 m) → `config/mode1/calibration.json`.
3. Settings → Floor zones: draw zones; Base height 0.304 on a platform, 0 on the floor; `gateway_url` = `http://<PC-IP>:8080`.
4. START.

CLI calibration with surveyed points (alternative to step 2):
```bash
python -m calibration.calibrate single-cam --camera-id cam_a --image-size 1280 720 \
  --pair u1,v1,X1,Y1 --pair u2,v2,X2,Y2 --pair u3,v3,X3,Y3 --pair u4,v4,X4,Y4 \
  --output config/mode1/calibration.json
```

## Stage 6 — verify
```bash
pytest -q tests calibration/tests
(cd monitor_web && pytest -q); (cd isicomms && pytest -q)
python tools/detection_smoke.py --xml models/pallet_seg_openvino/model.xml --image shot.jpg --input-size 320
python tools/rtsp_smoke.py rtsp://<camera-url>
curl -s http://127.0.0.1:8200/api/status | python3 -m json.tool | head -20   # light: green, cam_a live
curl -s http://127.0.0.1:8080/nodes
# floor: pallet in → palette_empty < 0.5 s; pallet out → no_palette < 2 s   (curl http://127.0.0.1:8080/zones)
```

## Stage 7 — headless (optional, instead of the dashboard's START)
```bash
sudo tee /etc/systemd/system/isi-backbone.service >/dev/null <<UNIT
[Unit]
Description=ISI Monitor 3D CPU metric engine
After=network-online.target docker.service
[Service]
User=$USER
WorkingDirectory=$HOME/isi_monitor3d_cpu
Environment=OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
ExecStart=$HOME/miniforge3/envs/monitor3d-cpu/bin/python -m backbone.runtime --config $HOME/isi_monitor3d_cpu/config/backbone.yaml
Restart=always
RestartSec=3
[Install]
WantedBy=multi-user.target
UNIT
sudo tee /etc/systemd/system/isistream.service >/dev/null <<UNIT
[Unit]
Description=ISI Monitor 3D CPU perception producer
After=isi-backbone.service
[Service]
User=$USER
WorkingDirectory=$HOME/isi_monitor3d_cpu
Environment=OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
ExecStart=$HOME/miniforge3/envs/monitor3d-cpu/bin/python -m isistream --config $HOME/isi_monitor3d_cpu/config/backbone.yaml
Restart=always
RestartSec=3
[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl daemon-reload && sudo systemctl enable --now isi-backbone isistream
```
Manual run (two terminals): `python -m backbone.runtime --config config/backbone.yaml` and `python -m isistream --config config/backbone.yaml`.

## New model
```bash
ovc model.onnx --output_model models/<name>/model.xml     # a path containing "pose" goes to the pose dropdown
```

## Notes
- `MONITOR_WEB_PORT=8200` is required; without it the dashboard starts on 8000.
- Static IP for the PC on the camera LAN; a lease change shows as "gateway unreachable".
- Expect 11–13 pairs/s (24 ms pallet @320, 31 ms pose @480 on the reference CPU). No other heavy inference on the box.
- Any manual edit of `backbone.yaml` → STOP/START.
- Fixes land on `main` first and are cherry-picked onto `cpu`.
