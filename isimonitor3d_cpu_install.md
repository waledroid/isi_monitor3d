# ISI Monitor 3D CPU — install (branch `cpu`)

Ubuntu 22.04/24.04 or WSL2 · no GPU · one RTSP camera · OpenVINO on CPU.
Ports: dashboard **8200** · gateway 8080 · MQTT 1883 · loopback UDP 9012/9003.
Models and `config/mode1/calibration.json` ship in git. Ignore `CLAUDE.md`, `docker/README.md`, `monitor_web/README.md` on this branch.

## 1. Install

```bash
sudo apt install -y git curl
git clone https://github.com/IsitecVision/isi_monitor3d.git isi_monitor3d_cpu
cd isi_monitor3d_cpu && git checkout cpu
./install.sh cpu                 # add --env NAME to use/create another conda env (default monitor3d-cpu)
```

Each stage prints DONE, NEEDS YOU (a manual step, listed above the summary) or FAILED.
Fix the NEEDS YOU items and rerun the same command; done stages are skipped.

```bash
./install.sh cpu --dry-run       # preview
./install.sh cpu --list          # stage ids
./install.sh cpu --only comms    # one stage
./install.sh cpu --skip comms    # skip one (broker already on a GPU node: one broker per machine)
./install.sh cpu --systemd       # also write + enable the two service units (sudo)
```

## 2. NEEDS YOU steps

**Config** — `nano config/backbone.yaml`: `node_id` (unique; `_cpu` suffix if a GPU node shares the broker) ·
`cameras.cam_a` RTSP URL (`source` + `detect_source`) · `ingestion.mode: points`, `listen_port: 9012` ·
`metadata.sinks` udp `127.0.0.1:9003` + mqtt `127.0.0.1:1883` · `detection.plugin: yolo_openvino_seg`, `device: CPU`, `zone_imgsz: 320`.

**Dashboard** (`3d_cpu` → http://localhost:8200):
1. Settings → Cameras: cam_a RTSP URL.
2. Calibrate (ruler): click the 4 pallet corners TL→TR→BR→BL + pallet size (1.200 × 0.800 m).
3. Settings → Floor zones: draw zones (Base height 0.304 on a platform, 0 on the floor); `gateway_url` = `http://<PC-IP>:8080`.
4. START.

CLI calibration with surveyed points instead of step 2: `python -m calibration.calibrate single-cam --help` (output `config/mode1/calibration.json`).

## 3. Run

```bash
3d_cpu                           # from the repo folder (sets MONITOR_WEB_PORT=8200)
```
Manual (two terminals): `python -m backbone.runtime --config config/backbone.yaml` and `python -m isistream --config config/backbone.yaml`.
Headless: `./install.sh cpu --systemd`, then `sudo systemctl start isi-backbone isistream`. Never together with the dashboard's START.

## 4. Verify

```bash
python tools/detection_smoke.py --xml models/pallet_seg_openvino/model.xml --image shot.jpg --input-size 320
python tools/rtsp_smoke.py rtsp://<camera-url>
curl -s http://127.0.0.1:8200/api/status | python3 -m json.tool | head -20   # light: green, cam_a live
curl -s http://127.0.0.1:8080/nodes
```
Floor check: pallet in → `palette_empty` < 0.5 s; pallet out → `no_palette` < 2 s (`curl http://127.0.0.1:8080/zones`).

## Notes

- Manual env: `conda env create -f environment.yml -n monitor3d-cpu` then `pip install --no-deps -e monitor_web -e isicomms`. Never install onnxruntime, torch, ultralytics or Multical into it.
- New model: `ovc model.onnx --output_model models/<name>/model.xml` (a path containing "pose" goes to the pose dropdown).
- Static IP for the PC on the camera LAN; a lease change shows as "gateway unreachable".
- Expect 11–13 pairs/s on the reference CPU. No other heavy inference on the box.
- Manual edit of `backbone.yaml` → STOP/START.
- Fixes land on `main` first and are cherry-picked onto `cpu`.
