# ISI Monitor 3D

Zone-based stereo warehouse vision system for Isitec. Each warehouse PC runs a
**Backbone node** that ingests RTSP from 1–2 fixed cameras and publishes metric,
identity-stable metadata — 2D floor tracks always, stereo 3D on demand, and per-zone
contents (objects + confidence + pallet occupancy) — over **MQTT** to a central
broker, where the **isi-gateway** aggregates every node into one REST API for
WMS/FMS/AGV consumers. A local **operator dashboard** provides live video, a 2D
digital-twin floor map, and node configuration.

```
cam_a ─RTSP┐                                        ┌─► WMS/FMS (direct MQTT)
cam_b ─RTSP┤► Backbone node ──publish──► mosquitto ─┤
           │  (per warehouse PC)         :1883      └─► isi-gateway :8080 ──REST──► AGVs, dashboards
           └► monitor_web :8000 (local operator UI + node config)
```

The full communication contract (topic tree `isiMonitor3D/v1/<node_id>/…`, message
schemas, use cases) is specified in `docs/rfc.md`.

## Quick start

```bash
sudo apt install -y git curl
git clone https://github.com/IsitecVision/isi_monitor3d.git && cd isi_monitor3d
./install.sh gpu                 # cpu branch: ./install.sh cpu · --env NAME picks the conda env (default monitor3d)
3d                               # from the repo folder: dashboard on :8000 → START
```

The installer is stage-by-stage and resumable: fix any NEEDS YOU item (models, camera URLs,
calibration, zones) and rerun it. Flags: `--dry-run`, `--list`, `--only STAGE`, `--skip STAGE`,
`--systemd`. Full guides: `isimonitor3d_install.md` (GPU) · `isimonitor3d_cpu_install.md` (CPU).

Development by hand:

```bash
conda env create -f environment.yml -n monitor3d && conda activate monitor3d
pytest                                                     # backbone + calibration suite
python -m backbone.runtime --config config/backbone.yaml   # metric engine
python -m isistream --config config/backbone.yaml          # perception producer
python -m monitor_web                                      # operator UI on :8000
```

## MQTT fabric + gateway (deployment)

The broker and gateway run as two containers on the central server:

```bash
docker compose -p on-prem -f isicomms/deploy/onprem/docker-compose.yml up -d --build
```

- **mosquitto** (`:1883`) — MQTT broker; persistence keeps retained messages
  (node `config`, per-zone state) across restarts.
- **isi-gateway** (`:8080`) — subscribes `isiMonitor3D/#`, caches warehouse state,
  serves `GET /v1/{nodes,tracks,zones,zones/<name>,passings,config,diagnostics}`
  and `/healthz`.

Always deploy with `--build`: the gateway image vendors the `backbone/` schemas,
so it must be rebuilt whenever the schema version bumps (a stale image silently
rejects newer payloads). A secured cloud profile (TLS/auth) lives in `isicomms/deploy/cloud/`.

Per-node MQTT settings (broker host, topic prefix) are edited in the dashboard's
Settings modal or in `config/backbone.yaml` (`metadata.sinks`).

## Calibration

- **isical Studio** (`isical/`, uvicorn `:8300`) walks an operator through the
  2-camera capture → solve → export flow.
- CLI backend: `python -m calibration.calibrate` (`calibrate-2cam` for stereo,
  `single-cam` for the 1-camera floor fit). Multical runs in its own isolated
  venv: `bash calibration/setup_multical.sh` (one-time).

## Training (external to the runtime)

`trainer/isidet/` (YOLO detection, `isi-train` conda env) produces the `.onnx`
the Backbone consumes; `trainer/isiGen/` generates synthetic training data.
The runtime is inference-only.

## Repo layout

```
backbone/         Vision pipeline package (core, shared, ingestion, detection,
                  homography, triangulation, comms, runtime)
monitor_web/      Operator dashboard (FastAPI, :8000)
isicomms/         MQTT broker + gateway unit: MQTT→REST aggregator (:8080), deploy/ profiles
isical/           Calibration studio web app (:8300)
calibration/      Calibration backend (Multical wrappers, boards, single-cam)
trainer/          isidet (YOLO training) + isiGen (synthetic data)
isistream/        Perception producer (capture + detect + pose → detection sets)
config/           YAML configs (backbone.yaml, zones.yaml, zone_patches.yaml, …)
docs/             RFC + architecture docs (docs/rfc.md, mqtt-architecture.md)
tools/            Operational tools (latency probe, detection smoke, …)
tests/            pytest suite
```

## Hardware

- **Dev:** NVIDIA RTX 5070 12 GB, Linux/WSL2.
- **Production (later):** NVIDIA Jetson Orin NX 16 GB — same `.onnx`, same
  calibration, same MQTT contract; deploy/env job only.
