# Dockerized ISI Monitor 3D

`./up.sh` from the repo root brings up the whole stack:

| Service | Image | Net | Port | What |
|---|---|---|---|---|
| `app` | `isi-monitor3d/app` (GPU) | host | 8000 | monitor_web dashboard; START spawns `backbone.runtime` + `isistream` **inside this container** (same supervision, `/dev/shm` bus, NAL sockets and loopback UDP as the native flow) |
| `isical` | same image | host | 8300 | calibration studio (Multical venv baked in) |
| `mosquitto` | `eclipse-mosquitto:2` | bridge | 1883 | MQTT broker (onprem profile, `isicomms/deploy/onprem/mosquitto.conf`) |
| `gateway` | `isi-monitor3d/gateway` | bridge | 8080 | isicomms REST aggregator + probe UI (`/ui`, `/docs`) |

## Design in one paragraph

The GPU image (`docker/Dockerfile.app`) is a faithful clone of the `monitor3d`
conda env (`docker/environment.app.yml` — keep in sync with the repo-root
`environment.yml`) plus the live env's pip deviations (`onnxruntime-gpu==1.23.2`,
`tensorrt-cu12==10.16.1.11`). Source is **baked** at `/opt/isi` (editable installs),
so the image alone is deployable to a site PC. At runtime compose mounts the repo
at **its own host path** and points the apps at it via `MONITOR_WEB_*` / `ISICAL_*`
env vars — every absolute path inside `config/backbone.yaml` (calibration JSON,
models under `trainer/`, zones) resolves identically in and out of Docker, and
Settings-modal saves land on the host as your uid. `network_mode: host` keeps the
config's `127.0.0.1` addresses valid: UDP 9010 (isistream→engine) and 9001
(engine→dashboard) never leave the container, and the backbone's mqtt sink at
`127.0.0.1:1883` reaches mosquitto's published port.

## Requirements

- Docker Engine + compose v2, `nvidia-container-toolkit` with the nvidia
  runtime registered (`docker info | grep nvidia`).
- `NVIDIA_DRIVER_CAPABILITIES=compute,utility,video` is set in the image —
  `video` is what enables NVDEC (`decoder: nvdec`) inside the container.
- Stop any natively-running dashboard first (the `3d` alias also binds :8000).

## Everyday commands

```bash
./up.sh                  # build + start + status + URLs
./up.sh logs -f app      # follow the dashboard (+ spawned engines') logs
./up.sh down             # stop everything
./up.sh build app        # rebuild just the app image after code changes
```

Code changes require an image rebuild (source is baked; the conda layer is
cached, so a rebuild is ~seconds). Config/model/calibration changes need **no**
rebuild — they live on the host mount.

## Site deployment sketch

Ship (or `docker save`/registry-push) `isi-monitor3d/app` + `isi-monitor3d/gateway`,
copy `docker-compose.yml` + `up.sh` + a data checkout (config/, models/,
calibration output), and run `./up.sh` from that directory: the mount-at-own-path
scheme means the site's `backbone.yaml` just uses the site's absolute paths.
TRT engines rebuild once per machine into `models/.trt_cache` (persisted on the
mount). For the internet-facing broker profile (TLS + auth + Caddy), use
`isicomms/deploy/cloud/` as before.

## Known limits

- The `app` container is the one place backbone + isistream + dashboard run
  together (deliberate — preserves the START/STOP supervision and the measured
  in-process couplings; see CLAUDE.md "Direction 1").
- USB/V4L2 cameras would need `devices: [/dev/video0]` added to the `app`
  service (current rig is RTSP-only).
- `metadata.images.out_dir` (`/var/lib/isi_monitor3d/snapshots`) is not mounted;
  add a volume if snapshot publishing is ever enabled in points mode's
  frames-mode rollback.
