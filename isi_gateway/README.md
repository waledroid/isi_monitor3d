# isi-gateway

Central cloud aggregator for the distributed ISI Monitor 3D deployment.

## What it is

N Backbone nodes (one per warehouse PC, each identified by a `node_id`) publish
metric track, zone-passing, diagnostics, and config messages to a central MQTT
broker under the topic tree `isi/<node_id>/{track2d/<cls>, track3d/<cls>,
zones/<zone>/passings, images/<zone>/<id>, diagnostics/heartbeat, config}`.

The gateway subscribes to that broker, aggregates per-node state keyed by
`node_id`, and serves a single polling REST API for free-moving AGVs and
supervisory systems.

## Run

```bash
conda activate monitor3d
pip install -e isi_gateway   # one-time, from repo root
python -m isi_gateway        # uvicorn on :8080
```

The gateway requires a reachable MQTT broker. Point it at one via env vars:

```bash
ISI_GATEWAY_MQTT_HOST=192.168.1.10 \
ISI_GATEWAY_MQTT_PORT=1883 \
python -m isi_gateway
```

All settings (see `isi_gateway/config.py`) use the `ISI_GATEWAY_` prefix.

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /healthz | Liveness probe — never touches the broker |
| GET | /nodes | Per-node summary (alive/stale, mode, cameras, latency/fps) |
| GET | /tracks | Flat track list with node_id tag; filters: `?node=&cls=&zone=` |
| GET | /diagnostics | Per-node diagnostics heartbeat |
| GET | /passings | Recent zone-passing events; filters: `?limit=&node=` |
| GET | /zones | Union of all nodes' config zones (global warehouse map) |
| GET | /config | Per-node raw config advertisement |

Optional bearer-token auth: set `ISI_GATEWAY_API_TOKEN` — all routes except
`/healthz` require `Authorization: Bearer <token>` when set.

## Deploy (central server)

A self-hosted Mosquitto + gateway stack:

```bash
cd deploy
docker compose up --build      # broker on :1883, polling API on :8080
```

The gateway image (`isi_gateway/Dockerfile`) builds from the repo root because it
needs the backbone package for `backbone.metadata.schemas` + `backbone.shared.zones`
(base deps only — no CUDA/OpenCV/GStreamer). Each warehouse-PC Backbone points its
mqtt sink at this broker (`host: <server>`, `prefix: isi/<node_id>`); the gateway
auto-discovers nodes from their retained `config` adverts. See
`docs/architecture-distributed.md` for the full topology + topic map.

## Security (default off — enable before internet exposure)

| Control | How |
|---|---|
| Broker auth | `allow_anonymous false` + `password_file`; node `username`/`password` + `ISI_GATEWAY_MQTT_USERNAME`/`_PASSWORD` |
| Broker TLS | TLS listener + certs; node `tls: true` + `ISI_GATEWAY_MQTT_TLS=true` |
| API token | `ISI_GATEWAY_API_TOKEN` → `Authorization: Bearer <token>` on every route but `/healthz` |
