# isi-gateway

Central cloud aggregator for the distributed ISI Monitor 3D deployment.

## What it is

N Backbone nodes (one per warehouse PC, each identified by a `node_id`) publish
metric track, zone-state, zone-passing, diagnostics, and config messages to a
central MQTT broker under the **version-namespaced** topic tree
`isiMonitor3D/v1/<node_id>/{track2d/<cls>, track3d/<cls>, zone/<zone>,
zone/<zone>/passings, zone/<zone>/images/<id>, diagnostics/heartbeat, config}`.
The `v1` segment is `TOPIC_VERSION` (`backbone/comms/schemas.py`) — the
topic-contract version.

The gateway subscribes `isiMonitor3D/#` and parses both the versioned layout
(`isiMonitor3D/v1/<node_id>/...`) and legacy unversioned topics (`isiMonitor3D/<node_id>/...`,
reported as `topic_version=v0`), aggregates per-node state keyed by `node_id`,
and serves a single polling REST API for free-moving AGVs and supervisory
systems.

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

All resource routes are mounted under the **`/v1`** prefix (`API_VERSION` in
`isi_gateway/config.py`) and **also** at the bare path as back-compat aliases, so
`/v1/nodes` and `/nodes` serve the same handler. `/healthz` is available both
un-prefixed and under `/v1`.

| Method | Path (versioned) | Bare alias | Description |
|--------|------|------|-------------|
| GET | /healthz | /healthz | Liveness probe — never touches the broker |
| GET | /v1/nodes | /nodes | Per-node summary (alive/stale, **topic_version**, mode, cameras, latency/fps) |
| GET | /v1/tracks | /tracks | Flat track list with node_id tag; filters: `?node=&cls=&zone=` |
| GET | /v1/diagnostics | /diagnostics | Per-node diagnostics heartbeat |
| GET | /v1/passings | /passings | Recent zone-passing events; filters: `?limit=&node=` |
| GET | /v1/zones | /zones | Union of all nodes' config zones (global warehouse map), each enriched with the zone's live contents (`objects` + confidence, `count`, `state_ts`) |
| GET | /v1/zones/{name} | /zones/{name} | One zone across all nodes defining it: spec + latest per-node `zone_state` |
| GET | /v1/config | /config | Per-node raw config advertisement (incl. `topic_version`) |

Adding a future `/v2` is a one-line extra include per router in `app.py`.

Optional bearer-token auth: set `ISI_GATEWAY_API_TOKEN` — all routes except
`/healthz` require `Authorization: Bearer <token>` when set.

## Deploy (central server)

Two deployment profiles are provided — see `deploy/README.md` for full instructions.

**On-prem (LAN, plaintext):**

```bash
docker compose -f deploy/onprem/docker-compose.yml up -d --build
# broker on :1883, polling API on http://<host>:8080
```

**Cloud (internet-facing, TLS + auth + API token):**

```bash
cd deploy/cloud
CERT_HOST=<server-ip-or-dns> ./gen-certs.sh     # mint CA + certs
# create broker credentials, fill .env (see deploy/README.md)
docker compose -f deploy/cloud/docker-compose.yml up -d --build
# MQTTS on :8883, HTTPS API on :443
```

The gateway image (`isi_gateway/Dockerfile`) builds from the repo root because it
needs the backbone package for `backbone.comms.schemas` + `backbone.shared.zones`
(base deps only — no CUDA/OpenCV/GStreamer). Each warehouse-PC Backbone points its
mqtt sink at this broker (`host: <server>`, `prefix: isiMonitor3D/v1/<node_id>`); the gateway
auto-discovers nodes from their retained `config` adverts. See
`docs/architecture-distributed.md` for the full topology + topic map.

## Security (default off — enable before internet exposure)

| Control | How |
|---|---|
| Broker auth | `allow_anonymous false` + `password_file`; node `username`/`password` + `ISI_GATEWAY_MQTT_USERNAME`/`_PASSWORD` |
| Broker TLS | TLS listener + certs; node `tls: true` + `ISI_GATEWAY_MQTT_TLS=true` |
| API token | `ISI_GATEWAY_API_TOKEN` → `Authorization: Bearer <token>` on every route but `/healthz` |
