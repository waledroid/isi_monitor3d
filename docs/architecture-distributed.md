# Distributed deployment — multi-node Backbone + central gateway

The system scales to a whole warehouse by running **one Backbone per PC**, each
watching a different area with its own dual-camera rig, all feeding **one central
cloud server** that exposes a single polling REST API. AGVs move freely across
areas and pull the global picture from that one API.

```
 Warehouse PC A                Warehouse PC B                Warehouse PC C
 ┌───────────────┐            ┌───────────────┐            ┌───────────────┐
 │ Backbone      │            │ Backbone      │            │ Backbone      │
 │ node_id=zone_a│            │ node_id=dock_1│            │ node_id=cold_3│
 └──────┬────────┘            └──────┬────────┘            └──────┬────────┘
        │ MQTT (outbound only)       │                           │
        └──────────────┬─────────────┴─────────────┬─────────────┘
                       ▼                            ▼
                ┌──────────────────────────────────────┐
                │     central cloud MQTT broker         │   (Mosquitto, TLS-able)
                │     subscribes: isi/#                  │
                └──────────────────┬───────────────────┘
                                   ▼
                ┌──────────────────────────────────────┐
                │   isi-gateway  (FastAPI, :8080)       │
                │   per-node cache keyed by node_id      │
                │   GET /tracks /nodes /zones /passings  │  ◄── AGVs / WMS poll
                │       /diagnostics /config /healthz     │
                └──────────────────────────────────────┘
```

## Identity & topics

Each node has a unique **`node_id`** (e.g. `zone_a`). Topics carry an explicit
**topic version** (`TOPIC_VERSION` in `backbone/comms/schemas.py`, currently `v1`)
between the base and the node id, so its MQTT sink `prefix` is `isi/v1/<node_id>`
and everything it emits is namespaced `<base>/<version>/<node_id>/<suffix>`:

| Topic | Payload | Notes |
|---|---|---|
| `isi/v1/<node_id>/track2d/<cls>` | `Track2DMessage` | per detection class |
| `isi/v1/<node_id>/track3d/<cls>` | `Track3DMessage` | subscribed tracks (Mode 2) |
| `isi/v1/<node_id>/zones/<zone>/passings` | `PassingEventMessage` | zone enter/leave |
| `isi/v1/<node_id>/images/<zone>/<id>` | `ImageRefMessage` | URL only, never bytes |
| `isi/v1/<node_id>/diagnostics/heartbeat` | `DiagnosticsMessage` | every ~5 s — node liveness |
| `isi/v1/<node_id>/config` | `ConfigMessage` | **retained**, once at startup — zones/cameras/mode |

Each Backbone owns its **own `track_id` space**, so global identity is
`(node_id, track_id)`. The gateway subscribes `<base>/#` and parses the version +
`node_id` out of the topic: a segment matching `^v\d+$` after the base is the
topic version (next segment is `node_id`); a legacy unversioned `isi/<node_id>/...`
topic is accepted as `version=v0`. Every aggregated item is tagged with `node_id`,
and each node's `topic_version` is surfaced on `/nodes` (and `/config`).

**Self-describing nodes:** the retained `config` advert means a freshly-started
gateway (or one that reconnects) learns each node's zones/cameras/mode with **zero
central configuration** — a new node simply appears in `/nodes` and `/zones`.

## Per-node config (`config/backbone.yaml` on each PC)

**On-prem (LAN, plaintext):**

```yaml
node_id: zone_a
metadata:
  area: "Zone A — racking"
  sinks:
    - plugin: mqtt
      host: <central-broker-host>
      port: 1883
      prefix: isi/v1/zone_a            # = isi/<TOPIC_VERSION>/<node_id>
  diagnostics: { enabled: true, interval_sec: 5.0, rms_gate_px: 2.0 }
```

**Cloud (TLS + auth — matches the cloud deploy profile):**

```yaml
node_id: zone_a
metadata:
  area: "Zone A — racking"
  sinks:
    - plugin: mqtt
      host: <cloud-server-ip-or-dns>
      port: 8883
      tls: true
      ca_cert: /etc/isi/ca.crt         # distributed from deploy/cloud/certs/ca.crt
      username: <MQTT_USERNAME>
      password: <MQTT_PASSWORD>
      prefix: isi/v1/zone_a            # = isi/<TOPIC_VERSION>/<node_id>
  diagnostics: { enabled: true, interval_sec: 5.0, rms_gate_px: 2.0 }
```

## Central server (`isi-gateway`)

Aggregates all nodes and serves the polling API — see `isi_gateway/README.md` for
the endpoint table and `deploy/README.md` for deployment profiles:

- `deploy/onprem/docker-compose.yml` — LAN / trusted-network stack (plaintext, :1883)
- `deploy/cloud/docker-compose.yml`  — internet-facing stack (TLS :8883 + Caddy :443)

Node liveness comes from the diagnostics-heartbeat freshness
(`ISI_GATEWAY_NODE_STALE_AFTER_S`, default 15 s): a node that stops heart-beating
flips to `stale` in `/nodes`.

## Security checklist (before any internet exposure)

The system is **functional-first** — everything works open on a trusted LAN, and
each control is config/env that defaults OFF:

1. **Broker auth** — `allow_anonymous false` + a `password_file`; set each node's
   mqtt `username`/`password` and the gateway's `ISI_GATEWAY_MQTT_USERNAME`/`_PASSWORD`.
2. **Broker TLS** — a TLS listener (`:8883`) + CA/cert/key; set node `tls: true`
   and `ISI_GATEWAY_MQTT_TLS=true`.
3. **API auth** — set `ISI_GATEWAY_API_TOKEN`; every route except `/healthz` then
   requires `Authorization: Bearer <token>`.
4. Never expose the broker or the API anonymously to the internet.
