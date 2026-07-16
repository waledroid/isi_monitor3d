# MQTT & Distributed Architecture

How the ISI Monitor 3D system scales from one camera rig to a whole warehouse:
many **Backbone** nodes (one per PC) publish metric metadata to a central **MQTT
broker**, an **isicomms** aggregates it, and AGVs/WMS poll a single REST API.

This document is the **field-level technical reference** — the comms module
internals, the full message-field contract, the broker, the gateway, the identity
and security models, and a start-to-finish runbook. For the design narrative and
the rationale behind each decision, see the companion `docs/rfc.md`; for the bare
topology, `docs/architecture-distributed.md`.

```
 cam_a  cam_b              cam_a  cam_b              cam_a  (Mode 1)
   └──RTSP──┐                └──RTSP──┐                └──RTSP──┐
            ▼                         ▼                         ▼
 Warehouse PC 1            Warehouse PC 2            Warehouse PC 3
 Backbone node_id=zone_a   Backbone node_id=dock_1   Backbone node_id=cold_3
        │ publish isiMonitor3D/v1/zone_a/… │ isiMonitor3D/v1/dock_1/…         │ isiMonitor3D/v1/cold_3/…
        └──────────┬──────────────┴─────────────┬───────────┘   MQTT over the LAN
                   ▼                             ▼
        ┌────────────────────────────────────────────────┐
        │  CENTRAL SERVER (e.g. 192.168.2.39)             │
        │   • Mosquitto broker      :1883  (TLS :8883)     │  routes isiMonitor3D/#
        │   • isicomms REST API  :8080  (HTTPS :443)    │  caches per node_id
        └────────────────────────────────────────────────┘
                   ▲  GET /v1/nodes /v1/tracks /v1/zones /v1/passings …
            AGVs / WMS poll here (HTTP only)
```

Cameras (1–2 per PC) attach **directly to their PC over RTSP** (PoE); video
never crosses the MQTT fabric — only extracted metadata does.

The broker and gateway live together on **one central host that is deliberately
*not* a warehouse PC** — a dedicated server, a small NUC, or a cloud VM, running
nothing but Mosquitto and isicomms. A warehouse PC runs **only a Backbone**; it
never hosts the hub. The only address that has to be reachable on the LAN is the
central host's IP (`192.168.2.39` above) — every PC publishes to it and every AGV
polls it. (You *may* co-locate everything on one box for a pilot, but a camera
PC's Docker is not where the broker belongs in production.)

---

## 1. Design principles

The MQTT layer obeys the Backbone's non-negotiables. These five are stated once
here; the rest of the document refers back to them rather than restating them:

1. **The schema is the only contract.** A node shares **zero code** with a
   consumer. Everything crosses the boundary as validated JSON
   (`backbone/comms/schemas.py`). To add a capability you expand the schema, not
   share a module.
2. **One identity space, per node.** Each Backbone owns its own `track_id`
   counter, so two PCs can both number a track `42`. Global identity is therefore
   the pair **`(node_id, track_id)`** — the gateway *tags* every track with the
   `node_id` it read from the topic, and never merges tracks across nodes.
3. **Outbound-only nodes.** A node only *publishes*. It never subscribes, never
   listens, never knows another node exists. Firewall-friendly and decoupled.
4. **Fail honestly.** Every sink call and every inbound parse is wrapped in
   `try/except` + log; a dead broker, a UDP hiccup, or a malformed packet is
   counted and degrades silently instead of crashing the pipeline or the gateway.
5. **Latency against the capture clock.** Every message's `ts` is the frame's
   `capture_ts` (Unix seconds), propagated unchanged through the whole pipeline —
   never `time.time()` at publish. A consumer measuring `now - ts` gets true
   end-to-end latency.

---

## 2. The message contract — `backbone/comms/schemas.py`

All messages are **frozen, `extra="forbid"` Pydantic models** carrying a
`schema_version` (currently **4**) and a discriminating `type`. `parse_envelope()`
reads `schema_version` first (accepts `{3, 4}` for rolling upgrades), then
dispatches on `type`.

The **on-wire** types are deliberately separate from the **in-process** types
(`backbone.core.types.Track2D/Track3D`) so internal refactors can't break the bus.

### The seven message types

| `type` | Model | Cadence |
|---|---|---|
| `track_2d` | `Track2DMessage` | every frame, every track — **always** |
| `track_3d` | `Track3DMessage` | **Mode 2** only, subscription-driven |
| `zone_state` | `ZoneStateMessage` | on zone-contents change + ~1 s refresh, **retained, QoS 1** |
| `proximity` | `ProximityMessage` (v5) | person↔object floor distances within `metadata.proximity.max_distance_m` (default = `detection.person_pallet_max_distance_m`, 6 m), throttled to `refresh_interval_s` (0.5 s), **retained, QoS 1** on `{prefix}/proximity`; an explicit empty `pairs` message clears the topic |
| `passing` | `PassingEventMessage` | on a zone enter/leave |
| `image_ref` | `ImageRefMessage` | on a passing, if snapshots enabled |
| `diagnostics` | `DiagnosticsMessage` | every `interval_sec` (~5 s) |
| `config` | `ConfigMessage` | once at startup, **retained** |

### Field reference

**`Track2DMessage`** — a person/object on the floor plane:
```
ts: float                       # capture_ts, Unix seconds
track_id: int                   # node-local identity
cls: str                        # "person", "forklift", "pallet", …
xy_m: (float, float)            # metric position on the floor
vxy_m: (float, float)           # metric velocity
confidence: float               # 0..1
cameras_seeing: (str, …)        # which cameras contributed
occupancy_state: str|None       # "empty"|"full"  (pallet KPI; optional)
occupancy_content: str|None     # "carton"|"polybag"
occupancy_confidence: float
```

**`Track3DMessage`** — the same track lifted to 3D (Mode 2, triangulation):
```
ts, track_id, cls
xyz_m: (float, float, float)
vxyz_m: (float, float, float)
contributing_cameras: (str, …)
max_reprojection_error_px: float   # the geometric-quality gate value
keypoints_xyz: [(x,y,z), …]|None   # pose (S5.5)
single_view: bool                  # Z pinned to floor from one camera (occlusion fallback)
confidence: float
```
The `track_id` is **identical** to the corresponding `Track2DMessage` (principle 2):
the triangulation layer augments an existing track with 3D and **never re-IDs**.

**`ZoneStateMessage`** — one zone's current contents (**the FMS/WMS signal**;
retained on `{prefix}/zone/{zone}`, QoS 1):
```
ts: float                       # capture_ts of the frame producing this state
zone: str                       # zone name (matches ZoneSpec.name in config)
objects: [ZoneObject, …]        # EMPTY list ⇒ zone is empty (explicit, not absent)
count: int                      # len(objects)
```
Each `ZoneObject`:
```
track_id: int
cls: str                        # detected class
confidence: float               # 0..1 — the detection confidence score
xy_m: (float, float)            # floor position inside the zone
occupancy_state: str|None       # pallet empty/full (mirrors Track2DMessage)
occupancy_content: str|None
occupancy_confidence: float
```
Published by the node's `ZoneStateTracker` (`backbone/shared/zone_state.py`)
**on change** (membership, class, or occupancy of any member) plus a periodic
refresh (`metadata.zone_state.refresh_interval_s`, default 1 s) while occupied.
At startup the node publishes a retained **empty** state for every configured
zone, so the whole `zone/` folder is discoverable before anything moves.
`node_id` is not in the payload — the gateway derives it from the topic.

**`PassingEventMessage`** — a boundary crossing:
```
ts, track_id, cls
zone: str
direction: "enter" | "leave"
```

**`ImageRefMessage`** — a snapshot pointer (mirrors the passing so consumers
correlate by `(track_id, zone, ts)`):
```
ts, track_id, cls, zone
url: str        # file:// or http(s):// to the JPEG — NEVER raw bytes
```
Image **bytes never cross the bus.** The node writes the JPEG locally
(`backbone/shared/snapshot_writer.py`) and publishes only a URL — keeps MQTT
payloads tiny and lets a separate file/HTTP server own the media.

**`DiagnosticsMessage`** — the node pulse:
```
ts: float                       # wall-clock at emit
node_id, mode
sources: {camera_id: "alive"|"exited"|"crashed"}
frame_count: int
fps: float
latency_ms: {p50, p95, p99, n}  # LatencyStats
zones: int                      # zone count
subscriptions: int              # triangulation subscription count
calibration: {loaded, rms_ok, mode}   # CalibrationFactCheck
```

**`ConfigMessage`** — the retained self-description:
```
ts, node_id, area, mode
cameras: [str, …]
zones: [{name, kind, type, severity, polygon:[[x,y],…]}, …]   # ZoneSpec
calibration: {loaded, rms_ok, mode}
```

### Versioning — two independent axes

The comms layer versions on **two orthogonal axes**, so a message's *content* and
its *addressing* evolve independently:

- **Payload — `schema_version`** (a single integer in every envelope; consumers
  MUST read it before parsing). Governs the message's *content*. **Adding optional,
  defaulted fields is non-breaking** (a v3 consumer ignores them); renaming or
  removing a field is breaking and bumps the version. `parse_envelope` currently
  accepts `{3, 4}`, so a v4 node and a v3 consumer interoperate during a rollout;
  anything outside that set raises `SchemaVersionError`.
- **Path / topic — `TOPIC_VERSION`** (`"v1"` in `backbone/comms/schemas.py`).
  Governs the *addressing*: the REST prefix (`/v1/...`) and the MQTT topic tree
  (`isiMonitor3D/v1/...`). Lets `v1` and a future `v2` run side-by-side so consumers migrate
  on their own schedule.

**How a `v2` lands:** nodes set `prefix: isiMonitor3D/v2/<node_id>` and publish
`isiMonitor3D/v2/...`; the gateway already parses any `v\d+` segment, so it serves both
trees at once and mounts `/v2/...` alongside `/v1/...`; consumers migrate when
ready; `v1` is deprecated on a published date.

**`v0` legacy fallback:** the gateway still accepts un-versioned
`isiMonitor3D/<node_id>/...` topics (no `v\d+` segment) during transition and reports those
nodes as `topic_version: "v0"`.

---

## 3. The comms module — `backbone/comms/`

The pipeline produces facts; comms is the **mouth** that puts them on the wire.

| File | Role |
|---|---|
| `schemas.py` | the message contract (above) |
| `udp_sink.py` | `UdpSink` — fire-and-forget UDP/JSON on localhost (the operator dashboard + latency probe consume this) |
| `mqtt_sink.py` | `MqttSink` — the network path; the sink that makes the system distributed |
| `publisher.py` | `Publisher` — fans one `publish_*` call out to every configured sink |
| `diagnostics_publisher.py` | `DiagnosticsPublisher` — the heartbeat thread |

### The `MetadataSink` seam

`MetadataSink` is **one of the Backbone's five plugin ABCs** (pinned by
`tests/test_registry.py::test_five_seams_present`). Implementations register
themselves and are built **only** by the orchestrator via the registry. A node
can run **several sinks at once** — typically UDP (local dashboard) **and** MQTT
(the network):

```yaml
metadata:
  sinks:
    - plugin: udp        # local
      host: 127.0.0.1
      port: 9001
    - plugin: mqtt       # network → central broker
      host: 192.168.2.39
      port: 1883
      prefix: isiMonitor3D/v1/zone_a
```

A non-abstract default no-op base keeps the seam count at five even though sinks
gained `publish_event` / `publish_image_ref` / `publish_zone_state` /
`publish_diagnostics` / `publish_config` over time.

### `MqttSink` behaviour (the details that matter operationally)

Everything a node emits is namespaced under `prefix = isiMonitor3D/v1/<node_id>`
(`= {base}/{TOPIC_VERSION}/{node_id}`). The sink substitutes one topic template per
message; each branch carries one kind of information — in plain language:

- **`{prefix}/track2d/{cls}`** — the **always-on** output: one metric 2D
  position-and-velocity on the warehouse floor for each tracked object of class
  `{cls}`. This is the signal an AGV navigates against. `track2d/person` carries
  people positions, `track2d/forklift` forklifts, and so on.
- **`{prefix}/track3d/{cls}`** — the same object lifted to full 3D `(X, Y, Z)`.
  Published **only in Mode 2** (two calibrated cameras) and **only for subscribed
  tracks** — when something downstream actually needs height or pose, not for
  everything.
- **`{prefix}/zone/{zone}`** *(retain=True, QoS 1)* — the **`zone` folder**: one
  subtopic per configured zone carrying its current object list + confidence
  (`ZoneStateMessage` above) — the FMS/WMS integration signal. A node defines
  **up to 6 zones** (`zone1`–`zone6`; the dashboard's `MAX_ZONES = 6` config
  limit — the sink/broker/gateway are zone-count-agnostic), and every one is
  monitored independently. Subscribe `…/zone/+` for every zone on a node,
  `isiMonitor3D/v1/+/zone/+` warehouse-wide.
- **`{prefix}/zone/{zone}/passings`** — an **event** each time a track crosses
  that zone's boundary: one `enter` or `leave` per crossing, not a continuous
  stream.
- **`{prefix}/zone/{zone}/images/{track_id}`** — a **URL pointer** to a saved
  snapshot JPEG for that event; the bytes themselves never touch the bus (see
  `ImageRefMessage` above).
- **`{prefix}/diagnostics/heartbeat`** — the node's health pulse (~5 s): frame
  rate, latency, and per-camera liveness. Absence of the pulse is how the gateway
  notices a PC went down.
- **`{prefix}/config`** *(retain=True)* — the node's self-description (area, mode,
  cameras, zones). **"Retained"** means the broker stores the last message on the
  topic and hands it to any subscriber the instant it connects — so a freshly
  started gateway learns the layout immediately (see §5).

```
{prefix}/track2d/{cls}              {prefix}/diagnostics/heartbeat
{prefix}/track3d/{cls}              {prefix}/config            (retain=True)
{prefix}/zone/{zone}                (retain=True, QoS 1 — current contents)
{prefix}/zone/{zone}/passings
{prefix}/zone/{zone}/images/{track_id}
```

The specifics that make these templates safe and cheap:

- **`track_id` is intentionally NOT in the topic** — it's in the payload. This
  keeps topic cardinality bounded to O(classes): a wildcard `…/track2d/person`
  follows a whole class, not one ephemeral id.
- **Topic sanitisation.** `cls` and `zone` are run through a translation that
  replaces `/ + #` with `_`, so a zone named "dock/door" can't break the topic
  hierarchy or inject wildcards.
- **QoS 0 by default** (set per sink). Tracks/passings/diagnostics are high-rate
  and disposable — at-most-once is the right trade, since the next message
  supersedes the last. `config` and `zone_state` are the exceptions: see below.
- **`config` is always published with `retain=True`**, regardless of the sink's
  `retain` setting. The broker keeps the last `config` per topic forever.
- **`zone_state` is always published with `retain=True` at `zone_state_qos`
  (default 1)** — low-rate absolute state that a WMS must not miss; duplicates
  are harmless, and a late subscriber immediately reads every zone's contents.
- **Broker-down-safe.** Construction uses `connect_async()` + `loop_start()`, so
  building the sink **never blocks or raises** even with no broker — paho retries
  in a background thread. A node boots whether or not the server is up yet.
- **Retained-config re-publish on connect.** The startup advert races the async
  CONNACK (a QoS-0 publish before connect is silently dropped). The sink caches
  the advert and re-publishes it from `_on_connect`, so it survives the race
  **and** re-announces the node after a broker restart.
- **Clean shutdown** is `disconnect()` then `loop_stop()` (so the disconnect is
  sent before the loop is torn down). Idempotent.
- **TLS / auth** are plain sink kwargs: `tls`, `ca_cert`, `tls_insecure`,
  `username`, `password` — all default-off (see §7).

### `DiagnosticsPublisher`

A daemon thread that every `interval_sec` computes fps from the frame-count delta
and emits a `DiagnosticsMessage`. This is the node's **liveness pulse**; the
gateway uses its freshness for alive/stale.

---

## 4. The conductor — `backbone/runtime/orchestrator.py`

The orchestrator is the **only** place that builds sinks (from `metadata.sinks`)
and the only place that calls `publish_*`. It detects the operational mode from
the camera count, wires identity, and drives the lifecycle:

```
run() startup ─► publisher.publish_config(build_config_message())   # retained advert
              ─► diagnostics.start()
per frame     ─► publisher.publish_track_2d(track)                   # always
   zone xing  ─► publisher.publish_event(passing)
   snapshot   ─► publisher.publish_image_ref(url)
   Mode 2     ─► publisher.publish_track_3d(track)                   # subscribed tracks
shutdown      ─► diagnostics.stop()  then  publisher.close()
```

- **`node_id`** comes from `backbone.yaml` (`cfg["node_id"]`, default `"node"`);
  **`area`** from `metadata.area`. Both go into the diagnostics + config messages.
- The producers feeding these — zone enter/leave detection
  (`backbone/shared/zone_transitions.py`) and snapshot JPEG I/O
  (`backbone/shared/snapshot_writer.py`) — live in `shared/`, **not** comms.
  Comms is strictly communication.

### Mode 1 vs Mode 2 (what changes on the wire)

| | Mode 1 (`single_cam_homography`) | Mode 2 (`dual_cam_homography_triangulation`) |
|---|---|---|
| Cameras | 1 | 2 |
| `track_2d` | ✅ always | ✅ always |
| `track_3d` | ✅ never | ✅ for subscribed tracks |
| `passing` / `image_ref` / `diagnostics` / `config` | ✅ | ✅ |

`track_3d` is **subscription-driven**: it's published only for tracks matching
rules in `config/subscriptions.yaml`. The default warehouse output is the cheap
`track_2d`; 3D is on-demand.

---

## 5. Mosquitto — the broker

Mosquitto is the **post office**: pure pub/sub routing. Nodes publish; the gateway
subscribes; no node talks to another node or to an AGV directly.

On-prem config (`deploy/onprem/mosquitto.conf`):
```
listener 1883
persistence true                    # ← keeps RETAINED config adverts across restarts
persistence_location /mosquitto/data/
allow_anonymous true                # fine on a trusted LAN
```

**The self-describing trick (why `persistence` + `retain` matter):** because each
node publishes its `config` *retained* (§3), the broker holds the latest one per
topic; `persistence true` keeps those retained adverts across a broker restart.
So whenever the gateway subscribes to `isiMonitor3D/#` — first boot, reconnect, or a
totally fresh gateway — the broker **immediately replays every node's config**,
and the gateway learns the entire warehouse layout (nodes, areas, modes, cameras,
zones) with **zero central configuration**. Adding a PC needs no registry and no
node list to maintain — it simply *appears*.

---

## 6. The gateway — `isicomms/`

A small FastAPI service that turns the MQTT firehose into a poll-able API.

### Subscriber + cache — `mqtt_subscriber.py`
- Subscribes `{base}/#` (`base = isiMonitor3D`, matches both `isiMonitor3D/v1/...` and legacy
  `isi/...`), parses the topic version-aware: a segment after the base matching
  `^v\d+$` is the `topic_version` and the next segment is `node_id`; a legacy
  un-versioned topic is accepted as `topic_version = "v0"`. Each message is
  `parse_envelope`d and folded into a **thread-safe per-node `NodeState`**: latest
  tracks (by id), a passings ring buffer (`passings_buffer`, default 200), last
  diagnostics, the config advert, `topic_version`, and `last_seen`.
- `connect_async()` + `loop_start()` (broker-down-safe, same as the sink).
- `snapshot_nodes()` copies the inner containers **under the lock** so a reader
  never sees a half-updated node.
- Counts malformed / wrong-version messages instead of crashing.

### Liveness
A node whose last heartbeat is older than `node_stale_after_s` (default **15 s**)
flips `alive → stale` in `/v1/nodes`. (Observed live: a stopped node goes stale.)

### REST endpoints (`api/routes_*.py`)

All resource routes mount under the **`/v1`** prefix (`API_VERSION` in
`isicomms/config.py`) **and** at the bare path as back-compat aliases, so
`/v1/nodes` and `/nodes` hit the same handler. `/healthz` stays unversioned (and
also answers under `/v1`). Adding a future `/v2` is one extra include per router.

| Endpoint | Returns |
|---|---|
| `GET /healthz` | liveness probe — never touches the broker (unversioned) |
| `GET /v1/nodes` | per-node summary: status, area, **topic_version**, mode, cameras, fps, latency |
| `GET /v1/tracks` | all tracks, each tagged `node_id`; filters `?node=&cls=&zone=` |
| `GET /v1/zones` | union of all nodes' zones — the global warehouse map |
| `GET /v1/passings` | recent crossings; `?limit=&node=` |
| `GET /v1/diagnostics` | per-node heartbeats |
| `GET /v1/config` | each node's raw self-description (incl. `topic_version`) |

`GET /v1/tracks?zone=<name>` does a **point-in-polygon** test (`backbone.shared.zones
.Zone.contains`, pure-numpy so the gateway image stays lean) to return only
tracks currently inside that zone — across all nodes.

### Configuration (`config.py`, env prefix `ISI_GATEWAY_`)

| Env var | Default | Meaning |
|---|---|---|
| `ISI_GATEWAY_HOST` / `_PORT` | `0.0.0.0` / `8080` | API bind |
| `ISI_GATEWAY_MQTT_HOST` / `_PORT` | `127.0.0.1` / `1883` | broker |
| `ISI_GATEWAY_MQTT_BASE` | `isiMonitor3D` | topic root subscribed (`isiMonitor3D/#`) |
| `ISI_GATEWAY_MQTT_TLS` | `false` | TLS to the broker |
| `ISI_GATEWAY_MQTT_CA_CERT` | — | CA to verify a self-signed broker |
| `ISI_GATEWAY_MQTT_TLS_INSECURE` | `false` | skip hostname check (test only) |
| `ISI_GATEWAY_MQTT_USERNAME` / `_PASSWORD` | — | broker auth |
| `ISI_GATEWAY_NODE_STALE_AFTER_S` | `15.0` | alive→stale threshold |
| `ISI_GATEWAY_PASSINGS_BUFFER` | `200` | per-node passings ring size |
| `ISI_GATEWAY_API_TOKEN` | — | if set, Bearer auth on every route but `/healthz` |

The gateway image is **lean by design** (`python:3.10-slim`, base deps only — no
CUDA/OpenCV/GStreamer). It imports only `backbone.comms.schemas` +
`backbone.shared.zones`; an import-discipline test (`tests/test_import_discipline.py`)
pins that it never pulls `backbone.runtime`, `cv2`, or `calibration`.

---

## 7. Security model

Functional-first: everything works open on a trusted LAN, and every control is
config/env that **defaults OFF**. Two profiles under `deploy/`:

| | `deploy/onprem/` (LAN) | `deploy/cloud/` (internet-facing) |
|---|---|---|
| Broker | plaintext `:1883`, anonymous | TLS `:8883`, `allow_anonymous false` + password file |
| Gateway API | `:8080`, open | behind Caddy `:443` HTTPS, Bearer token |
| Certs | none | self-signed via `gen-certs.sh` (one CA → broker + API certs) |

Before any internet exposure (the cloud checklist):
1. **Broker auth** — `allow_anonymous false` + `password_file`; set each node's
   mqtt `username`/`password` and the gateway's `ISI_GATEWAY_MQTT_USERNAME/_PASSWORD`.
2. **Broker TLS** — TLS listener + CA/cert/key; node `tls: true` + `ca_cert:`,
   gateway `ISI_GATEWAY_MQTT_TLS=true`.
3. **API token** — `ISI_GATEWAY_API_TOKEN` → `Authorization: Bearer <token>` on
   every route but `/healthz`.
4. Never expose the broker or API anonymously to the internet.

The **node code, schema, topics, and gateway are identical** between profiles —
only transport security changes.

---

## 8. Running the system, start to finish

### Step 1 — central server (once, on the warehouse server/NUC)
```bash
docker compose -p on-prem -f deploy/onprem/docker-compose.yml up -d
docker compose -p on-prem -f deploy/onprem/docker-compose.yml ps     # both Up
curl http://<server-ip>:8080/healthz                                 # {"ok":true}
```
→ Mosquitto on `<server-ip>:1883`, isicomms on `<server-ip>:8080`, both
`restart: unless-stopped`.

### Step 2 — each warehouse PC (its own Backbone)
Edit that PC's `config/backbone.yaml`:
```yaml
node_id: zone_a                       # UNIQUE per PC
calibration_path: /…/calibration.json
cameras:
  cam_a: { source: { name: rtsp, url: rtsp://…/cam_a } }
  cam_b: { source: { name: rtsp, url: rtsp://…/cam_b } }   # omit ⇒ Mode 1
detection: { plugin: yolo_onnx_seg, onnx_path: /…/best.onnx }
metadata:
  area: "Zone A — racking"
  diagnostics: { enabled: true, interval_sec: 5.0 }
  sinks:
    - plugin: udp                     # optional local dashboard
      host: 127.0.0.1
      port: 9001
    - plugin: mqtt                    # → central server
      host: <server-ip>
      port: 1883
      prefix: isiMonitor3D/v1/zone_a           # = isiMonitor3D/<TOPIC_VERSION>/<node_id>
```
Run it (under systemd in production):
```bash
conda activate monitor3d
python -m backbone.runtime --config config/backbone.yaml
```
On start it publishes its retained `config`, begins heart-beating, and streams
`track2d` (+ `track3d`/`passings`/`image_ref`) once its cameras yield detections.

For a **cloud** node, point the mqtt sink at `port: 8883`, add `tls: true`,
`ca_cert: /etc/isi/ca.crt`, `username`, `password`.

### Step 3 — AGVs / WMS consume the global picture
```bash
curl http://<server-ip>:8080/v1/nodes                    # who's online, where
curl http://<server-ip>:8080/v1/tracks                   # every track, all PCs
curl "http://<server-ip>:8080/v1/tracks?node=zone_a&cls=person"
curl "http://<server-ip>:8080/v1/tracks?zone=dock_door"  # tracks inside a zone
curl http://<server-ip>:8080/v1/zones                    # global warehouse map
```
(The bare paths `/nodes`, `/tracks`, … remain as back-compat aliases.)
An AGV polls `/v1/tracks` (or `/v1/tracks?zone=…`) on its own loop and steers — no MQTT
client, no per-node awareness, one HTTP endpoint.

### Verify the whole chain
```bash
# watch every node's traffic on the broker:
mosquitto_sub -h <server-ip> -t 'isiMonitor3D/#' -v
#  → isiMonitor3D/v1/zone_a/config (retained) · isiMonitor3D/v1/zone_a/diagnostics/heartbeat · isiMonitor3D/v1/zone_a/track2d/person …
# confirm the gateway reflects it:
curl http://<server-ip>:8080/v1/nodes   # zone_a present, status alive, topic_version, mode, cameras
```

---

## 9. Adding a PC / scaling

To add an area: stand up one more PC, give it a **unique `node_id`**, and point
its mqtt sink at the same broker. It self-registers via its retained `config` (§5),
`/v1/nodes` grows by one, and **nothing on the central host changes** — identity
stays globally unique by `(node_id, track_id)` (principle 2).

---

## 10. Reliability & threading

How the principles above hold up under load and failure:

- Each `MqttSink` / `MqttSubscriber` owns a paho network loop in a background
  thread, so publishes are non-blocking; `DiagnosticsPublisher` is a daemon thread.
- The gateway cache is mutated under a lock, and `snapshot_nodes()` copies the
  inner containers under that same lock, so a reader never sees a half-updated node.
- Fail-honestly (principle 4) in practice: a bad message, a dead broker, or one
  failing sink is caught and logged, never fatal to the pipeline or the gateway.
- Identity-preserving degradation: a Mode-2 node that loses one camera keeps
  serving `track2d` from the survivor while `track3d` (which needs two views)
  simply stops matching — track IDs persist across the drop and recovery, and the
  pipeline does not die.

---

## 11. Extending

- **New transport** (MQTT→MQTT-over-WebSocket, ROS, AMQP): implement
  `MetadataSink`, register it, add it to `metadata.sinks`. The orchestrator and
  schema are untouched.
- **New message type**: add a model + `MessageType` member + a `parse_envelope`
  branch; if it's additive within v4, no version bump. Add a `publish_*` default
  no-op to the sink base, a topic template to `MqttSink`, and an orchestrator
  call site.
- **New gateway view**: add a `routes_*.py` reading the per-node cache.

---

## 12. File map

| Concern | Location |
|---|---|
| Message contract | `backbone/comms/schemas.py` |
| UDP / MQTT sinks | `backbone/comms/{udp_sink,mqtt_sink}.py` |
| Fan-out + heartbeat | `backbone/comms/{publisher,diagnostics_publisher}.py` |
| Publish orchestration | `backbone/runtime/orchestrator.py` |
| Zone transitions / snapshots (producers) | `backbone/shared/{zone_transitions,snapshot_writer}.py` |
| Gateway subscriber + cache | `isicomms/isicomms/mqtt_subscriber.py` |
| Gateway REST routes | `isicomms/isicomms/api/routes_*.py` |
| Gateway config / auth | `isicomms/isicomms/{config,api/auth}.py` |
| Deploy profiles | `deploy/{onprem,cloud}/` |
| Topology overview | `docs/architecture-distributed.md` |

## 13. Tests

- Node side: `tests/test_metadata_schemas.py`, `test_udp_sink.py`,
  `test_mqtt_sink.py`, `test_publisher.py`, `test_diagnostics_publisher.py`,
  `test_zone_transitions.py`, `test_snapshot_writer.py`.
- Gateway: `isicomms/tests/` (subscriber, every route, import discipline).
- All MQTT unit tests mock paho. The real paho/broker wire was validated with a
  live two-node replay smoke against a containerized Mosquitto + gateway (it
  surfaced the retained-config-on-connect race fixed in `mqtt_sink.py`).
