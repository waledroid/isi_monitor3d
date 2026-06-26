# MQTT & Distributed Architecture

How the ISI Monitor 3D system scales from one camera rig to a whole warehouse:
many **Backbone** nodes (one per PC) publish metric metadata to a central **MQTT
broker**, an **isi-gateway** aggregates it, and AGVs/WMS poll a single REST API.

This document is the end-to-end reference: the comms module internals, the
message contract, the broker, the gateway, the identity model, the security
model, and a start-to-finish runbook.

```
 Warehouse PC 1            Warehouse PC 2            Warehouse PC 3
 Backbone node_id=zone_a   Backbone node_id=dock_1   Backbone node_id=cold_3
        │ publish isi/zone_a/…    │ isi/dock_1/…            │ isi/cold_3/…
        └──────────┬──────────────┴─────────────┬───────────┘   MQTT over the LAN
                   ▼                             ▼
        ┌────────────────────────────────────────────────┐
        │  CENTRAL SERVER (e.g. 192.168.2.39)             │
        │   • Mosquitto broker      :1883  (TLS :8883)     │  routes isi/#
        │   • isi-gateway REST API  :8080  (HTTPS :443)    │  caches per node_id
        └────────────────────────────────────────────────┘
                   ▲  GET /nodes /tracks /zones /passings …
            AGVs / WMS poll here (HTTP only)
```

---

## 1. Design principles

The MQTT layer obeys the Backbone's non-negotiables:

1. **The schema is the only contract.** A node shares **zero code** with a
   consumer. Everything crosses the boundary as validated JSON
   (`backbone/comms/schemas.py`). To add a capability you expand the schema, not
   share a module.
2. **One identity space, per node.** Each Backbone owns its own `track_id`
   counter. Global identity is the pair **`(node_id, track_id)`** — the gateway
   tags every track with the `node_id` it read from the topic.
3. **Outbound-only nodes.** A node only *publishes*. It never subscribes, never
   listens, never knows another node exists. Firewall-friendly and decoupled.
4. **Fail honestly.** Every sink call is wrapped in `try/except` + log; a dead
   broker or a UDP hiccup degrades silently instead of crashing the pipeline.
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

### The six message types

| `type` | Model | Cadence |
|---|---|---|
| `track_2d` | `Track2DMessage` | every frame, every track — **always** |
| `track_3d` | `Track3DMessage` | **Mode 2** only, subscription-driven |
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
The `track_id` is **identical** to the corresponding `Track2DMessage` — the "one
identity space" principle. The triangulation layer augments tracks with 3D and
**never re-IDs**.

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

### Schema versioning

`schema_version` is a single integer consumers MUST read before parsing. **Adding
optional, defaulted fields is non-breaking** (a v3 consumer ignores them);
renaming/removing is breaking and bumps the version. `parse_envelope` currently
accepts `{3, 4}` so a v4 node and a v3 consumer interoperate during a rollout;
anything outside that set raises `SchemaVersionError`.

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
      prefix: isi/zone_a
```

A non-abstract default no-op base keeps the seam count at five even though sinks
gained `publish_event` / `publish_image_ref` / `publish_diagnostics` /
`publish_config` over time.

### `MqttSink` behaviour (the details that matter operationally)

- **Topic templates** (substituted per message; `prefix = isi/<node_id>`):
  ```
  {prefix}/track2d/{cls}              {prefix}/diagnostics/heartbeat
  {prefix}/track3d/{cls}              {prefix}/config            (retain=True)
  {prefix}/zones/{zone}/passings
  {prefix}/images/{zone}/{track_id}
  ```
- **`track_id` is intentionally NOT in the topic** — it's in the payload. This
  keeps the topic cardinality bounded (a wildcard `…/track2d/person` follows a
  whole class, not one ephemeral id).
- **Topic sanitisation.** `cls` and `zone` are run through a translation that
  replaces `/ + #` with `_`, so a zone named "dock/door" can't break the topic
  hierarchy or inject wildcards.
- **QoS 0 by default** (set per sink). Tracks are high-rate and transient —
  at-most-once is the right trade. `config` is the exception: see below.
- **`config` is always published with `retain=True`**, regardless of the sink's
  `retain` setting. The broker keeps the last `config` per topic forever.
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

**The self-describing trick (why `persistence` + `retain` matter):** each node
publishes its `config` retained to `isi/<node_id>/config`. The broker holds the
latest one permanently. So when the gateway connects — first boot, reconnect, or
a totally fresh gateway — and subscribes to `isi/#`, the broker **immediately
replays every node's config**. The gateway learns the entire warehouse layout
(nodes, areas, modes, cameras, zones) with **zero central configuration**. Add a
PC and it simply *appears*; no registry, no node list to maintain.

---

## 6. The gateway — `isi_gateway/`

A small FastAPI service that turns the MQTT firehose into a poll-able API.

### Subscriber + cache — `mqtt_subscriber.py`
- Subscribes `{base}/#` (`base = isi`), derives `node_id` from the topic
  (`topic.split('/')[1]`), `parse_envelope`s each message, and updates a
  **thread-safe per-node `NodeState`**: latest tracks (by id), a passings ring
  buffer (`passings_buffer`, default 200), last diagnostics, the config advert,
  and `last_seen`.
- `connect_async()` + `loop_start()` (broker-down-safe, same as the sink).
- `snapshot_nodes()` copies the inner containers **under the lock** so a reader
  never sees a half-updated node.
- Counts malformed / wrong-version messages instead of crashing.

### Liveness
A node whose last heartbeat is older than `node_stale_after_s` (default **15 s**)
flips `alive → stale` in `/nodes`. (Observed live: a stopped node goes stale.)

### REST endpoints (`api/routes_*.py`)

| Endpoint | Returns |
|---|---|
| `GET /healthz` | liveness probe — never touches the broker |
| `GET /nodes` | per-node summary: status, area, mode, cameras, fps, latency |
| `GET /tracks` | all tracks, each tagged `node_id`; filters `?node=&cls=&zone=` |
| `GET /zones` | union of all nodes' zones — the global warehouse map |
| `GET /passings` | recent crossings; `?limit=&node=` |
| `GET /diagnostics` | per-node heartbeats |
| `GET /config` | each node's raw self-description |

`GET /tracks?zone=<name>` does a **point-in-polygon** test (`backbone.shared.zones
.Zone.contains`, pure-numpy so the gateway image stays lean) to return only
tracks currently inside that zone — across all nodes.

### Configuration (`config.py`, env prefix `ISI_GATEWAY_`)

| Env var | Default | Meaning |
|---|---|---|
| `ISI_GATEWAY_HOST` / `_PORT` | `0.0.0.0` / `8080` | API bind |
| `ISI_GATEWAY_MQTT_HOST` / `_PORT` | `127.0.0.1` / `1883` | broker |
| `ISI_GATEWAY_MQTT_BASE` | `isi` | topic root subscribed (`isi/#`) |
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
→ Mosquitto on `<server-ip>:1883`, isi-gateway on `<server-ip>:8080`, both
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
      prefix: isi/zone_a              # = isi/<node_id>
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
curl http://<server-ip>:8080/nodes                       # who's online, where
curl http://<server-ip>:8080/tracks                      # every track, all PCs
curl "http://<server-ip>:8080/tracks?node=zone_a&cls=person"
curl "http://<server-ip>:8080/tracks?zone=dock_door"     # tracks inside a zone
curl http://<server-ip>:8080/zones                       # global warehouse map
```
An AGV polls `/tracks` (or `/tracks?zone=…`) on its own loop and steers — no MQTT
client, no per-node awareness, one HTTP endpoint.

### Verify the whole chain
```bash
# watch every node's traffic on the broker:
mosquitto_sub -h <server-ip> -t 'isi/#' -v
#  → isi/zone_a/config (retained) · isi/zone_a/diagnostics/heartbeat · isi/zone_a/track2d/person …
# confirm the gateway reflects it:
curl http://<server-ip>:8080/nodes      # zone_a present, status alive, mode, cameras
```

---

## 9. Adding a PC / scaling

To add an area: stand up one more PC, give it a **unique `node_id`**, point its
mqtt sink at the same broker. It self-registers via its retained `config`, and
`/nodes` grows by one. **Nothing on the server changes.** Identity stays unique
because everything is namespaced `(node_id, track_id)`.

---

## 10. Reliability & threading

- Each `MqttSink` / `MqttSubscriber` owns a paho network loop in a background
  thread; publishes are non-blocking.
- `DiagnosticsPublisher` is a daemon thread.
- The gateway cache is mutated under a lock; snapshots copy-under-lock.
- Every sink call and every inbound parse is `try/except` + log — a bad message,
  a dead broker, or one bad sink never takes down the pipeline or the gateway.
- A Mode-2 node that loses one camera keeps serving `track2d` from the survivor;
  `track3d` (which needs 2 views) simply stops matching. The pipeline does not die.

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
| Gateway subscriber + cache | `isi_gateway/isi_gateway/mqtt_subscriber.py` |
| Gateway REST routes | `isi_gateway/isi_gateway/api/routes_*.py` |
| Gateway config / auth | `isi_gateway/isi_gateway/{config,api/auth}.py` |
| Deploy profiles | `deploy/{onprem,cloud}/` |
| Topology overview | `docs/architecture-distributed.md` |

## 13. Tests

- Node side: `tests/test_metadata_schemas.py`, `test_udp_sink.py`,
  `test_mqtt_sink.py`, `test_publisher.py`, `test_diagnostics_publisher.py`,
  `test_zone_transitions.py`, `test_snapshot_writer.py`.
- Gateway: `isi_gateway/tests/` (subscriber, every route, import discipline).
- All MQTT unit tests mock paho. The real paho/broker wire was validated with a
  live two-node replay smoke against a containerized Mosquitto + gateway (it
  surfaced the retained-config-on-connect race fixed in `mqtt_sink.py`).
