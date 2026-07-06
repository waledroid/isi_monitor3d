# RFC-001 — Distributed Network & Communications System

| | |
|---|---|
| **Title** | Warehouse-wide distributed comms: Backbone nodes → MQTT broker → polling gateway |
| **Status** | **Implemented & Validated (on-prem)** · cloud profile pending live smoke |
| **Author** | ISI Monitor 3D — Backbone team (Isitec) |
| **Date** | 2026-07-02 (rev. 2 — PM review: root topic `isiMonitor3D`, `zone` folder topic, per-topic payload schemas, node-config REST answer) |
| **Scope** | The network/comms layer only — the vision pipeline (detection, homography, triangulation) is upstream and out of scope here |
| **Companion** | `docs/mqtt-architecture.md` (field-level technical reference) · `docs/architecture-distributed.md` (topology) |

---

## Abstract

This RFC describes how a single warehouse-vision PC scales to an entire
warehouse. Each PC runs a **Backbone** that converts its camera feeds into
**metric, identity-stable metadata** and *publishes* it to a central **MQTT
broker**. A central **isi-gateway** subscribes to all PCs, aggregates their state,
and serves a single **polling REST API** that AGVs and the WMS consume. The design
is deliberately decoupled, outbound-only, and self-describing: adding a PC
requires **no change to the server**. This document explains the design top-down,
the engineering **methods** chosen at each layer, the **expected result at every
operational stage**, and the **tests conducted** to date.

Three properties recur throughout and are worth fixing up front, so the rest of
the document can refer back to them rather than restate them:
- **Outbound-only edge** — a PC only ever *publishes*; it opens no inbound ports
  and holds no consumer connections.
- **One identity space per PC** — each PC numbers its own tracks; global identity
  is the pair `(node_id, track_id)`, and the gateway *tags*, never merges.
- **Self-describing** — each PC publishes a retained `config` advert, so the
  warehouse map assembles itself with no central list to maintain.

---

## 1. Motivation

The customer operates multiple zones across a warehouse. The requirement:

- Each zone is watched by its own dual-camera PC (latency-sensitive, runs locally).
- AGVs move **freely across zones** and need **one** place to ask "where is
  everything, warehouse-wide?" — not N per-PC connections.
- The whole thing must run on the **intranet** (no cloud dependency), with a
  cloud option available later.

A point-to-point design (every AGV connects to every PC) does not scale and
couples everything together. Instead we use a **publish/subscribe hub**: PCs
publish once; a single aggregator serves everyone.

---

## 2. Goals & Non-Goals

**Goals**
- One Backbone per PC, each independently deployable and restartable.
- A single, poll-able source of truth for AGVs/WMS.
- Add/remove a PC with **zero central reconfiguration**.
- Survive partial failure (a PC, the broker, or an AGV can drop) without taking
  down the rest.
- Run plaintext on a trusted LAN; harden to TLS+auth for internet exposure.

**Non-Goals**
- Image/video transport over the bus (only *references* to media are sent).
- Cross-PC track fusion (per the one-identity-space property above, the gateway
  tags, never merges).
- Controlling the AGVs (we *inform*; the AGV/WMS decides).

---

## 3. High-level architecture

```
            EDGE (per warehouse PC)                     CENTRE (one server)        CONSUMERS
 ┌cameras┐
 │ cam_a ├─RTSP─►┌─────────────────────────────┐
 │ cam_b ├─RTSP─►│ Backbone  node_id = zone_a  │── publish ─┐
 └───────┘       │  vision → metric metadata   │            │ isiMonitor3D/v1/zone_a/…
                 └─────────────────────────────┘            │
 ┌cameras┐                                                  ▼        ┌────────────────────┐   ┌──────────┐
 │ cam_a ├─RTSP─►┌─────────────────────────────┐  ┌──────────────────┐│                    │   │  AGVs    │
 │ cam_b ├─RTSP─►│ Backbone  node_id = dock_1  │─►│ MQTT broker :1883│► …                  │   │  WMS/FMS │
 └───────┘       │  vision → metric metadata   │  │  (Mosquitto)     ││                    │   │  HMI     │
                 └─────────────────────────────┘  └────────┬─────────┘│                    │   └────┬─────┘
 ┌cameras┐                                                 │subscribe │                    │        │
 │ cam_a ├─RTSP─►┌─────────────────────────────┐           │isiMonitor3D/#                 │   GET  │
 └───────┘       │ Backbone  node_id = cold_3  │── publish ┘          ▼                    │        │
  (Mode 1:       │  vision → metric metadata   │           ┌────────────────────┐         │        │
   1 camera)     └─────────────────────────────┘           │  isi-gateway :8080 │◄────────┴────────┘
                                                           │  cache per node_id │ /v1/nodes /v1/tracks
        MQTT over the LAN (intranet, plaintext)            │  REST polling API  │ /v1/zones /v1/passings
        — outbound only from the PCs                       └────────────────────┘
```

Each PC's cameras (1 in Mode 1, 2 in Mode 2) connect **directly to that PC over
RTSP** (PoE, local LAN segment) — camera video never touches the MQTT fabric;
only the extracted metadata does.

**Three roles, three protocols:**
- PCs **publish** metadata (MQTT, outbound only).
- The broker **routes** it (pub/sub).
- The gateway **aggregates** and **serves** it (HTTP polling).

AGVs never speak MQTT and never know how many PCs exist — they poll one URL.

### Where each piece runs — the machines on the LAN

The broker and the gateway live together on **one central machine that is
deliberately *not* one of the warehouse PCs**. That central machine can be a
dedicated server, a small NUC, or a cloud VM — its only job is to run Mosquitto
and isi-gateway. A warehouse PC runs **only a Backbone**; it is never the hub.

For a tiny pilot you *may* co-locate everything on one box, but in production the
central server is its own host. Either way, the only thing the warehouse PCs and
the AGVs need is that the central server's **IP is reachable on the LAN**.

```
   WAREHOUSE PCs (one Backbone each)                CENTRAL SERVER                 AGVs (DHCP)
 cam_a ─RTSP┐
 cam_b ─RTSP┤(PoE)
            ▼
 ┌──────────────────────────────┐
 │ PC-1   192.168.2.41          │── MQTT ─┐
 │ Backbone  node_id = zone_a   │ :1883   │
 └──────────────────────────────┘         │   ┌──────────────────────────┐
 cam_a ─RTSP┐                             │   │                          │
 cam_b ─RTSP┤(PoE)                        │   │                          │
            ▼                             │   │                          │
 ┌──────────────────────────────┐         ├──►│   192.168.2.39           │   ┌────────────┐
 │ PC-2   192.168.2.42          │── MQTT ─┘   │  Mosquitto      :1883    │   │ AGV-1, AGV-2│
 │ Backbone  node_id = dock_1   │ :1883       │  isi-gateway    :8080    │◄──│  (DHCP)     │
 └──────────────────────────────┘             └──────────────────────────┘   │  poll :8080 │
                                                                              └────────────┘
   cameras attach to their PC only;             a separate host — NOT          AGVs poll
   each PC publishes outbound to                any warehouse PC's Docker      192.168.2.39:8080
   192.168.2.39:1883 only
```

The central server here (192.168.2.39) runs the `deploy/` Docker compose stack
(Mosquitto + gateway). The warehouse PCs (192.168.2.41/.42) point their MQTT sink
at `192.168.2.39:1883`; the AGVs poll `http://192.168.2.39:8080/v1/...`.

### Containerized deployment

On the central server the broker and gateway run as **two Docker containers**,
orchestrated with Docker Compose (the profiles live under `deploy/`). The
warehouse PCs run only the Backbone and connect over the LAN — they are **not**
part of this stack.

| Container | Image | Port(s) | Role |
|---|---|---|---|
| `mosquitto` | `eclipse-mosquitto:2` | 1883 (TLS 8883) | the MQTT broker; **persistence on** so retained `config` adverts survive a restart |
| `gateway` | `isi-gateway` (custom, ≈300 MB, `python:3.10-slim`) | 8080 (HTTPS 443 via Caddy, cloud) | subscribes to the broker, caches per node, serves the REST API |

Two profiles ship under `deploy/`:

- **On-prem (LAN):** plaintext broker `:1883` + gateway `:8080`, anonymous — for a
  trusted network.
  Bring it up with `docker compose -f deploy/onprem/docker-compose.yml up -d`.
- **Cloud (internet-facing):** broker over **TLS `:8883`** with authentication, the
  gateway behind a **Caddy** reverse proxy on `:443` with a Bearer token
  (`deploy/cloud/`).

Both containers run `restart: unless-stopped`, so they return after a reboot.
Deploying a new monitoring zone needs **no change to this stack** — only a new
Backbone PC pointed at the broker (§6).

---

## 4. System breakdown — high to low

### Layer 0 — The unit of deployment: a *node*
A "node" is one Backbone process on one PC, identified by a unique **`node_id`**
(`zone_a`, `dock_1`, …). It is self-contained: its own cameras, calibration,
detection model, and identity space.

### Layer 1 — The hub: broker + gateway
The central server runs exactly two services:
- **Mosquitto** — the message router (the post office).
- **isi-gateway** — the subscriber + cache + REST API (the front desk).

### Layer 2 — The transport: MQTT pub/sub
Chosen because it is **lightweight, broker-mediated, and decoupled**: publishers
and subscribers never know each other, connections are long-lived, and the broker
buffers/retains. Topics are hierarchical and namespaced by node (Layer 3).

### Layer 3 — The address space: the topic tree
Everything a node emits is namespaced under `isiMonitor3D/<TOPIC_VERSION>/<node_id>/…`
(currently `TOPIC_VERSION = "v1"`, in `backbone/comms/schemas.py`). Each branch of
the tree carries one kind of information — in plain language:

- **`isiMonitor3D/v1/<node_id>/track2d/<cls>`** — the always-on output. One metric 2D
  position-and-velocity on the warehouse floor for each tracked object of class
  `<cls>`. This is what an AGV navigates against. The `<cls>` is the object class,
  so `track2d/person` carries people positions, `track2d/forklift` forklifts, etc.
- **`isiMonitor3D/v1/<node_id>/track3d/<cls>`** — the same object lifted to full 3D
  `(X, Y, Z)`. Published **only in Mode 2** (two calibrated cameras) and **only for
  subscribed tracks** — i.e. when something downstream actually needs height or
  pose, not for everything.
- **`isiMonitor3D/v1/<node_id>/zone/<zone>`** *(retained)* — the **`zone` folder**: one
  subtopic per zone defined on the node — **up to 6 zones per node**
  (`zone1`–`zone6`, enforced by the node's configuration UI), every one of them
  monitored independently — whose payload is the **current list of objects
  detected inside that zone with their confidence scores** (plus, for pallets,
  the empty/full occupancy verdict). This is the primary integration
  signal for the **FMS/WMS**: subscribe `isiMonitor3D/v1/+/zone/+` and you hold every
  zone's live contents warehouse-wide. Retained + published on change (and a ~1 s
  refresh while occupied), so a late-joining consumer reads the state instantly;
  an emptied zone publishes an explicit empty list, never silence.
- **`isiMonitor3D/v1/<node_id>/zone/<zone>/passings`** — an **event** each time a tracked
  object crosses that zone's boundary: an enter or a leave. One message per
  crossing, not a continuous stream.
- **`isiMonitor3D/v1/<node_id>/zone/<zone>/images/<id>`** — a **URL pointer** to a saved
  snapshot JPEG for that event. The image **bytes are never on the bus** — only a
  link a consumer can fetch out-of-band.
- **`isiMonitor3D/v1/<node_id>/diagnostics/heartbeat`** — the node's health pulse, emitted
  about every 5 s: frame rate, latency, and whether each camera source is alive.
  Absence of the pulse is how the gateway notices a PC went down.
- **`isiMonitor3D/v1/<node_id>/config`** *(retained)* — the node's self-description: its
  area, operational mode, cameras, and zones. **"Retained"** means the broker
  stores the last one and immediately hands it to any subscriber that connects
  later — so a freshly-started gateway learns the layout instantly.

```
isiMonitor3D/v1/zone_a/track2d/person             isiMonitor3D/v1/zone_a/diagnostics/heartbeat
isiMonitor3D/v1/zone_a/track3d/forklift           isiMonitor3D/v1/zone_a/config       (retained)
isiMonitor3D/v1/zone_a/zone/dock_door             (retained — objects now inside + confidence)
isiMonitor3D/v1/zone_a/zone/dock_door/passings    (enter/leave events)
isiMonitor3D/v1/zone_a/zone/dock_door/images/42   (snapshot URL)
```

### Layer 4 — The payload: the message contract, topic by topic

Seven validated JSON message types, one per topic family. The schemas are
enforced in code (pydantic models in `backbone/comms/schemas.py`, strict —
unknown fields are rejected) and that code is the single source of truth; the
tables below are its rendering for integrators. **This schema is the only thing
shared across the process boundary**, and — see *REST parity* at the end of this
layer — **the exact same JSON shapes are served by the REST API**.

Every payload carries two envelope fields; consumers MUST check them first:

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | int | payload-contract version (currently **4**; consumers accept `{3, 4}`) |
| `type` | string | discriminator naming the message type (values below) |

#### The complete topic tree and its payloads

| Topic (under `isiMonitor3D/v1/<node_id>/`) | `type` | QoS | Retained | Cadence |
|---|---|---|---|---|
| `track2d/<cls>` | `track_2d` | 0 | no | per frame, per tracked object |
| `track3d/<cls>` | `track_3d` | 0 | no | Mode 2 only, subscribed tracks |
| **`zone/<zone>`** | `zone_state` | **1** | **yes** | on change + ~1 s refresh while occupied |
| `zone/<zone>/passings` | `passing` | 0 | no | one per boundary crossing |
| `zone/<zone>/images/<track_id>` | `image_ref` | 0 | no | per snapshot-enabled passing |
| `diagnostics/heartbeat` | `diagnostics` | 0 | no | every ~5 s |
| `config` | `config` | 0 | **yes** | once at startup + on every (re)connect |

`node_id` never appears inside a payload — it is carried by the topic (M1), and
the gateway tags REST responses with it.

#### `track_2d` — metric 2D track (the always-on output)

| Field | Type | Meaning |
|---|---|---|
| `ts` | float | capture timestamp, Unix seconds (the honest-latency clock, M5) |
| `track_id` | int | stable per-node identity |
| `cls` | string | object class (`person`, `palette`, `forklift`, …) |
| `xy_m` | [float, float] | floor position, meters |
| `vxy_m` | [float, float] | floor velocity, m/s |
| `confidence` | float 0–1 | detection confidence |
| `cameras_seeing` | [string] | camera ids that observed it this frame |
| `occupancy_state` | string \| null | pallets: `"empty"` / `"full"` / null |
| `occupancy_content` | string \| null | pallets: `"carton"` / `"polybag"` / null |
| `occupancy_confidence` | float 0–1 | confidence of the occupancy verdict |

```json
{"schema_version": 4, "type": "track_2d", "ts": 1751443200.123, "track_id": 42,
 "cls": "person", "xy_m": [3.1, 4.2], "vxy_m": [0.4, -0.1], "confidence": 0.93,
 "cameras_seeing": ["cam_a", "cam_b"], "occupancy_state": null,
 "occupancy_content": null, "occupancy_confidence": 0.0}
```

#### `track_3d` — subscription-driven 3D lift

| Field | Type | Meaning |
|---|---|---|
| `ts`, `track_id`, `cls` | — | as `track_2d` (same identity space) |
| `xyz_m` / `vxyz_m` | [float ×3] | 3D position / velocity, meters |
| `contributing_cameras` | [string] | cameras used in the triangulation |
| `max_reprojection_error_px` | float | the gate value — honesty about geometry quality |
| `keypoints_xyz` | [[float ×3]] \| null | per-keypoint 3D (pose mode, S5.5) |
| `single_view` | bool | true = floor-pinned single-camera fallback |
| `confidence` | float 0–1 | track confidence |

#### `zone_state` — one zone's current contents (**the FMS/WMS signal**)

Published retained on `zone/<zone>` — one topic per configured zone, **up to 6
per node** (`zone1`–`zone6`; the per-node configuration UI enforces the limit,
the fabric itself is zone-count-agnostic). Every configured zone is monitored:
each gets its own retained state at node startup and its own independent
change/refresh publishing. The payload is the full list of objects currently
inside the zone; an empty zone is an explicit `"objects": []`, so a consumer
can always distinguish *empty* from *unknown*.

| Field | Type | Meaning |
|---|---|---|
| `ts` | float | capture timestamp of the frame that produced this state |
| `zone` | string | zone name (matches the `config` advert's zone list) |
| `objects` | array | every tracked object currently inside (fields below) |
| `count` | int | `len(objects)` — consumer convenience |

Each element of `objects`:

| Field | Type | Meaning |
|---|---|---|
| `track_id` | int | the object's per-node track identity |
| `cls` | string | detected class |
| `confidence` | float 0–1 | detection confidence score |
| `xy_m` | [float, float] | floor position inside the zone, meters |
| `occupancy_state` / `occupancy_content` / `occupancy_confidence` | as `track_2d` | pallet empty/full verdict |

```json
{"schema_version": 4, "type": "zone_state", "ts": 1751443201.456, "zone": "dock_door",
 "count": 2, "objects": [
   {"track_id": 42, "cls": "palette", "confidence": 0.91, "xy_m": [3.0, 4.0],
    "occupancy_state": "full", "occupancy_content": "carton", "occupancy_confidence": 0.88},
   {"track_id": 57, "cls": "person", "confidence": 0.87, "xy_m": [3.6, 4.4],
    "occupancy_state": null, "occupancy_content": null, "occupancy_confidence": 0.0}]}
```

#### `passing` — zone boundary-crossing event

| Field | Type | Meaning |
|---|---|---|
| `ts` | float | capture timestamp of the crossing frame |
| `track_id` / `cls` | int / string | who crossed |
| `zone` | string | which zone |
| `direction` | `"enter"` \| `"leave"` | which way |

```json
{"schema_version": 4, "type": "passing", "ts": 1751443201.456,
 "track_id": 42, "cls": "person", "zone": "dock_door", "direction": "enter"}
```

#### `image_ref` — snapshot pointer (URL only, never bytes — M8)

| Field | Type | Meaning |
|---|---|---|
| `ts` / `track_id` / `cls` / `zone` | — | correlate with the `passing` event |
| `url` | string | `file://` or HTTP URL of the saved JPEG |

#### `diagnostics` — the ~5 s heartbeat

| Field | Type | Meaning |
|---|---|---|
| `ts` | float | wall-clock at emit |
| `node_id` | string | (also in the payload here — the heartbeat is the liveness record) |
| `mode` | string | `single_cam_homography` / `dual_cam_homography_triangulation` |
| `sources` | {camera: string} | per-camera `alive` / `exited` / `crashed` |
| `frame_count` / `fps` | int / float | throughput |
| `latency_ms` | {p50, p95, p99, n} | capture→publish percentiles (the < 200 ms KPI) |
| `zones` / `subscriptions` | int | configured counts |
| `calibration` | {loaded, rms_ok, mode} | calibration fact-check |

#### `config` — the retained self-description (M2)

| Field | Type | Meaning |
|---|---|---|
| `ts` / `node_id` / `area` / `mode` | — | node identity |
| `cameras` | [string] | configured camera ids |
| `zones` | array of zone specs | each: `name`, `kind`, `type`, `severity`, `polygon` ([[x, y], … ] meters) |
| `calibration` | {loaded, rms_ok, mode} | calibration fact-check |

#### REST parity — one format, two transports

The gateway's REST responses are **these same models serialized verbatim**
(`model_dump()`), each item tagged with the `node_id` recovered from the topic.
A consumer that parses the MQTT payload parses the REST response, and vice
versa:

| REST endpoint | Payload it returns |
|---|---|
| `/v1/tracks` items | `track_2d` / `track_3d` payloads + `node_id` |
| `/v1/zones` items | zone spec (from `config`) **+ the zone's latest `zone_state` fields** (`objects`, `count`, `state_ts`) |
| `/v1/zones/<name>` | one zone across all nodes defining it: spec + latest `zone_state` per node |
| `/v1/passings` items | `passing` payloads + `node_id` |
| `/v1/diagnostics` | `diagnostics` payloads |
| `/v1/config` | `config` payloads |

Full field-level reference (kept in lock-step with the code):
`docs/mqtt-architecture.md §2`.

### Layer 5 — The mechanisms
The engineering methods that make the above robust — detailed next (§5).

---

## 5. Methods in place

This is the core of the design: ten decisions that make the architecture above
robust. Each is stated as **Problem → Method → ▶ Expected result** so it can be
followed without reading the code. They build on one another — identity and
self-description come first, then resilience, then the seams that let the system
evolve.

### M1 — Identity by topic namespacing
**Problem:** two PCs both number their tracks from 1, so "track 42" is ambiguous.
**Method:** each node prefixes every topic with `isiMonitor3D/v1/<node_id>`. The gateway
reads the `node_id` back out of the topic and tags every record with it, making
global identity the pair `(node_id, track_id)`.
▶ **Expected result:** `track 42` from `zone_a` and `track 42` from `dock_1` never
collide; `/v1/tracks` shows both, distinctly tagged.

### M2 — Self-describing nodes via retained config
**Problem:** the server would otherwise need a hand-maintained list of every PC.
**Method:** at startup each node publishes a **retained** `config` (its area, mode,
cameras, zones). The broker stores the last one and replays it to any subscriber —
even a brand-new gateway — the instant it connects.
▶ **Expected result:** the gateway learns the entire warehouse layout with zero
configuration. Plug in a new PC and it simply *appears* in `/v1/nodes` and
`/v1/zones`; nothing on the server changes.

### M3 — Broker-down-safe startup
**Problem:** if a node crashed because the broker wasn't up yet, deployment order
would matter.
**Method:** nodes connect with paho's `connect_async()` + `loop_start()` — a
background thread retries forever; construction never blocks or raises.
▶ **Expected result:** start the PCs and the server in **any order**. A node boots,
runs its pipeline, and silently connects whenever the broker appears.

### M4 — QoS and retain tuned per message
**Problem:** tracks are high-rate and disposable; config and zone contents must
never be missed.
**Method:** tracks, passings, and diagnostics use **QoS 0** (at-most-once — cheap,
right for a 20 fps stream where the next message supersedes the last); `config`
and the per-zone **`zone_state`** are **retained** (M2) so late subscribers still
receive them, and `zone_state` additionally uses **QoS 1** (at-least-once — it is
low-rate, WMS-consequential absolute state, so duplicates are harmless and a
lost update is not).
▶ **Expected result:** track throughput stays cheap, a consumer that joins late
still learns each node's identity *and every zone's current contents*
immediately. (Whether `passings` should also move to QoS 1 is an open question —
see §10.)

### M5 — Liveness and honest latency via heartbeat
**Problem:** the gateway must tell a downed PC from a merely quiet one, and a
manager must see *true* end-to-end latency, not latency hidden by measuring at
publish time.
**Method:** every node emits a `diagnostics` heartbeat about every 5 s carrying
fps, source liveness, and latency. Latency is computed against the frame's
**capture timestamp**, carried unchanged through the whole pipeline, so a consumer
reading `now − ts` gets the real number. The gateway marks a node **`stale`** if
its last heartbeat is older than `NODE_STALE_AFTER_S` (default 15 s).
▶ **Expected result:** kill a PC and within ~15 s `/v1/nodes` flips it
`alive → stale` (verified live); the heartbeat reports p50/p95/p99 latency.

### M6 — Fail-honestly isolation
**Problem:** one bad message or one dead sink shouldn't cascade.
**Method:** every publish and every inbound parse is wrapped in `try/except` + log;
malformed or incompatible messages are counted, not fatal.
▶ **Expected result:** a corrupt packet, a dead broker, or one failing sink
degrades silently; the pipeline and the gateway keep running.

### M7 — Identity-preserving degradation (Mode 2)
**Problem:** a PC's second camera dies mid-shift.
**Method:** the node keeps serving `track2d` from the surviving camera; `track3d`
(which needs two views) simply stops matching, and track IDs persist across the
drop and recovery.
▶ **Expected result:** 3D output pauses cleanly while 2D output and identities
continue — no restart, no crash.

### M8 — Media by reference, never by value
**Problem:** JPEG bytes would bloat the bus and the broker.
**Method:** snapshots are written locally; only an `image_ref` **URL** is published,
and consumers fetch the bytes out-of-band over HTTP/file.
▶ **Expected result:** MQTT payloads stay small and media scales independently of
the metadata bus.

### M9 — Pluggable transport (the `MetadataSink` seam)
**Problem:** future deployments may want MQTT-over-WebSocket, ROS, or AMQP.
**Method:** UDP and MQTT are both implementations of one ABC, `MetadataSink`, that
the orchestrator builds from config; a node can run several at once (e.g. UDP → a
local dashboard **and** MQTT → the central broker).
▶ **Expected result:** a new transport is a new plugin — the schema, pipeline, and
gateway are untouched.

### M10 — Versioning on two axes for rolling upgrades
**Problem:** you can't restart every PC at once during an upgrade, and a future
breaking change must run *alongside* the old contract — across both what messages
*say* and how they're *addressed*.
**Method:** two independent version axes.
1. **Content** — a single integer `schema_version` on each payload; consumers
   accept a set (currently `{3, 4}`), and adding optional fields is non-breaking.
2. **Addressing** — `TOPIC_VERSION = "v1"` (shared in `backbone/comms/schemas.py`)
   namespaces both the MQTT topics (`isiMonitor3D/v1/<node_id>/…`) and the REST paths
   (`/v1/nodes`, …). The gateway parses any `v\d+` segment out of the topic and
   mounts every resource route under its version prefix; bare paths (`/nodes`) stay
   as back-compat aliases and `/healthz` stays unversioned. A legacy un-versioned
   topic (`isiMonitor3D/<node_id>/…`) is still accepted and reported as `topic_version:
   "v0"`.
▶ **Expected result:** mixed old/new nodes interoperate during a staged rollout. A
future **v2** lands without breaking anyone — nodes publish `isiMonitor3D/v2/…`, the gateway
serves both trees and mounts `/v2/…` alongside `/v1/…`, and consumers migrate on
their own schedule.

---

## 6. Operational stages & expected results

A walkthrough of the system's life, with what a manager should **observe** at
each stage.

### Stage A — Bring up the central server
```bash
docker compose -p on-prem -f deploy/onprem/docker-compose.yml up -d
```
▶ **Expected:** two containers `Up` (`mosquitto`, `gateway`); `GET /healthz` →
`{"ok":true}`; `GET /v1/nodes` → `{"nodes":[],"count":0}` (no PCs yet).

### Stage B — A PC starts publishing
Run a Backbone with `node_id: zone_a` and an mqtt sink (`prefix: isiMonitor3D/v1/zone_a`)
pointing at the server.
▶ **Expected:**
- The broker shows `isiMonitor3D/v1/zone_a/config` (retained) and one retained
  **empty `…/zone/<zone>` state per configured zone** (the folder is discoverable
  before anything moves), then `…/diagnostics/heartbeat` every ~5 s, then
  `…/track2d/<cls>` and per-zone `…/zone/<zone>` updates once detections begin.
- `GET /v1/nodes` → `zone_a`, status **`alive`**, `topic_version: "v1"`, with its
  area/mode/cameras.
- `GET /v1/config` → `zone_a`'s self-description.

### Stage C — Steady state, multiple PCs
Each PC publishes independently under its own prefix.
▶ **Expected:** `GET /v1/nodes` lists every PC; `GET /v1/tracks` returns every track
warehouse-wide, each tagged with its `node_id`; `GET /v1/tracks?zone=<name>` returns
only tracks currently inside that polygon.

### Stage D — An AGV consumes
The AGV polls `GET /v1/tracks` (or `?zone=`) on its own loop.
▶ **Expected:** the AGV sees a consistent, single-source view and steers around
it — with no MQTT client and no awareness of how many PCs exist.

#### The full consumer API surface

Every GET an AGV or the WMS can call today, with its purpose:

| Endpoint | Purpose |
|---|---|
| `GET /v1/nodes` | list every PC and its status (`alive`/`stale`), area, mode, cameras |
| `GET /v1/tracks` | every tracked object warehouse-wide, each tagged with its `node_id` |
| `GET /v1/tracks?node=<id>` | tracks from one PC only |
| `GET /v1/tracks?cls=<class>` | tracks of one object class only (e.g. `person`) |
| `GET /v1/tracks?zone=<name>` | tracks currently inside one zone's polygon |
| `GET /v1/zones` | every zone defined across all PCs (the warehouse map), each with its **live contents** (`objects` + confidence, `count`, `state_ts`) |
| `GET /v1/zones/<name>` | one zone: its spec + the latest per-node object list |
| `GET /v1/passings` | recent enter/leave boundary-crossing events |
| `GET /v1/passings?limit=<n>` | cap how many events come back |
| `GET /v1/passings?node=<id>` | events from one PC only |
| `GET /v1/diagnostics` | each PC's latest heartbeat (fps, latency, source liveness) |
| `GET /v1/config` | each PC's retained self-description |
| `GET /healthz` | gateway up-check (unversioned) |

(The query filters compose: `GET /v1/tracks?cls=person&zone=dock_door`.)

**Growth path.** New endpoints or changed response shapes are introduced under a
**new version prefix** (`/v2/...`) served *alongside* `/v1` (per M10). Existing AGVs
keep calling `/v1` and migrate to `/v2` on their own schedule — no flag day.

### Stage E — Failure: a PC drops
▶ **Expected:** within ~15 s that node flips to **`stale`** in `/v1/nodes`; its last
tracks linger then age out; **all other PCs are unaffected**. On recovery it
re-announces (retained config re-published) and returns to `alive`.

### Stage F — Failure: the broker restarts
▶ **Expected:** nodes and the gateway reconnect automatically; nodes re-assert
their retained `config`; the warehouse map rebuilds itself with no manual step.

### Stage G — Scaling: add a zone
Stand up one more PC with a new `node_id` pointed at the same broker.
▶ **Expected:** it appears in `/v1/nodes`/`/v1/zones` on its own; **nothing on the
server changes.**

### Configuring a node — is there a REST API on the backbone nodes?

**Yes — with a deliberate split.** Each warehouse PC hosts a configuration REST
API, but it belongs to the **co-located operator dashboard (`monitor_web`, port
8000)**, not to the Backbone process itself:

- **`GET/POST /api/config`** on the PC's dashboard edits that node's
  configuration — cameras (RTSP/USB), detection model, zones, and the MQTT sink
  (broker host + topic prefix) — writing `backbone.yaml`/`zones.yaml`
  atomically. **`POST /api/control/{start,stop}`** restarts the Backbone to
  apply. This is LAN-local operator tooling on each PC, deliberately **outside
  the comms fabric**: the broker and gateway stay read-only aggregation.
- **The Backbone process itself intentionally opens no inbound port.** This
  preserves the outbound-only edge property (firewall-friendly, zero attack
  surface on the vision process) and the industrial default of
  config-as-a-file under systemd with deterministic restart.
- The centre never needs write access to *see* a config change: the node
  re-publishes its retained `config` advert (M2), so the new zones/cameras
  appear in `/v1/nodes`, `/v1/zones` automatically.

If **centralized remote configuration** is ever required, the
architecture-consistent path is **config-over-MQTT**: each node subscribes to a
command topic (`isiMonitor3D/v1/<node_id>/cmd/config`) over its *existing
outbound* broker connection — still no inbound port on the PC. This is recorded
as an open question in §10 (it needs an authorization story — who may command a
node — before it is safe to build).

---

## 7. Security model

Functional-first: works open on a trusted LAN; every control defaults **off** and
is enabled for internet exposure.

```
 ON-PREM (LAN)                         CLOUD (internet-facing)
 ┌───────────────────────────┐         ┌──────────────────────────────────────┐
 │ Mosquitto :1883  plaintext│         │ Mosquitto :8883  TLS + password file  │
 │ allow_anonymous true      │   ──►   │ allow_anonymous false                 │
 │ gateway :8080  open        │         │ gateway behind Caddy :443  HTTPS      │
 │ no certs                  │         │ + ISI_GATEWAY_API_TOKEN (Bearer)       │
 └───────────────────────────┘         │ self-signed CA via gen-certs.sh        │
                                        └──────────────────────────────────────┘
```

Hardening checklist before any internet exposure: broker auth, broker TLS, API
Bearer token, never expose the broker/API anonymously. The **node code, schema,
topics, and gateway are identical** between profiles — only transport security
changes.

---

## 8. Tests conducted & results

Validation was done at three levels. **All green.**

### 8.1 Unit / component (logic correctness — paho mocked)
| Suite | Tests | Proves |
|---|---|---|
| `test_metadata_schemas` | 40 | every message type (incl. `zone_state`) validates/serialises; version gate |
| `test_udp_sink` / `test_mqtt_sink` | 10 / 28 | topic templates (incl. the `zone/` folder, all 6 per-zone topics), retain-config, retained QoS-1 zone state, broker-down-safe, TLS kwargs, shutdown order |
| `test_publisher` / `test_diagnostics_publisher` | 15 / 14 | per-sink isolation (incl. `publish_zone_state`); heartbeat content |
| `test_zone_transitions` / `test_zone_state` / `test_snapshot_writer` | 16 / 12 / 6 | passing detection; zone-contents change/refresh/explicit-empty; **6 zones per node each independently monitored**; URL-only image refs |
| `isi_gateway/tests` | 82 | subscriber cache (incl. per-zone state, **all 6 zones of a node served enriched**), every REST route (incl. `/v1/zones` enrichment + `/v1/zones/{name}`), liveness, **import discipline**, **`/v1` prefix + bare aliases** (`test_app_versioning`), **version-aware topic parse + `topic_version`** (subscriber tests) |
| Backbone full suite | **519** | no regression across the pipeline |
| monitor_web (dashboard consumer) | **278** | the operator UI consumes the contract |

▶ **Result:** the *logic* of every component is correct in isolation. **Versioning
(M10) is implemented and covered:** `test_app_versioning.py` asserts every resource
route answers under `/v1` and the bare alias with identical shape (and `/healthz`
both ways); the subscriber tests assert `isiMonitor3D/v1/zone_a/...` parses to
`topic_version: "v1"` and legacy `isiMonitor3D/zone_a/...` to `"v0"`; `TOPIC_VERSION == "v1"`
is pinned in `test_metadata_schemas`.

### 8.2 Live wire smoke (real broker — the part mocks can't prove)
Setup: a real Mosquitto, **two** Backbone nodes (`zone_a`, `zone_b`) on recorded
video (no cameras needed), and the gateway.
```
node zone_a ─┐                         ┌─ GET /v1/nodes  → zone_a + zone_b, both alive
node zone_b ─┼─► mosquitto :1883 ─► gateway ─┤ GET /v1/config → both retained adverts
             │   (real paho wire)            └─ liveness  → alive → stale on stop
mosquitto_sub -t 'isiMonitor3D/#' shows BOTH node trees
```
▶ **Result:** `config` (retained), `diagnostics`, and `track2d` all delivered on
the real wire for both nodes; the gateway aggregated both; liveness flipped
correctly. **This smoke caught a real bug** — the retained `config` advert raced
the async connection and was being dropped — now fixed (re-publish on connect).

### 8.3 Deploy validation (the actual artifact — Docker)
Brought up the **on-prem `docker compose` profile**: a containerised Mosquitto +
the lean gateway image, then pointed a node at the containerised stack.
▶ **Result:** both containers healthy on standard ports; the gateway image
**builds lean (~300 MB) and runs**; a node published through the *containerised*
broker and appeared in the *containerised* gateway's `/v1/nodes`. **This run caught a
second real bug** — the gateway image was missing a transitive dependency
(`calibration`/OpenCV leaked in through a zone import) — now fixed by making the
zone geometry import-light, keeping the image minimal.

### 8.4 Not yet tested (open)
- **Cloud profile** (TLS `:8883`, Caddy `:443`, Bearer token, cert generation):
  artifacts exist and are statically valid, but no live `docker compose up` with a
  real TLS handshake has been run.

---

## 9. Status, risks, open items

**Status:** the on-prem distributed comms system is **implemented and validated
end-to-end** (unit + live wire + containerised deploy). It is currently on the
`backbone-mqtt-sink` branch.

| Item | State |
|---|---|
| Node publishing (7 message types, incl. `zone_state`) | ✅ done + live-validated |
| Central gateway + REST API | ✅ done + live-validated |
| Versioning (M10: `/v1` API + `isiMonitor3D/v1/...` topics, `v0` legacy fallback) | ✅ done + unit-covered |
| On-prem Docker deploy | ✅ stood up + validated |
| Cloud (TLS) Docker deploy | 🟡 artifacts ready, live smoke pending |
| Merge to `main` | 🟡 pending |

**Risks / notes for the manager**
- The two bugs above were **only** discoverable by running real infrastructure —
  reinforcing that the live smoke and Docker validation were necessary, not
  optional. Both are fixed and committed.
- The cloud profile should get one live TLS smoke before any internet-facing
  pilot.
- Latency stays within the < 200 ms KPI; the comms layer adds negligible overhead
  (publish is non-blocking).

---

## 10. Request for comments

Feedback sought on:

1. **Delivery guarantee for safety-critical messages (QoS).** MQTT offers a choice
   of delivery guarantee per message:
   - **QoS 0 — at-most-once.** What tracks use today: the cheapest option, with no
     acknowledgement. If a message is lost it is simply gone, which is fine for a
     20 fps stream because the next position message (50 ms later) supersedes it.
   - **QoS 1 — at-least-once.** The broker re-sends until the subscriber
     acknowledges, *guaranteeing* delivery at the cost of extra traffic and the
     possibility of duplicates the consumer must tolerate.

   Separately, **retained** means the broker stores the *last* message on a topic
   and hands it to any late subscriber — which is why `config` is retained (a new
   gateway gets the layout instantly), but it is not the same as a delivery
   guarantee for the *next* message.

   The per-zone **`zone_state`** already uses **QoS 1 + retained** (M4): it is
   low-rate, WMS-consequential absolute state, so the cost is negligible and
   duplicates are harmless.

   **Open question:** should safety-critical, AGV-facing messages — notably
   `passings` (e.g. a person *entering a danger zone*) — also move to **QoS 1**
   for guaranteed delivery, accepting the added cost and duplicate-handling,
   while the high-rate `track2d`/`track3d` streams stay on **QoS 0**? This is a
   deliberate trade the manager can weigh: guaranteed safety events vs. a busier
   bus.

2. **API surface** — are the current endpoints (see §6, *The full consumer API
   surface*) sufficient for the WMS/AGV integration, or is a push/streaming
   (WebSocket) variant wanted alongside polling?

3. **Auth** — is Bearer-token API auth + broker password/TLS sufficient for the
   site's network policy?

4. **Centralized remote configuration.** Today a node is configured on-site via
   its co-located dashboard REST API (see §6, *Configuring a node*); the
   Backbone itself opens no inbound port. Should we add **config-over-MQTT**
   (a `isiMonitor3D/v1/<node_id>/cmd/config` command topic consumed over the
   node's existing outbound broker connection) so nodes can be reconfigured
   from the centre? It preserves the outbound-only edge, but requires an
   authorization story (who may command a node, and how commands are signed)
   before it is safe to build.

---

*References: `docs/mqtt-architecture.md` (field-level reference),
`docs/architecture-distributed.md` (topology), `deploy/README.md` (profiles),
`isi_gateway/README.md` (gateway API).*
