# RFC-001 — Distributed Network & Communications System

| | |
|---|---|
| **Title** | Warehouse-wide distributed comms: Backbone nodes → MQTT broker → polling gateway |
| **Status** | **Implemented & Validated (on-prem)** · cloud profile pending live smoke |
| **Author** | ISI Monitor 3D — Backbone team (Isitec) |
| **Date** | 2026-06-26 |
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
- Cross-PC track fusion (each PC owns its own identity; the gateway tags, not
  merges).
- Controlling the AGVs (we *inform*; the AGV/WMS decides).

---

## 3. High-level architecture

```
        EDGE (per warehouse PC)                    CENTRE (one server)        CONSUMERS
 ┌─────────────────────────────┐
 │ Backbone  node_id = zone_a  │── publish ─┐
 │  cameras → vision → metadata│ isi/v1/zone_a/…│
 └─────────────────────────────┘            │
 ┌─────────────────────────────┐            ▼         ┌────────────────────┐      ┌──────────┐
 │ Backbone  node_id = dock_1  │── publish ─┼────────►│  MQTT broker :1883 │      │  AGVs    │
 │  cameras → vision → metadata│ isi/v1/dock_1/…│      │  (Mosquitto)       │      │  WMS     │
 └─────────────────────────────┘            │         └─────────┬──────────┘      │  HMI     │
 ┌─────────────────────────────┐            │                   │ subscribe isi/# └────┬─────┘
 │ Backbone  node_id = cold_3  │── publish ─┘                   ▼                      │
 │  …                          │                       ┌────────────────────┐   GET   │
 └─────────────────────────────┘                       │  isi-gateway :8080 │◄────────┘
                                                        │  cache per node_id │ /v1/nodes /v1/tracks
        MQTT over the LAN (intranet, plaintext)         │  REST polling API  │ /v1/zones /v1/passings
        — outbound only from the PCs                    └────────────────────┘
```

**Three roles, three protocols:**
- PCs **publish** metadata (MQTT, outbound only).
- The broker **routes** it (pub/sub).
- The gateway **aggregates** and **serves** it (HTTP polling).

AGVs never speak MQTT and never know how many PCs exist — they poll one URL.

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
Everything a node emits is namespaced under `isi/<TOPIC_VERSION>/<node_id>/…`
(currently `TOPIC_VERSION = "v1"`, in `backbone/comms/schemas.py`):
```
isi/v1/zone_a/track2d/person          isi/v1/zone_a/diagnostics/heartbeat
isi/v1/zone_a/track3d/forklift        isi/v1/zone_a/config           (retained)
isi/v1/zone_a/zones/dock_door/passings
isi/v1/zone_a/images/dock_door/42
```

### Layer 4 — The payload: the message contract
Six validated JSON message types (`track_2d`, `track_3d`, `passing`, `image_ref`,
`diagnostics`, `config`), each with a `schema_version` and `type`. This schema is
the **only** thing shared across the process boundary. Full field reference:
`docs/mqtt-architecture.md §2`.

### Layer 5 — The mechanisms
The engineering methods that make the above robust — detailed next (§5).

---

## 5. Methods in place

This is the core of the design: *how* each guarantee is achieved.

### M1 — Identity by topic namespacing
**Problem:** two PCs both number their tracks from 1.
**Method:** each node prefixes every topic with `isi/v1/<node_id>`. The gateway
reads the `node_id` back out of the topic (skipping the `v\d+` version segment) and
tags every record with it. Global identity is the pair **`(node_id, track_id)`**.
▶ **Expected result:** `track 42` from `zone_a` and `track 42` from `dock_1` never
collide; `/v1/tracks` shows both, distinctly tagged.

### M2 — Self-describing nodes via *retained* config
**Problem:** the server would otherwise need a hand-maintained list of every PC.
**Method:** at startup each node publishes a **retained** `config` message
(its area, mode, cameras, zones, calibration status). The broker keeps the last
one forever (`persistence true`). Any gateway that connects — even a brand-new one
— is immediately replayed every node's `config`.
▶ **Expected result:** a freshly-started gateway learns the entire warehouse
layout with **zero configuration**. Plug in a new PC and it simply *appears* in
`/v1/nodes` and `/v1/zones`.

### M3 — Broker-down-safe startup
**Problem:** if a node crashed because the broker wasn't up yet, deployment order
would matter.
**Method:** nodes connect with paho's `connect_async()` + `loop_start()` — a
background thread retries forever; construction never blocks or raises.
▶ **Expected result:** you can start the PCs and the server **in any order**. A
node boots, runs its pipeline, and silently connects whenever the broker appears.

### M4 — QoS & retain tuned per message
**Problem:** tracks are high-rate and disposable; config must never be missed.
**Method:** tracks/passings/diagnostics use **QoS 0** (at-most-once — cheap,
right for a 20 fps stream where the next message supersedes the last). `config`
is **retained** so late subscribers still get it.
▶ **Expected result:** track throughput stays cheap; a consumer that joins late
still learns each node's identity immediately.

### M5 — Liveness by heartbeat freshness
**Problem:** how does the gateway know a PC went down (vs. just quiet)?
**Method:** every node emits a `diagnostics` heartbeat every ~5 s. The gateway
marks a node **`stale`** if its last heartbeat is older than `NODE_STALE_AFTER_S`
(default 15 s).
▶ **Expected result:** kill a PC and within ~15 s `/v1/nodes` flips it
`alive → stale`. (Verified live.)

### M6 — Fail-honestly isolation
**Problem:** one bad message or one dead sink shouldn't cascade.
**Method:** every publish and every inbound parse is wrapped `try/except` + log.
Malformed/incompatible messages are counted, not fatal.
▶ **Expected result:** a corrupt packet, a dead broker, or one failing sink
degrades silently; the pipeline and the gateway keep running.

### M7 — Identity-preserving degradation (Mode 2)
**Problem:** a PC's second camera dies mid-shift.
**Method:** the node keeps serving `track2d` from the surviving camera; `track3d`
(which needs two views) simply stops matching. Track IDs persist across the
drop/recovery.
▶ **Expected result:** 3D output pauses cleanly; 2D output and identities
continue; no restart, no crash.

### M8 — Honest latency (capture-clock)
**Problem:** latency measured at publish time hides the real pipeline cost.
**Method:** every message's timestamp `ts` is the frame's **capture time**,
carried unchanged through the whole pipeline.
▶ **Expected result:** a consumer computing `now − ts` reads true end-to-end
latency; the diagnostics heartbeat reports p50/p95/p99.

### M9 — Media by reference, never by value
**Problem:** JPEG bytes would bloat the bus and the broker.
**Method:** snapshots are written locally; only an `image_ref` **URL** is
published. Consumers fetch bytes out-of-band over HTTP/file.
▶ **Expected result:** MQTT payloads stay small; media scales independently.

### M10 — Pluggable transport (the `MetadataSink` seam)
**Problem:** future deployments may want MQTT-over-WebSocket, ROS, or AMQP.
**Method:** UDP and MQTT are both implementations of one ABC, `MetadataSink`;
the orchestrator builds them from config. A node can run several at once
(e.g. UDP→local dashboard **and** MQTT→central).
▶ **Expected result:** a new transport is a new plugin — the schema, pipeline,
and gateway are untouched.

### M11 — Lean, isolated gateway image
**Problem:** the gateway shouldn't drag in CUDA/OpenCV/the whole vision stack.
**Method:** the gateway imports only the schema + the zone geometry; the zone
point-in-polygon test is pure-numpy (no OpenCV); a build allowlist + an
import-discipline test keep the Docker image minimal.
▶ **Expected result:** a ~300 MB gateway image that builds and runs anywhere,
including a tiny cloud VM, with no GPU.

### M12 — Schema versioning for rolling upgrades
**Problem:** you can't restart every PC at once during an upgrade.
**Method:** a single integer `schema_version` on the **payload**; consumers accept
a set (currently `{3, 4}`); adding optional fields is non-breaking. This is the
*content* axis — see M13 for the orthogonal *addressing* axis.
▶ **Expected result:** mixed old/new nodes interoperate during a staged rollout.

### M13 — Explicit `/v1` API + topic versioning
**Problem:** `schema_version` evolves message *content*, but the *addressing* — the
REST paths and the MQTT topic tree — had no version, so a future breaking change to
the contract/shape of the API or topics couldn't run alongside the old one.
**Method:** a second, orthogonal version axis, `TOPIC_VERSION = "v1"` (in
`backbone/comms/schemas.py`, shared so node + gateway agree). Topics are namespaced
`<base>/<version>/<node_id>/<suffix>` (e.g. `isi/v1/zone_a/track2d/person`); the
operator sets the MQTT `prefix` to `isi/v1/<node_id>`. The gateway mounts every
resource route under **`/v1`** (`/v1/nodes`, `/v1/tracks`, …) **and** keeps the bare
paths (`/nodes`, …) as back-compat aliases; `/healthz` stays unversioned. The
subscriber parses the version out of the topic (a `^v\d+$` segment after the base);
an un-versioned legacy topic (`isi/<node_id>/...`) is still accepted and reported as
`topic_version: "v0"`, and each node's `topic_version` is surfaced on `/v1/nodes`
and `/v1/config`.
▶ **Expected result:** **two independent version axes** — payload (`schema_version`)
for content, path/topic (`/v1`, `isi/v1/...`) for addressing. A future **v2** lands
without breaking anyone: nodes publish `isi/v2/...`; the gateway already parses any
`v\d+` segment so it serves both trees at once; it mounts `/v2/...` alongside
`/v1/...`; consumers migrate on their own schedule; `v1` is deprecated on a
published date.

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
Run a Backbone with `node_id: zone_a` and an mqtt sink (`prefix: isi/v1/zone_a`)
pointing at the server.
▶ **Expected:**
- The broker shows `isi/v1/zone_a/config` (retained), then `…/diagnostics/heartbeat`
  every ~5 s, then `…/track2d/<cls>` once detections begin.
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
| `test_metadata_schemas` | 32 | every message type validates/serialises; version gate |
| `test_udp_sink` / `test_mqtt_sink` | 10 / 23 | topic templates, retain-config, broker-down-safe, TLS kwargs, shutdown order |
| `test_publisher` / `test_diagnostics_publisher` | 12 / 14 | per-sink isolation; heartbeat content |
| `test_zone_transitions` / `test_snapshot_writer` | 16 / 6 | passing detection; URL-only image refs |
| `isi_gateway/tests` | 76 | subscriber cache, every REST route, liveness, **import discipline**, **`/v1` prefix + bare aliases** (`test_app_versioning`), **version-aware topic parse + `topic_version`** (subscriber tests) |
| Backbone full suite | **489** | no regression across the pipeline |
| monitor_web (dashboard consumer) | **272** | the operator UI consumes the contract |

▶ **Result:** the *logic* of every component is correct in isolation. **Versioning
(M13) is implemented and covered:** `test_app_versioning.py` asserts every resource
route answers under `/v1` and the bare alias with identical shape (and `/healthz`
both ways); the subscriber tests assert `isi/v1/zone_a/...` parses to
`topic_version: "v1"` and legacy `isi/zone_a/...` to `"v0"`; `TOPIC_VERSION == "v1"`
is pinned in `test_metadata_schemas`.

### 8.2 Live wire smoke (real broker — the part mocks can't prove)
Setup: a real Mosquitto, **two** Backbone nodes (`zone_a`, `zone_b`) on recorded
video (no cameras needed), and the gateway.
```
node zone_a ─┐                         ┌─ GET /v1/nodes  → zone_a + zone_b, both alive
node zone_b ─┼─► mosquitto :1883 ─► gateway ─┤ GET /v1/config → both retained adverts
             │   (real paho wire)            └─ liveness  → alive → stale on stop
mosquitto_sub -t 'isi/#' shows BOTH node trees
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
| Node publishing (6 message types) | ✅ done + live-validated |
| Central gateway + REST API | ✅ done + live-validated |
| Versioning (M13: `/v1` API + `isi/v1/...` topics, `v0` legacy fallback) | ✅ done + unit-covered |
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
1. **Cloud rollout** — do we pilot the on-prem profile first and defer cloud, or
   smoke the TLS profile now?
2. **Retention policy** — should `track2d`/`passing` use QoS 1 (at-least-once) for
   any safety-critical AGV use, accepting the cost?
3. **API surface** — are the current endpoints (`/v1/nodes /v1/tracks /v1/zones
   /v1/passings /v1/diagnostics /v1/config`; bare aliases retained) sufficient for
   the WMS/AGV integration, or is a push/streaming (WebSocket) variant wanted
   alongside polling?
4. **Auth** — is Bearer-token API auth + broker password/TLS sufficient for the
   site's network policy?

---

*References: `docs/mqtt-architecture.md` (field-level reference),
`docs/architecture-distributed.md` (topology), `deploy/README.md` (profiles),
`isi_gateway/README.md` (gateway API).*
