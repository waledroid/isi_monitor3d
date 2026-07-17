# Wire, MQTT & the probe — schema v6

**WHY** — process boundaries are contractual (principle #5): the only way anything talks to anything is versioned JSON. `backbone/comms/schemas.py` is the single contract file; consumers read `schema_version` before parsing; evolution is additive within a version, breaking changes bump it.

**WHAT** — `SCHEMA_VERSION = 6` (accepted: 3–6), `TOPIC_VERSION = "v1"` (topic layout, independent of payload version). All types are pydantic, `extra="forbid"`, frozen.

## Message types (`MessageType` enum, all real)

| Type | Direction | Transport | Purpose |
|---|---|---|---|
| `track_2d` | engine → out | UDP + MQTT | always-on metric track: `track_id`, `cls`, `xy_m`, `vxy_m`, `cameras_seeing` + v2 pallet `occupancy_*` |
| `track_3d` | engine → out | UDP + MQTT | subscription-driven: `xyz_m`, `contributing_cameras`, `max_reprojection_error_px`, `single_view` fallback — **same `track_id`** as 2D |
| `zone_state` | engine → out | UDP + MQTT (**retained** on `{prefix}/zone/{zone}`) | current contents of one zone (the WMS/FMS signal); empty zone = explicit `objects=()`, never silence |
| `passing` | engine → out | UDP + MQTT | zone `enter`/`leave` event; key on `zone_id` (stable), not `zone` (renamable label) |
| `image_ref` | engine → out | UDP + MQTT | snapshot URL alongside a passing — **never raw bytes** |
| `proximity` | engine → out | UDP + MQTT (retained) | person↔object floor distances — the safety signal |
| `observations` | engine → out | **UDP only** | per-camera raw detections (bbox, foot, `mask_poly`, `keypoints_uv`) — display concern, kept off the broker; ONE perception rendered everywhere |
| `diagnostics` | engine → out | MQTT `diagnostics/heartbeat` | periodic heartbeat: node, mode, source liveness, fps, `LatencyStats` p50/p95/p99, calibration fact-check |
| `config` | engine → out | MQTT, `retain=True` at startup | node advertisement: zones/cameras/mode for late joiners |
| `detection_set` | **isistream → engine** | UDP loopback :9010 | Direction-1 ingest (below) |
| `fragment` | transport-level | UDP only | application-layer fragmentation (below) |

**MQTT topic layout** — `<base>/<version>/<node_id>/<suffix>`, e.g. `isiMonitor3D/v1/zone_a/track2d/person`. Operator sets the sink `prefix` to `isiMonitor3D/v1/<node_id>`; the gateway parses the version segment back out (legacy unversioned topics → `v0`).

## The inbound contract: `DetectionSetMessage` (points mode, :9010)

isistream publishes one per **camera per perception tick** to `ingestion.points.listen_port` (default **9010**, loopback, never on the outbound bus). Contract rules — each one load-bearing:

- `ts` = **`capture_ts` of the source frame** (the single KPI clock), never send time.
- One message per tick **even when `dets` is empty** — the explicit-empty heartbeat distinguishes "empty scene" from "dead producer" (silence ⇒ runtime degradation, as if the camera died).
- `seq` is per-camera monotonic — gaps = UDP loss, surfaced in diagnostics instead of silent.
- `config_fingerprint` detects producer/engine config drift (model, zones, calibration) — warn, never drop.
- `WireDetection.confidence` is the **raw** score — ByteTrack needs the low-confidence band; producers must not pre-threshold.

## Fragmentation (`FragmentMessage` + `FragmentBuffer`)

WSL2 `networkingMode=mirrored` silently drops **every loopback UDP datagram over ~1.5 KB** — observations with mask polygons never arrive. `UdpSink` therefore fragments at the **application layer**: JSON text sliced into chunks wrapped in `{fid, i, n, data}` envelopes; consumers reassemble with `FragmentBuffer` (incomplete groups pruned after 5 s, max 64 groups — UDP loss must not leak memory). MQTT (TCP) never fragments.

## isicomms gateway — REST out (`:8080`, Bearer token)

| Endpoint | Returns |
|---|---|
| `GET /nodes` | every node seen on the broker: mode, liveness, diagnostics |
| `GET /zones`, `GET /zones/{name}` | zone occupancy (from retained `zone_state`) |
| `GET /tracks` | recent tracks (2D/3D) |
| `GET /passings` | zone-crossing events, ts-sorted |
| `GET /diagnostics` | latest heartbeats |
| `GET /config` | node config advertisements |
| `GET /recent` | raw MQTT tail (feeds the probe) |
| `GET /healthz` | liveness (no token) |
| `GET /ui` | **live probe page** — nodes / zones / tracks / passings + raw MQTT tail, same glass design language as this site |
| `GET /docs` | Swagger |

## Launch — HOW

```bash
# full stack (broker :1883 + gateway :8080 + /ui) — the normal path
cd isicomms/deploy/onprem && docker compose up -d
# port collisions? host side only moves; containers stay 8080/1883:
ISICOMMS_GATEWAY_PORT=9090 ISICOMMS_MQTT_PORT=11883 docker compose up -d

# cloud stack (TLS, Caddy fronting — gateway not host-published):
cd isicomms/deploy/cloud && ./gen-certs.sh && docker compose up -d

# gateway alone, no docker (needs a reachable broker):
python -m isicomms                     # config via ISI_GATEWAY_* env (frozen prefix)
```

!!! note "Poll it"
    `curl -H "Authorization: Bearer $TOKEN" http://localhost:8080/zones` — that's the whole AGV/WMS integration surface. Any producer publishing versioned MQTT JSON feeds the gateway; consumers only ever poll.
