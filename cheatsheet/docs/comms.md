# Wire, MQTT & the probe — schema v6

**WHY** — everything talks through versioned JSON; `backbone/comms/schemas.py` is the single contract.

**WHAT** — `SCHEMA_VERSION = 6` (accepts 3–6), `TOPIC_VERSION = "v1"`. Pydantic, `extra="forbid"`, frozen.

## Message types

All engine → out unless noted.

| Type | Transport | Purpose |
|---|---|---|
| `track_2d` | UDP + MQTT | `track_id`, `cls`, `xy_m`, `vxy_m`, `occupancy_*` |
| `track_3d` | UDP + MQTT | `xyz_m`, `max_reprojection_error_px` — same `track_id` as 2D |
| `zone_state` | + MQTT retained `{prefix}/zone/{zone}` | zone contents (WMS signal); empty = explicit `objects=()` |
| `passing` | UDP + MQTT | zone `enter`/`leave`; key on `zone_id`, not `zone` |
| `image_ref` | UDP + MQTT | snapshot URL — never raw bytes |
| `proximity` | + MQTT retained | person↔object floor distances |
| `observations` | **UDP only** | per-camera raw dets (`mask_poly`, `keypoints_uv`) — display |
| `diagnostics` | MQTT `diagnostics/heartbeat` | mode, liveness, fps, p50/p95/p99 |
| `config` | MQTT `retain=True` | zones/cameras/mode for late joiners |
| `detection_set` | **isistream → engine**, UDP :9010 | Direction-1 ingest (below) |
| `fragment` | UDP only | app-layer fragmentation (below) |

**Topics** — `<base>/<version>/<node_id>/<suffix>`, e.g. `isiMonitor3D/v1/zone_a/track2d/person`.

## Inbound: `DetectionSetMessage` (:9010)

One per camera per tick, loopback only:

- `ts` = source-frame `capture_ts`, never send time.
- Empty `dets` still sent — heartbeat vs dead producer.
- `seq` monotonic — gaps = UDP loss, surfaced.
- `config_fingerprint` — drift warning, never drop.
- `confidence` raw — ByteTrack needs the low band.

## Fragmentation

WSL2 mirrored mode drops loopback UDP > ~1.5 KB. `UdpSink` slices JSON into `FragmentMessage {fid, i, n, data}`; `FragmentBuffer` reassembles (5 s prune, 64 groups). MQTT never fragments.

## Gateway REST (`:8080`, Bearer token)

| Endpoint | Returns |
|---|---|
| `GET /nodes` | nodes: mode, liveness, diagnostics |
| `GET /zones`, `GET /zones/{name}` | zone occupancy |
| `GET /tracks` | recent 2D/3D tracks |
| `GET /passings` | zone-crossing events |
| `GET /diagnostics` | latest heartbeats |
| `GET /config` | node config ads |
| `GET /recent` | raw MQTT tail |
| `GET /healthz` | liveness (no token) |
| `GET /ui` | live probe page |
| `GET /docs` | Swagger |

!!! note "Poll it"
    `curl -H "Authorization: Bearer $TOKEN" http://localhost:8080/zones` — the whole AGV/WMS integration surface.
