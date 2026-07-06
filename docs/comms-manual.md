# Communication (MQTT) — How-To Manual

A direct, step-by-step guide for **using** the comms system. Two audiences:

- **Part A — Operator**: run and manage the central server + the warehouse PCs.
- **Part B — AGV / WMS integrator**: consume the data over the REST API.

For *why* it's built this way, see `docs/rfc.md` and `docs/mqtt-architecture.md`.
Throughout, the example central server is **`192.168.2.39`** — replace it with
your server's IP.

---

# Part A — Operator

## A1. Start the central server (once)
```bash
cd /home/aatanda/isi_monitor3d
docker compose -p on-prem -f deploy/onprem/docker-compose.yml up -d
```
▶ **Expected:** two containers start — `on-prem-mosquitto-1` (broker, port 1883)
and `on-prem-gateway-1` (REST API, port 8080). Both auto-restart on reboot.

## A2. Verify the server is up
```bash
docker compose -p on-prem -f deploy/onprem/docker-compose.yml ps
curl http://192.168.2.39:8080/healthz
```
▶ **Expected:** both containers `Up`; `curl` returns `{"ok":true}`.

## A3. Connect a warehouse PC (a "node")
On each PC, edit its `config/backbone.yaml` — set a **unique `node_id`** and point
its MQTT sink at the server:
```yaml
node_id: zone_a                     # MUST be unique per PC (zone_a, dock_1, …)
metadata:
  area: "Zone A — racking"
  diagnostics: { enabled: true, interval_sec: 5.0 }
  sinks:
    - plugin: mqtt
      host: 192.168.2.39            # the central server
      port: 1883
      prefix: isiMonitor3D/v1/zone_a         # = isiMonitor3D/v1/<node_id>   (note the v1)
```
Start the Backbone:
```bash
conda activate monitor3d
python -m backbone.runtime --config config/backbone.yaml
```
▶ **Expected:** the PC begins publishing immediately — first a retained `config`,
then a `diagnostics` heartbeat every ~5 s, then `track2d` once it detects objects.

## A4. Confirm the node is reporting
```bash
curl http://192.168.2.39:8080/v1/nodes
```
▶ **Expected:** the node appears, e.g.:
```json
{"nodes":[{"node_id":"zone_a","area":"Zone A","status":"alive",
           "topic_version":"v1","mode":"single_cam_homography",
           "cameras":["cam_a"],"fps":18.0,"latency_ms":{...}}],"count":1}
```
`status:"alive"` = heartbeating. Add more PCs (each with its own `node_id`) and
they appear here automatically — **no server change needed**.

## A5. Watch raw traffic (debug)
```bash
mosquitto_sub -h 192.168.2.39 -t 'isiMonitor3D/#' -v
```
▶ **Expected:** lines like `isiMonitor3D/v1/zone_a/config {...}`,
`isiMonitor3D/v1/zone_a/diagnostics/heartbeat {...}`, `isiMonitor3D/v1/zone_a/track2d/person {...}`.

## A6. Day-to-day management
```bash
# logs
docker compose -p on-prem -f deploy/onprem/docker-compose.yml logs -f gateway
# stop / start / restart
docker compose -p on-prem -f deploy/onprem/docker-compose.yml stop
docker compose -p on-prem -f deploy/onprem/docker-compose.yml start
docker compose -p on-prem -f deploy/onprem/docker-compose.yml restart gateway
# tear down completely
docker compose -p on-prem -f deploy/onprem/docker-compose.yml down
```

## A7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `/v1/nodes` is empty | no node publishing, or wrong `host`/`prefix` | check the PC's mqtt sink `host` = server IP, `prefix: isiMonitor3D/v1/<node_id>`; check `mosquitto_sub -t 'isiMonitor3D/#'` shows traffic |
| node shows `status:"stale"` | the PC stopped heart-beating (process down / network) | restart the Backbone on that PC; check LAN connectivity to :1883 |
| node shows `topic_version:"v0"` | the PC's `prefix` is missing the `v1` segment | set `prefix: isiMonitor3D/v1/<node_id>` (it still works as `v0`, but use v1) |
| `curl` to :8080 refused | gateway container down | `docker compose -p on-prem … ps`; `… logs gateway`; `… up -d` |
| port 1883 "address in use" | a system mosquitto is squatting | `sudo systemctl stop mosquitto && sudo systemctl disable mosquitto`, then `… up -d` |

---

# Part B — AGV / WMS integrator

You consume **HTTP only** — no MQTT client. Poll the gateway's REST API.

## B1. Base URL & endpoints
Base: `http://192.168.2.39:8080` — all data endpoints are under **`/v1`**:

| Method | Endpoint | Use |
|---|---|---|
| GET | `/v1/tracks` | every tracked object, all PCs (the main one for AGVs) |
| GET | `/v1/nodes` | which PCs are online + healthy |
| GET | `/v1/zones` | the global warehouse zone map |
| GET | `/v1/passings` | recent zone enter/leave events |
| GET | `/v1/diagnostics` | per-PC health detail |
| GET | `/healthz` | server alive (no `/v1`) |

> The bare paths (`/tracks`, `/nodes`, …) also work as back-compat aliases, but
> **use `/v1/…`** — it's the stable, versioned contract.

## B2. Get all tracks
```bash
curl http://192.168.2.39:8080/v1/tracks
```
▶ **Expected:**
```json
{"tracks":[
   {"type":"track_2d","node_id":"zone_a","track_id":7,"cls":"person",
    "ts":1782460092.23,
    "xy_m":[3.42,1.87],          // metric floor position (metres)
    "vxy_m":[0.10,-0.55],        // metric velocity (m/s)
    "confidence":0.91,
    "cameras_seeing":["cam_a"]}
 ],"count":1}
```

**Field meanings (what an AGV cares about):**
| Field | Meaning |
|---|---|
| `node_id` | which warehouse PC/zone reported it (identity is `(node_id, track_id)`) |
| `track_id` | stable id **within that node** (do not assume unique across nodes) |
| `cls` | object class — `person`, `forklift`, `pallet`, … |
| `xy_m` | **position on the floor in metres** — what you navigate against |
| `vxy_m` | velocity in m/s (heading + speed) |
| `confidence` | 0..1 |
| `ts` | capture time (Unix seconds). `now - ts` ≈ data age/latency |

## B3. Filter the query
```bash
# only one zone's PC
curl "http://192.168.2.39:8080/v1/tracks?node=zone_a"
# only people
curl "http://192.168.2.39:8080/v1/tracks?cls=person"
# only tracks currently INSIDE a named zone (point-in-polygon, all PCs)
curl "http://192.168.2.39:8080/v1/tracks?zone=dock_door"
# combine
curl "http://192.168.2.39:8080/v1/tracks?node=zone_a&cls=person&zone=dock_door"
```
▶ **Expected:** the same shape as B2, narrowed to matches.

## B4. The AGV polling loop (Python)
```python
import requests, time

GATEWAY = "http://192.168.2.39:8080"
POLL_HZ = 5                                   # poll 5×/second

while True:
    try:
        r = requests.get(f"{GATEWAY}/v1/tracks", params={"cls": "person"}, timeout=1.0)
        r.raise_for_status()
        for t in r.json()["tracks"]:
            x, y = t["xy_m"]                  # metres on the floor
            vx, vy = t["vxy_m"]               # m/s
            # → feed (x, y, vx, vy) into your obstacle map / avoidance
            print(t["node_id"], t["track_id"], t["cls"], (round(x,2), round(y,2)))
    except requests.RequestException:
        pass                                  # gateway momentarily unreachable; retry next tick
    time.sleep(1.0 / POLL_HZ)
```
▶ **Expected:** a steady stream of metric positions to steer around. The gateway
is stateless-to-poll — call it as often as you need (it serves the latest cached
state instantly).

## B5. Check which PCs are live before trusting their data
```bash
curl http://192.168.2.39:8080/v1/nodes
```
▶ **Expected:** each node with `status` `"alive"` or `"stale"`. **Ignore tracks
from a `stale` node** — it stopped reporting (default >15 s silent). Example guard:
```python
nodes = requests.get(f"{GATEWAY}/v1/nodes", timeout=1).json()["nodes"]
alive = {n["node_id"] for n in nodes if n["status"] == "alive"}
# then: use only tracks whose t["node_id"] in alive
```

## B6. Get the zone map (once, or on change)
```bash
curl http://192.168.2.39:8080/v1/zones
```
▶ **Expected:**
```json
{"zones":[{"node_id":"zone_a","area":"Zone A","name":"dock_door",
           "kind":"danger","type":"...","severity":"...",
           "polygon":[[x1,y1],[x2,y2],...]}],"count":N}
```
Polygons are in **metres**, same frame as `xy_m`. Use these to know where the
danger/no-go areas are.

## B7. Recent zone crossings (events)
```bash
curl "http://192.168.2.39:8080/v1/passings?limit=20"
curl "http://192.168.2.39:8080/v1/passings?node=zone_a"
```
▶ **Expected:** `{"passings":[{"node_id","ts","track_id","cls","zone","direction":"enter"|"leave"},...]}`.

## B8. If the server has auth enabled
When the operator sets an API token (cloud deployments), every call except
`/healthz` needs a bearer header:
```bash
curl -H "Authorization: Bearer <TOKEN>" http://192.168.2.39:8080/v1/tracks
```
```python
HEADERS = {"Authorization": "Bearer <TOKEN>"}
requests.get(f"{GATEWAY}/v1/tracks", headers=HEADERS, timeout=1)
```
▶ **Expected:** without the header you get `401`; with it, normal data.

---

## Quick reference

**Operator**
```bash
docker compose -p on-prem -f deploy/onprem/docker-compose.yml up -d     # start
curl http://<server>:8080/v1/nodes                                       # who's online
mosquitto_sub -h <server> -t 'isiMonitor3D/#' -v                                  # raw traffic
docker compose -p on-prem -f deploy/onprem/docker-compose.yml down       # stop
```
Node config: `node_id: <unique>` · mqtt sink `host: <server>`, `prefix: isiMonitor3D/v1/<node_id>`.

**AGV / WMS**
```bash
GET http://<server>:8080/v1/tracks            # all tracks (xy_m in metres)
GET http://<server>:8080/v1/tracks?zone=X     # only inside zone X
GET http://<server>:8080/v1/nodes             # health (use alive only)
GET http://<server>:8080/v1/zones             # zone polygons
```
Identity = `(node_id, track_id)` · positions/velocities in metres / m/s · poll as
often as needed.
