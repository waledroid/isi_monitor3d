# isiMonitor3D — AGV System Test: Minimal Integration Guide

Operational companion to the full RFC (`docs/rfc.md`), reduced to what the joint
pick-and-place test with the AGV team needs. isiMonitor3D publishes the live
state of marked floor zones — palette presence, load state (empty/full) and load
type (carton/polybag); the AGV system consumes that signal and launches its
pick-and-place.

**Interactive console:** open `http://<SERVER_IP>:8080/test` in any browser on
the LAN (it opens the isicomms UI — the test cards run automatically) — one
card per state below, each showing the live answer plus the exact REST URL and
MQTT topic to use.

## 1. What you receive

One JSON message per zone, **retained** on the broker: the current state arrives
immediately on connect, then on every change (~1 s refresh while occupied). An
empty zone publishes an explicit `"objects": []` — *empty* never means *offline*.

| Field | Meaning |
|---|---|
| `zone` | Zone name (e.g. `Sortie_1`) — **filter on this field**, not on the topic (the topic segment is an internal zone id). |
| `objects[].cls` | Detected class: `palette`, `person`, `carton`, … |
| `objects[].confidence` | Detection confidence 0–1. |
| `objects[].xy_m` | Position in the zone, meters (informative here). |
| `objects[].occupancy_state` | Palette only: `"empty"` / `"full"`. |
| `objects[].occupancy_content` | Palette only: `"carton"` / `"polybag"`. |
| `count` | Number of objects in the zone. |

Real payload captured from the running system:

```json
{"schema_version": 6, "type": "zone_state", "ts": 1785156682.85,
 "zone": "Sortie_1", "zone_id": "zp_mr8z7cot",
 "objects": [{"track_id": 1220, "cls": "palette", "confidence": 0.24,
              "xy_m": [-1.00, -0.12],
              "occupancy_state": "full", "occupancy_content": "carton",
              "occupancy_confidence": 0.12}],
 "count": 1}
```

## 2. How to connect (pick one)

**Option A — MQTT (recommended, event-driven).** Broker `<SERVER_IP>:1883`,
plain TCP, no authentication (trusted-LAN test profile). Subscribe
`isiMonitor3D/v1/+/zone/+` at QoS 1, filter on `payload.zone`.
Quick check: `mosquitto_sub -h <SERVER_IP> -t 'isiMonitor3D/v1/+/zone/+' -v`

**Option B — REST (poll-based).** `GET http://<SERVER_IP>:8080/v1/zones` (all
zones, enriched with the latest state). Poll at 1–2 Hz. Same JSON fields.
Quick check: `curl http://<SERVER_IP>:8080/v1/zones`
Send an `X-Client-Name: agv_07` header with your requests — your client then
appears by name in the operator UI's Consumers panel (instead of by IP).

## 3. Reference client (tested against the live system)

`isicomms/examples/agv_min_client.py` — Python 3, one dependency
(`pip install paho-mqtt`). Set `BROKER` and `ZONE`, run, and it prints one line
per state change:

```
connected: Success
[Sortie_1] PALETTE present  conf=0.24  state=full  content=carton  pos=(-1.00, -0.12) m
```

Suggested decision rules (thresholds agreed on test day): pick when a `palette`
is present; use `occupancy_state` / `occupancy_content` to choose the mission;
hold while a `person` is in the zone; zone confirmed clear when `objects` is
empty again after the pick.

## 4. Network requirements

- AGV client and isiMonitor3D server on the same LAN (or routed, no NAT/proxy).
- Reachable from the AGV side: `<SERVER_IP>` **TCP 1883** (MQTT) and **TCP 8080**
  (REST). No inbound ports needed on AGV equipment.
- Plaintext, no credentials for this test (the secured profile is out of scope).
- Fixed IP or DHCP reservation for the server — `<SERVER_IP>` is communicated
  before the test.
- No clock synchronization required — react to message arrival; `ts` is
  informative.
- AGV on Wi-Fi is fine: MQTT keepalive + client auto-reconnect handle roaming
  drops; the retained message restores current state instantly on reconnect.

## 5. Pre-test checklist (isiMonitor3D side)

- Backbone running, cameras alive (`GET /v1/nodes` → `alive`, or the `/test`
  console's Node card).
- Zones drawn over the physically marked floor areas; names agreed with the AGV
  team (`Sortie_1`, `Sortie_2`, …).
- Stale retained topics cleared after any zone renaming
  (`mosquitto_pub -r -n -t '<old topic>'` per leftover).
- Broker + gateway up (see `isicomms/deploy/onprem/`).
- Detection confidence sanity-checked under the day's lighting; thresholds
  agreed with the AGV team.

## 6. Test sequence & success criteria

| # | Step | Expected at the AGV client |
|---|---|---|
| 1 | Connect client | Retained zone state received < 2 s |
| 2 | Zone empty | `empty - nothing to pick` |
| 3 | Place an empty palette in the zone | `PALETTE present, state=empty` within ~2 s |
| 4 | Load the palette with cartons | `state=full, content=carton` |
| 5 | Person enters the zone | `HOLD - person in zone` |
| 6 | Person leaves; AGV picks the palette | back to `PALETTE present`, then `empty` after pick |

**Success:** every transition visible at the AGV client within the agreed
latency (target < 2 s), no pick trigger while a person is in the zone, and one
complete pick-and-place cycle driven solely by this signal.
