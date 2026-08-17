# Module reuse — detachable, copy-portable apps

Every module app exports to a **self-contained folder** you copy into another
project and launch. One command produces it:

```bash
scripts/export_module.sh <isical|isistream|isigen|isidet|isicomms> <dest-dir> [onprem|cloud]
```

The shared core (`backbone* + calibration* + isistream*`) travels as a
**wheel** built fresh from this checkout into `<export>/wheels/` — no vendored
source copies (nothing to drift), no git access needed at the destination.
The single source of truth stays in `backbone/`.

## Detachability map

| Module | In-repo coupling | Export contents | Launch at destination |
|---|---|---|---|
| **isiGen** (`trainer/isiGen/`) | none — fully standalone | plain folder copy (in-folder `requirements.txt` + `launch.sh` included) | `./launch.sh` → Studio :8200 (`ISIGEN_*` env) |
| **isidet** (`trainer/isidet/`) | none — fully standalone | plain folder copy (in-folder `isi-train.yml`) | `conda env create -f isi-train.yml` → train scripts |
| **isical** (`isical/` + `calibration/`) | `calibration/` + `backbone.shared.geometry` + lazy `backbone.ingestion.{rtsp,v4l2}` — all inside the wheel | isical source + backbone wheel + `setup_multical.sh` + launcher | `./launch.sh` → Studio :8300 (`ISICAL_*` env for paths); `./setup_multical.sh` once for Multical extrinsics |
| **isistream** (`isistream/`) | a backbone application by design — its code ships **inside** the wheel | launcher + `config.example.yaml` + wheel (no source copy — it would shadow the package) | install system GStreamer → `./launch.sh --config config.yaml` |
| **isicomms** (`isicomms/`) | gateway imports only the light backbone surface (schemas + zones: numpy/pyyaml/pydantic) | compose stack (onprem or cloud) + gateway & backbone wheels + `Dockerfile.portable` (builds offline from the wheels) | onprem: `docker compose up -d`; cloud: `./gen-certs.sh` → `.env` → `up -d` |

## The two contracts (reuse without sharing code)

- **Backbone wire (UDP/JSON)** — `backbone/comms/schemas.py` is the only
  contract between the metric engine and its consumers. isistream publishes
  `DetectionSetMessage`s to the engine; the engine publishes tracks/zone
  state/observations on the bus.

  | Message | Topic (MQTT) | Retained | Payload summary |
  |---|---|---|---|
  | `zone_state` | `{prefix}/zone/<zone>` | yes | a floor zone's current occupants/count |
  | `etagere_state` | `{prefix}/etagere/{zone_id}` | yes | one bin-rack zone's stabilised 3x3 (or rows*cols) cell grid — `cells[].state` in `filled`/`empty`/`unknown` + `confidence`, `stabilized: true` |

- **isicomms (MQTT in, REST out)** — a live probe UI ships at
  ``http://<host>:8080/ui`` (nodes / zones / tracks / passings + a raw MQTT
  tail; Swagger at ``/docs``) — any producer publishing the versioned
  MQTT JSON messages (SCHEMA_VERSION 6: zone_state, tracks, passings,
  diagnostics, config) feeds the gateway; consumers poll `GET /nodes`,
  `/zones`, … with a Bearer token. Ports, paths, and the `ISI_GATEWAY_*` env
  prefix are frozen interface. Evolution = expand `schemas.py`, never share
  code (architecture principle #5).

## Notes

- Wheels are built at export time and never committed to git. Re-run the
  exporter to refresh a deployed copy with current code.
- isistream **cannot** be reused as a bare folder copy of `isistream/` — its
  imports live in the backbone package. The exported folder (wheel + launcher)
  is the honest unit.
- isicomms keeps working inside isiMonitor3d exactly as before the rename:
  same REST API, same port, same MQTT topics, same env prefix.
