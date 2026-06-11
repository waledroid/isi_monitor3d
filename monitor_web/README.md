# monitor_web — ISI Monitor 3D operator dashboard

FastAPI + Jinja2 + HTMX + Material Web Components. Consumes the **Backbone**
(separate process) via UDP/JSON, displays a 2D digital-twin map, per-camera
live feeds, logs, and status. Supervises the Backbone subprocess via
START/STOP buttons.

## Quick start

```bash
conda activate monitor3d
cd monitor_web
pip install -e ".[dev]"
python -m monitor_web                       # uvicorn on :8000
# open http://localhost:8000/
```

## Architecture

`monitor_web` is a **separate Python process** from the Backbone. Per the
architecture's "process boundaries are contractual" rule, it imports only
consumer-side helpers from `isi-monitor3d-backbone`:

- `backbone.metadata.schemas` — typed decode of UDP `Track2D` / `Track3D`
  envelopes.
- `backbone.shared.zones.ZoneRegistry` — reads `zones.yaml` to draw zone
  polygons on the floor map.
- `backbone.ingestion.RtspFrameSource` — re-stream RTSP as MJPEG for the
  browser.

It does **NOT** import `backbone.runtime.Orchestrator`. START spawns the
orchestrator as a subprocess (`python -m backbone.runtime.orchestrator
--config <yaml>`); STOP sends SIGTERM.

## Endpoints

- `GET /` — the dashboard page.
- `GET /api/status` — mode, sources, latency, freshness.
- `GET /api/zones` — world-space zone polygons (from `zones.yaml`).
- `GET /api/logs` — HTMX partial of the latest log lines.
- `POST /api/control/start` / `stop` — supervise the Backbone subprocess.
- `GET /stream/video/{camera_id}` — multipart MJPEG.
- `WS /ws/tracks` — pushes `Track2D` + `Track3D` envelopes to the browser.
