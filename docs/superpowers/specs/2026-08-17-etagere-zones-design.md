# Étagère zones — design (2026-08-17)

## Goal

Report the occupancy of a 3x3 bin rack ("étagère") as a per-cell matrix
(`r1c1 … r3c3` ∈ filled / empty / unknown), computed from the live camera
by the trained 2-class YOLO26-nano (`empty_box` / `filled_box`, 320 px, per-cell
crops), stabilised against transient occlusion, published on the UDP/JSON bus
and relayed to MQTT by isicomms. Grid definition is drawn once per zone in the
dashboard Settings (4 corners → auto-split → per-cell drag-adjust).

Non-goals (v1): multi-camera fusion of one rack, per-cell confidence smoothing
beyond hysteresis, whole-rack single-pass detection, dashboard-side inference.

## Where things live

| Concern | Owner | Notes |
|---|---|---|
| Grid + model config | `config/etagere.yaml` (new) | authored by dashboard, consumed by isistream |
| Cell inference | `isistream/etagere.py` (new) | Direction 1: producer owns pixels + detection |
| Wire message | `backbone/comms/schemas.py` `EtagereStateMessage` | additive, no SCHEMA_VERSION bump |
| Ingest routing | `backbone/ingestion/points_in.py` | second `isinstance` branch → `on_etagere` |
| Stabilisation + publish | `backbone/shared/etagere_state.py` (new) + `Publisher.publish_etagere_state` | hysteresis per cell |
| Sinks | `UdpSink` (dashboard) + `MqttSink` (`{prefix}/etagere/{zone_id}`, retained) | ABC default no-op |
| Relay / REST / UI card | `isicomms` | `NodeState.etagere_by_zone`, `GET /etagere`, /ui card |
| Settings UI + cam overlay + matrix widget | `monitor_web` | no detector import |

## 1. `config/etagere.yaml`

```yaml
model:
  onnx_path: trainer/isidet/runs/detect/models/yolo/yolo26n_e100_320px_17-08-2026_12-10-27/weights/best.onnx
  class_names: [empty_box, filled_box]
  imgsz: 320
  confidence_threshold: 0.3
  crop_margin: 0.08          # == grid_click.MARGIN used at training time
  max_fps: 2.0               # per-zone inference cap (rack changes on human timescales)
zones:
  - id: et_1
    name: "Étagère A"
    camera: cam_a
    frame_wh: [1920, 1080]   # coordinate space of corners/cells (source frame px)
    corners: [[u,v],[u,v],[u,v],[u,v]]   # outer quad TL,TR,BR,BL (auto-split source)
    cells:                                # exactly rows*cols rects, reading order
      - {r: 1, c: 1, rect: [x0, y0, x1, y1]}
      # ...
    rows: 3
    cols: 3
```

Rationale: étagère zones are per-camera *image-space* rectangles, not floor
polygons. Keeping them out of `zones.yaml`/`zone_patches` keeps them out of the
floor-projection pipeline (ZoneScopedDetector, floor-zone sync, twins, zone
advertising). The model block lives here, not under `backbone.yaml detection:`
(that is the pallet model). Loader + pydantic schema: `backbone/shared/etagere.py`
(`EtagereConfig`, `EtagereZone`, `EtagereCell`, `load_etagere_config(path)`),
shared by isistream, backbone, monitor_web (all may import `backbone.shared`).
Path: `backbone.yaml` `etagere.config_path` (default `config/etagere.yaml`
beside backbone.yaml); missing file ⇒ feature off, no error.

## 2. isistream — `EtagereDetector`

- Built in `build_isistream_core()` when the config has ≥1 zone. Own detector via
  `detector_registry.create("yolo_onnx", onnx_path, class_names,
  confidence_threshold, input_size=(imgsz, imgsz), providers)`. Relies on the
  end-to-end (NMS-free) detect decode added in f89ca35.
- `tick(frames_by_cam) -> list[EtagereStateMessage]`: for each zone whose
  camera has a fresh frame and whose `max_fps` interval elapsed: scale rects from
  `frame_wh` to the actual frame, crop `rect + margin` (clip), key crops
  `f"{zone_id}:{r}:{c}"`, batch ALL due zones' crops into ONE `FramePair` →
  one `detect()` call (letterbox to 320 inside the plugin, dynamic batch).
- Per cell: top-confidence det ≥ threshold ⇒ `filled` (cls filled_box) /
  `empty` (cls empty_box); none ⇒ `unknown`. Emit one message per zone with
  `seq`, `producer_id`, `config_fingerprint`, sent with `send_json_datagram`
  to the same ingest port as detection sets. Stage timing recorded as
  `stage_ms["etagere"]`.
- Independent of floor zones (runs even when `zones.yaml` is empty).

## 3. Wire contract

```python
class MessageType(str, Enum): ... etagere_state = "etagere_state"

class EtagereCellState(BaseModel):   # frozen, extra=forbid
    r: int; c: int
    state: Literal["filled", "empty", "unknown"]
    confidence: float = 0.0

class EtagereStateMessage(BaseModel):
    schema_version, type = etagere_state, ts: float
    camera_id: str; zone_id: str; name: str = ""
    rows: int = 3; cols: int = 3
    cells: list[EtagereCellState]
    seq: int = 0; producer_id: str = ""; config_fingerprint: str = ""
    stabilized: bool = False      # False = raw producer tick, True = backbone output
```
`parse_envelope` learns the type. Additive + defaulted ⇒ no version bump
(documented in the SCHEMA_VERSION docstring per convention).

## 4. Backbone

- `points_in.py`: `elif isinstance(msg, EtagereStateMessage): self._on_etagere(msg)`
  (callback optional; default drop-count). Camera filter applies as for sets.
- `backbone/shared/etagere_state.py`: `EtagereStateTracker` — per
  `(zone_id, r, c)` an `OccupancyStabilizer`-style vote (window 15, flip ratio
  0.7; `unknown` does not vote, but a cell with no vote for `unknown_after_s`
  = 5 s decays to `unknown`). `update(msg) -> EtagereStateMessage | None`
  returns the stabilised message when any cell state changed or every
  `heartbeat_s` = 5 s (retained + heartbeat, same hygiene as zone_state).
- Orchestrator wires `points_in.on_etagere → tracker.update → publisher.publish_etagere_state`.
- `MetadataSink.publish_etagere_state(msg)`: non-abstract no-op default.
  `UdpSink` implements (JSON datagram, dashboard consumes). `MqttSink`
  implements: topic `{prefix}/etagere/{zone_id}`, `retain=True`,
  qos = `zone_state_qos`.

## 5. isicomms

Rebuild against the schema. Generic passthrough already surfaces the type in
the /ui schema tree. Add `NodeState.etagere_by_zone: dict[str, EtagereStateMessage]`,
the `isinstance` branch, `GET /etagere` + `GET /etagere/{zone_id}` (per node),
and a /ui card drawing the 3x3 matrix (filled=green, empty=grey, unknown=hatched)
with staleness. Docker: `compose -p on-prem build gateway` (rebuild, not restart).

## 6. Dashboard (monitor_web)

- Settings → **Étagère** section: camera picker; "Draw" → `startDraw({minPoints:4,
  maxPoints:4})` on the cam image → 4 corners → auto-split (bilinear thirds) into
  9 rects; overlay editor: drag a cell to move, drag its corner handle to resize;
  Save → `POST /api/etagere` (atomic write of `etagere.yaml`) → isistream
  hot-restart via the existing config-save path. `GET /api/etagere` returns config.
- CAM view overlay: cell rects coloured by latest wire state (`etagere_state`
  ingested by the dashboard's UDP bus like observations); COMMUNICATION card:
  3x3 matrix widget per zone.
- No detector import; zone workers untouched.

## 7. Testing

- Unit: schema round-trip + parse_envelope; `EtagereDetector` decision on
  synthetic detections (filled/empty/unknown, margin+scale mapping); tracker
  hysteresis (flip needs ≥70 % window; unknown decay; heartbeat); points_in
  routing; MQTT topic/retain via existing sink test harness; config loader.
- Hermetic e2e: orchestrator-style — feed `EtagereStateMessage`s into
  `points_in`, assert stabilised messages on a loopback UDP socket.
- Live: draw grid on cam_a, `mosquitto_sub -t 'isiMonitor3D/v1/+/etagere/#'`.
