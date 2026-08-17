# Étagère Zones Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a per-cell 3x3 occupancy matrix (filled/empty/unknown) for each configured shelf rack, computed by isistream from per-cell 320-px crops through the trained 2-class YOLO26-nano, stabilised by the Backbone, published on UDP + MQTT (retained), relayed by isicomms, and drawn/edited in the dashboard.

**Architecture:** A new `config/etagere.yaml` (dashboard-authored, isistream-consumed) holds the model block + per-camera image-space cell rects. `isistream/etagere.py` crops the cells, batches them through its own `yolo_onnx` session (input 320, end-to-end head decode from f89ca35) and emits raw `EtagereStateMessage`s to the Backbone's points ingest port. The Backbone routes them (`points_in.py` second branch) through a per-cell hysteresis tracker and fans out via `Publisher.publish_etagere_state` (UDP sink for the dashboard, MQTT sink retained on `{prefix}/etagere/{zone_id}`). isicomms caches per node and exposes REST + a /ui matrix card. monitor_web adds a Settings section (4 corners → auto-split → drag-adjust) and cam-overlay/matrix rendering. No dashboard inference.

**Tech Stack:** Python 3.10, pydantic 2, numpy/OpenCV, onnxruntime (`yolo_onnx` plugin), paho-mqtt (via `MqttSink`), FastAPI + Jinja2 + vanilla JS (monitor_web, isicomms), pytest.

**Spec:** `docs/superpowers/specs/2026-08-17-etagere-zones-design.md`

## Global Constraints

- Python target 3.10 — no 3.12+ syntax (no `type` alias keyword). Runtime env: `monitor3d` (`/home/aatanda/miniforge3/envs/monitor3d/bin/python`); run backbone tests with `pytest`, monitor_web tests with `cd monitor_web && pytest`, isicomms tests with `cd isicomms && pytest`.
- Lint: `ruff check backbone calibration tests` (+ the touched isistream/monitor_web/isicomms files) must pass before every commit.
- Process boundaries: Backbone imports nothing from modules; monitor_web imports only `backbone.comms.schemas`, `backbone.shared.*`, `backbone.ingestion`; isicomms imports only `backbone.comms.schemas`. **isistream owns all inference**; monitor_web runs no detector for étagère.
- Wire changes are ADDITIVE and DEFAULTED → no `SCHEMA_VERSION` bump; document the addition in the `SCHEMA_VERSION` docstring.
- Exactly five plugin ABCs — `MetadataSink.publish_etagere_state` is a NON-abstract default no-op (`tests/test_registry.py::test_five_seams_present` must stay green).
- Étagère config file default: `config/etagere.yaml` beside `backbone.yaml`; missing file ⇒ feature off, never an error.
- Cell crop = rect + `crop_margin` (0.08 of w/h) then letterbox 320 — identical to training (`trainer/isidet/scripts/grid_click.py` MARGIN/letterbox).
- Model artifact: `trainer/isidet/runs/detect/models/yolo/yolo26n_e100_320px_17-08-2026_12-10-27/weights/best.onnx`, classes `[empty_box, filled_box]`, imgsz 320.
- GPU: never run a CUDA process beside a live stack for tests; unit tests use fakes (no ONNX session).
- Commit after every task with the trailer:
  ```
  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01XmWWLMt9ZtLKgh5ZbAkTYr
  ```

---

## File structure

| File | Responsibility |
|---|---|
| `backbone/comms/schemas.py` (modify) | `MessageType.ETAGERE_STATE`, `EtagereCellState`, `EtagereStateMessage`, `parse_envelope` branch |
| `backbone/shared/etagere.py` (create) | pydantic config models + `load_etagere_config(path)` + `cells_from_corners(corners, rows, cols)` |
| `backbone/shared/etagere_state.py` (create) | `EtagereStateTracker` — per-cell hysteresis, unknown decay, heartbeat |
| `backbone/core/interfaces.py` (modify) | `MetadataSink.publish_etagere_state` default no-op |
| `backbone/comms/publisher.py` (modify) | `Publisher.publish_etagere_state` fan-out |
| `backbone/comms/udp_sink.py`, `backbone/comms/mqtt_sink.py` (modify) | sink implementations (`etagere_topic`, retained) |
| `backbone/ingestion/points_in.py` (modify) | `on_etagere` callback + `EtagereStateMessage` branch |
| `backbone/runtime/orchestrator.py` (modify) | wire ingest → tracker → publisher; `etagere.config_path` |
| `isistream/etagere.py` (create) | `EtagereDetector` (crop/batch/decide/emit) |
| `isistream/core.py` (modify) | build + call per tick, `stage_ms["etagere"]` |
| `isicomms/isicomms/mqtt_subscriber.py`, `api/routes_etagere.py` (create), `app.py`, `api/ui_page.py` (modify) | cache, REST, /ui card |
| `monitor_web/monitor_web/bus_subscriber.py` (modify), `api/routes_etagere.py` (create), `app.py` (modify) | wire cache + GET/POST `/api/etagere` |
| `monitor_web/monitor_web/static/js/etagere.js` (create), `live_overlay.js`, `comms_nodes.js`, `templates/dashboard.html` (modify) | Settings editor, cam overlay, matrix widget |
| `config/etagere.yaml` (create) | example config (1 zone, cam_a) |
| tests: `tests/test_metadata_schemas.py`, `tests/test_etagere_config.py`, `tests/test_etagere_state.py`, `tests/test_publisher.py`, `tests/test_udp_sink.py`, `tests/test_mqtt_sink.py`, `tests/test_points_in_etagere.py`, `tests/test_isistream_etagere.py`, `isicomms/tests/test_routes_etagere.py`, `monitor_web/tests/test_routes_etagere.py` | per task |

---

### Task 1: Wire schema — `EtagereStateMessage`

**Files:**
- Modify: `backbone/comms/schemas.py` (`MessageType` at ~96, models after `DetectionSetMessage` ~480, `parse_envelope` ~637-691, `SCHEMA_VERSION` docstring ~56)
- Test: `tests/test_metadata_schemas.py`

**Interfaces:**
- Produces:
  ```python
  class MessageType(str, Enum): ETAGERE_STATE = "etagere_state"
  class EtagereCellState(BaseModel):  # ConfigDict(extra="forbid", frozen=True)
      r: int = Field(..., ge=1); c: int = Field(..., ge=1)
      state: Literal["filled", "empty", "unknown"]
      confidence: float = Field(0.0, ge=0.0, le=1.0)
  class EtagereStateMessage(BaseModel):  # ConfigDict(extra="forbid", frozen=True)
      schema_version: int = Field(default=SCHEMA_VERSION, ge=1)
      type: Literal[MessageType.ETAGERE_STATE] = MessageType.ETAGERE_STATE
      ts: float
      camera_id: str
      zone_id: str
      name: str = ""
      rows: int = Field(3, ge=1); cols: int = Field(3, ge=1)
      cells: tuple[EtagereCellState, ...]
      seq: int = Field(0, ge=0)
      producer_id: str = ""
      config_fingerprint: str | None = None
      stabilized: bool = False
  ```
  `parse_envelope(dict)` returns `EtagereStateMessage` for `type == "etagere_state"`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_metadata_schemas.py`)

```python
def test_etagere_state_round_trip() -> None:
    from backbone.comms.schemas import (
        EtagereCellState, EtagereStateMessage, MessageType, parse_envelope,
    )
    cells = tuple(
        EtagereCellState(r=r, c=c, state="filled" if (r + c) % 2 else "empty",
                         confidence=0.9)
        for r in (1, 2, 3) for c in (1, 2, 3)
    )
    msg = EtagereStateMessage(ts=1.5, camera_id="cam_a", zone_id="et_1",
                              name="Étagère A", cells=cells, seq=4,
                              producer_id="isistream")
    data = json.loads(msg.model_dump_json())
    assert data["type"] == "etagere_state"
    assert data["rows"] == 3 and data["cols"] == 3
    assert len(data["cells"]) == 9
    back = parse_envelope(data)
    assert isinstance(back, EtagereStateMessage)
    assert back == msg
    assert MessageType.ETAGERE_STATE.value == "etagere_state"


def test_etagere_state_rejects_bad_state_and_extra() -> None:
    from backbone.comms.schemas import EtagereCellState, EtagereStateMessage
    with pytest.raises(ValidationError):
        EtagereCellState(r=1, c=1, state="half")
    with pytest.raises(ValidationError):
        EtagereStateMessage(ts=0.0, camera_id="cam_a", zone_id="z", cells=(),
                            bogus=1)


def test_etagere_state_defaults() -> None:
    from backbone.comms.schemas import EtagereStateMessage
    msg = EtagereStateMessage(ts=0.0, camera_id="cam_a", zone_id="z", cells=())
    assert msg.stabilized is False and msg.seq == 0 and msg.name == ""
    assert msg.schema_version == SCHEMA_VERSION
```
Ensure the file already imports `json`, `pytest`, `ValidationError` (from pydantic) and `SCHEMA_VERSION`; add any missing import at the top.

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_metadata_schemas.py -k etagere -v`
Expected: FAIL — `ImportError: cannot import name 'EtagereCellState'`

- [ ] **Step 3: Implement**

In `backbone/comms/schemas.py`:
1. Add `ETAGERE_STATE = "etagere_state"` to `MessageType` (after `DETECTION_SET`).
2. After the `DetectionSetMessage` class add:
```python
class EtagereCellState(BaseModel):
    """One shelf cell (row r, col c, 1-based) of an étagère grid."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    r: int = Field(..., ge=1)
    c: int = Field(..., ge=1)
    state: Literal["filled", "empty", "unknown"]
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class EtagereStateMessage(BaseModel):
    """Occupancy matrix of one étagère (bin rack) as seen by one camera.

    Produced RAW by the perception producer (isistream) per tick on the
    Backbone's points ingest port (``stabilized=False``), and re-published
    STABILISED by the Backbone (``stabilized=True``) on the metadata sinks —
    on MQTT retained at ``{prefix}/etagere/{zone_id}``. ``cells`` holds
    exactly ``rows*cols`` entries in reading order (r1c1, r1c2, …).
    ``unknown`` = no confident detection for that cell.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(default=SCHEMA_VERSION, ge=1)
    type: Literal[MessageType.ETAGERE_STATE] = MessageType.ETAGERE_STATE
    ts: float = Field(..., description="capture_ts of the source frame")
    camera_id: str
    zone_id: str
    name: str = ""
    rows: int = Field(3, ge=1)
    cols: int = Field(3, ge=1)
    cells: tuple[EtagereCellState, ...]
    seq: int = Field(0, ge=0)
    producer_id: str = ""
    config_fingerprint: str | None = None
    stabilized: bool = False
```
3. In `parse_envelope`: add `| EtagereStateMessage` to the return union and, before the final unknown-type raise, `if msg_type == MessageType.ETAGERE_STATE.value: return EtagereStateMessage.model_validate(data)`.
4. In the `SCHEMA_VERSION` docstring, append a line: `v6 additive (2026-08-17): etagere_state — EtagereStateMessage; defaulted fields, no bump.`

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_metadata_schemas.py -v && ruff check backbone/comms/schemas.py`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add backbone/comms/schemas.py tests/test_metadata_schemas.py
git commit -m "feat(comms): EtagereStateMessage wire type (additive, no schema bump)"
```

---

### Task 2: Étagère config loader — `backbone/shared/etagere.py`

**Files:**
- Create: `backbone/shared/etagere.py`
- Create: `config/etagere.yaml`
- Test: `tests/test_etagere_config.py`

**Interfaces:**
- Produces:
  ```python
  class EtagereCell(BaseModel): r: int; c: int; rect: tuple[float, float, float, float]  # x0,y0,x1,y1 source px
  class EtagereZone(BaseModel):
      id: str; name: str = ""; camera: str
      frame_wh: tuple[int, int]
      corners: tuple[tuple[float, float], ...] = ()   # 0 or 4 points TL,TR,BR,BL
      rows: int = 3; cols: int = 3
      cells: tuple[EtagereCell, ...]                   # exactly rows*cols, reading order (validated)
      max_fps: float | None = None                     # per-zone override
  class EtagereModel(BaseModel):
      onnx_path: str; class_names: list[str] = ["empty_box", "filled_box"]
      imgsz: int = 320; confidence_threshold: float = 0.3
      crop_margin: float = 0.08; max_fps: float = 2.0
      providers: str | None = None                      # passthrough to yolo_onnx ("auto" default)
  class EtagereConfig(BaseModel):
      model: EtagereModel | None = None; zones: tuple[EtagereZone, ...] = ()
      @property enabled -> bool  # model is not None and len(zones) > 0
  def load_etagere_config(path: str | Path | None) -> EtagereConfig   # missing/None → EtagereConfig()
  def cells_from_corners(corners, rows=3, cols=3) -> list[EtagereCell]  # bilinear split, axis-aligned rect per cell
  def resolve_config_path(backbone_cfg: dict, backbone_yaml_path: str | Path) -> Path
      # backbone_cfg.get("etagere", {}).get("config_path") or <dir of backbone.yaml>/etagere.yaml
  ```

- [ ] **Step 1: Write the failing tests** (`tests/test_etagere_config.py`)

```python
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from backbone.shared.etagere import (
    EtagereCell, EtagereConfig, cells_from_corners, load_etagere_config,
    resolve_config_path,
)


def _cells9():
    return [{"r": r, "c": c, "rect": [c * 10, r * 10, c * 10 + 8, r * 10 + 8]}
            for r in (1, 2, 3) for c in (1, 2, 3)]


def _cfg(tmp_path: Path, **over) -> Path:
    data = {
        "model": {"onnx_path": "models/etagere.onnx"},
        "zones": [{"id": "et_1", "name": "A", "camera": "cam_a",
                   "frame_wh": [1920, 1080], "cells": _cells9()}],
    }
    data.update(over)
    p = tmp_path / "etagere.yaml"
    p.write_text(yaml.safe_dump(data))
    return p


def test_load_missing_file_is_disabled(tmp_path: Path) -> None:
    cfg = load_etagere_config(tmp_path / "nope.yaml")
    assert isinstance(cfg, EtagereConfig) and not cfg.enabled
    assert load_etagere_config(None).enabled is False


def test_load_valid_config(tmp_path: Path) -> None:
    cfg = load_etagere_config(_cfg(tmp_path))
    assert cfg.enabled
    z = cfg.zones[0]
    assert z.id == "et_1" and z.camera == "cam_a" and len(z.cells) == 9
    assert z.cells[0].r == 1 and z.cells[0].c == 1
    assert z.cells[-1].r == 3 and z.cells[-1].c == 3
    assert cfg.model.imgsz == 320 and cfg.model.crop_margin == 0.08
    assert cfg.model.class_names == ["empty_box", "filled_box"]


def test_cells_count_and_order_validated(tmp_path: Path) -> None:
    bad = _cells9()[:8]
    with pytest.raises(ValidationError):
        load_etagere_config(_cfg(tmp_path, zones=[{
            "id": "z", "camera": "cam_a", "frame_wh": [10, 10], "cells": bad}]))
    swapped = _cells9(); swapped[0], swapped[1] = swapped[1], swapped[0]
    with pytest.raises(ValidationError):
        load_etagere_config(_cfg(tmp_path, zones=[{
            "id": "z", "camera": "cam_a", "frame_wh": [10, 10], "cells": swapped}]))


def test_cells_from_corners_axis_aligned_square() -> None:
    cells = cells_from_corners([[0, 0], [90, 0], [90, 90], [0, 90]])
    assert len(cells) == 9 and all(isinstance(c, EtagereCell) for c in cells)
    assert cells[0].rect == pytest.approx((0, 0, 30, 30))
    assert cells[4].rect == pytest.approx((30, 30, 60, 60))
    assert cells[8].rect == pytest.approx((60, 60, 90, 90))
    assert [(c.r, c.c) for c in cells][:4] == [(1, 1), (1, 2), (1, 3), (2, 1)]


def test_cells_from_corners_perspective_uses_bbox_of_quad() -> None:
    # trapezoid: top narrower than bottom
    cells = cells_from_corners([[30, 0], [60, 0], [90, 90], [0, 90]])
    x0, y0, x1, y1 = cells[0].rect
    assert x0 < x1 and y0 < y1 and y0 == pytest.approx(0)


def test_resolve_config_path(tmp_path: Path) -> None:
    by = tmp_path / "config" / "backbone.yaml"
    assert resolve_config_path({}, by) == by.parent / "etagere.yaml"
    assert resolve_config_path({"etagere": {"config_path": "/x/e.yaml"}}, by) == Path("/x/e.yaml")
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_etagere_config.py -v`
Expected: FAIL — `ModuleNotFoundError: backbone.shared.etagere`

- [ ] **Step 3: Implement `backbone/shared/etagere.py`**

```python
"""Étagère (bin-rack) zone configuration — shared by isistream, the Backbone
and monitor_web (all may import backbone.shared).

An étagère zone is a per-camera IMAGE-SPACE grid of cells (rows x cols
axis-aligned rectangles in source-frame pixels), NOT a floor polygon — it is
deliberately kept out of zones.yaml / zone_patches so it never enters the
floor-projection pipeline. Authored by the dashboard Settings (4 corners →
auto-split → per-cell drag-adjust), consumed by isistream for per-cell crop
inference. Missing file ⇒ feature off.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_FILENAME = "etagere.yaml"


class EtagereCell(BaseModel):
    model_config = ConfigDict(extra="forbid")
    r: int = Field(..., ge=1)
    c: int = Field(..., ge=1)
    rect: tuple[float, float, float, float]   # x0, y0, x1, y1 (source px)

    @model_validator(mode="after")
    def _ordered(self) -> "EtagereCell":
        x0, y0, x1, y1 = self.rect
        if not (x1 > x0 and y1 > y0):
            raise ValueError(f"cell r{self.r}c{self.c}: rect must be x1>x0, y1>y0")
        return self


class EtagereZone(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    name: str = ""
    camera: str
    frame_wh: tuple[int, int]
    corners: tuple[tuple[float, float], ...] = ()
    rows: int = Field(3, ge=1)
    cols: int = Field(3, ge=1)
    cells: tuple[EtagereCell, ...]
    max_fps: float | None = None

    @model_validator(mode="after")
    def _grid(self) -> "EtagereZone":
        if self.corners and len(self.corners) != 4:
            raise ValueError("corners must be empty or exactly 4 points (TL,TR,BR,BL)")
        expect = [(r, c) for r in range(1, self.rows + 1) for c in range(1, self.cols + 1)]
        got = [(cell.r, cell.c) for cell in self.cells]
        if got != expect:
            raise ValueError(
                f"zone {self.id!r}: cells must be exactly rows*cols in reading order "
                f"(expected {expect[:3]}…, got {got[:3]}…)")
        return self


class EtagereModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    onnx_path: str
    class_names: list[str] = ["empty_box", "filled_box"]
    imgsz: int = Field(320, ge=64)
    confidence_threshold: float = Field(0.3, ge=0.0, le=1.0)
    crop_margin: float = Field(0.08, ge=0.0, le=0.5)
    max_fps: float = Field(2.0, gt=0.0)
    providers: str | None = None


class EtagereConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: EtagereModel | None = None
    zones: tuple[EtagereZone, ...] = ()

    @property
    def enabled(self) -> bool:
        return self.model is not None and len(self.zones) > 0


def load_etagere_config(path: str | Path | None) -> EtagereConfig:
    """Load ``etagere.yaml``; a missing/None path is the disabled config."""
    if path is None:
        return EtagereConfig()
    p = Path(path)
    if not p.exists():
        return EtagereConfig()
    data = yaml.safe_load(p.read_text()) or {}
    return EtagereConfig.model_validate(data)


def resolve_config_path(backbone_cfg: dict, backbone_yaml_path: str | Path) -> Path:
    """``backbone.yaml``'s ``etagere.config_path`` or ``<its dir>/etagere.yaml``."""
    explicit = (backbone_cfg.get("etagere") or {}).get("config_path")
    if explicit:
        return Path(explicit)
    return Path(backbone_yaml_path).parent / DEFAULT_FILENAME


def cells_from_corners(corners, rows: int = 3, cols: int = 3) -> list[EtagereCell]:
    """Auto-split an outer quad (TL,TR,BR,BL) into rows*cols cells.

    Bilinear interpolation of the quad at the grid fractions gives each cell's
    4 corners; the cell rect is that quad's axis-aligned bounding box (crops
    are rectangles). Reading order: r1c1, r1c2, …
    """
    (tlx, tly), (trx, try_), (brx, bry), (blx, bly) = [(float(x), float(y)) for x, y in corners]

    def pt(u: float, v: float) -> tuple[float, float]:
        topx, topy = tlx * (1 - u) + trx * u, tly * (1 - u) + try_ * u
        botx, boty = blx * (1 - u) + brx * u, bly * (1 - u) + bry * u
        return topx * (1 - v) + botx * v, topy * (1 - v) + boty * v

    out: list[EtagereCell] = []
    for r in range(rows):
        for c in range(cols):
            us = (c / cols, (c + 1) / cols)
            vs = (r / rows, (r + 1) / rows)
            quad = [pt(us[0], vs[0]), pt(us[1], vs[0]), pt(us[1], vs[1]), pt(us[0], vs[1])]
            xs = [p[0] for p in quad]
            ys = [p[1] for p in quad]
            out.append(EtagereCell(r=r + 1, c=c + 1,
                                   rect=(min(xs), min(ys), max(xs), max(ys))))
    return out
```

Then create `config/etagere.yaml` (example, one zone; the operator overwrites it from Settings):
```yaml
# Étagère (bin-rack) zones — authored by the dashboard Settings (Étagère
# section), consumed by isistream. Image-space per-camera cell rects.
model:
  onnx_path: trainer/isidet/runs/detect/models/yolo/yolo26n_e100_320px_17-08-2026_12-10-27/weights/best.onnx
  class_names: [empty_box, filled_box]
  imgsz: 320
  confidence_threshold: 0.3
  crop_margin: 0.08
  max_fps: 2.0
zones: []
```

- [ ] **Step 4: Run tests + lint**

Run: `pytest tests/test_etagere_config.py -v && ruff check backbone/shared/etagere.py tests/test_etagere_config.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backbone/shared/etagere.py config/etagere.yaml tests/test_etagere_config.py
git commit -m "feat(shared): étagère zone config loader + auto-split from 4 corners"
```

---

### Task 3: Sinks + Publisher fan-out

**Files:**
- Modify: `backbone/core/interfaces.py` (after `publish_zone_state` ~122-130)
- Modify: `backbone/comms/publisher.py` (after `publish_zone_state` ~81-91)
- Modify: `backbone/comms/udp_sink.py` (after `publish_zone_state` ~118-120)
- Modify: `backbone/comms/mqtt_sink.py` (ctor kwargs ~95-106, stores ~190-198, method after `publish_zone_state` ~327-346)
- Test: `tests/test_publisher.py`, `tests/test_udp_sink.py`, `tests/test_mqtt_sink.py`

**Interfaces:**
- Produces: `MetadataSink.publish_etagere_state(self, msg: object) -> None` (default no-op); `Publisher.publish_etagere_state(msg)`; `MqttSink(..., etagere_topic="{prefix}/etagere/{zone_id}")` publishing `retain=True, qos=zone_state_qos`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_mqtt_sink.py`:
```python
def _make_etagere_state():
    from backbone.comms.schemas import EtagereCellState, EtagereStateMessage
    cells = tuple(EtagereCellState(r=r, c=c, state="filled", confidence=0.9)
                  for r in (1, 2, 3) for c in (1, 2, 3))
    return EtagereStateMessage(ts=1.0, camera_id="cam_a", zone_id="et_1",
                               name="A", cells=cells, stabilized=True)


def test_publish_etagere_state_topic_retained() -> None:
    with patch("backbone.comms.mqtt_sink.mqtt.Client") as MockClient:
        mock_instance = MagicMock()
        MockClient.return_value = mock_instance
        from backbone.comms.mqtt_sink import MqttSink
        sink = MqttSink(host="127.0.0.1", port=1883, prefix="isiMonitor3D/v1/node_a",
                        retain=False)
        sink.publish_etagere_state(_make_etagere_state())
        topic, payload = mock_instance.publish.call_args[0][:2]
        kwargs = mock_instance.publish.call_args[1]
        assert topic == "isiMonitor3D/v1/node_a/etagere/et_1"
        assert kwargs.get("retain") is True and kwargs.get("qos") == 1
        msg = json.loads(payload.decode("utf-8"))
        assert msg["type"] == "etagere_state" and len(msg["cells"]) == 9
        sink.close()


def test_publish_etagere_state_custom_topic() -> None:
    with patch("backbone.comms.mqtt_sink.mqtt.Client") as MockClient:
        mock_instance = MagicMock()
        MockClient.return_value = mock_instance
        from backbone.comms.mqtt_sink import MqttSink
        sink = MqttSink(host="127.0.0.1", port=1883, prefix="p",
                        etagere_topic="{prefix}/shelf/{zone_id}", zone_state_qos=2)
        sink.publish_etagere_state(_make_etagere_state())
        assert mock_instance.publish.call_args[0][0] == "p/shelf/et_1"
        assert mock_instance.publish.call_args[1]["qos"] == 2
        sink.close()
```
Append to `tests/test_udp_sink.py` (uses the file's existing `_bind_receiver()` helper):
```python
def test_publish_etagere_state_arrives_as_json() -> None:
    from backbone.comms.schemas import EtagereCellState, EtagereStateMessage
    sock, port = _bind_receiver()
    try:
        sink = UdpSink(host="127.0.0.1", port=port)
        sink.publish_etagere_state(EtagereStateMessage(
            ts=1.0, camera_id="cam_a", zone_id="et_1", rows=1, cols=1,
            cells=(EtagereCellState(r=1, c=1, state="empty"),), stabilized=True))
        payload, _ = sock.recvfrom(8192)
        msg = json.loads(payload.decode("utf-8"))
        assert msg["type"] == "etagere_state" and msg["zone_id"] == "et_1"
        assert msg["cells"][0]["state"] == "empty" and msg["stabilized"] is True
        sink.close()
    finally:
        sock.close()
```
Append to `tests/test_publisher.py`. First extend the file's `_RecordingSink`: add `self.etagere_states: list[object] = []` in `__init__` and the method
```python
    def publish_etagere_state(self, msg: object) -> None:
        self.etagere_states.append(msg)
```
then the tests:
```python
def test_fan_out_etagere_state() -> None:
    from backbone.comms.schemas import EtagereStateMessage
    a, b = _RecordingSink(), _RecordingSink()
    pub = Publisher([a, b])
    msg = EtagereStateMessage(ts=0.0, camera_id="cam_a", zone_id="z", cells=())
    pub.publish_etagere_state(msg)
    assert a.etagere_states == [msg] and b.etagere_states == [msg]


def test_sink_default_publish_etagere_state_is_noop() -> None:
    from backbone.comms.schemas import EtagereStateMessage
    class _Bare(MetadataSink):
        # only the abstract methods, copied from how _RecordingSink satisfies the ABC
        def publish_track_2d(self, track): pass
        def publish_track_3d(self, track): pass
        def close(self): pass
    _Bare().publish_etagere_state(
        EtagereStateMessage(ts=0.0, camera_id="cam_a", zone_id="z", cells=()))   # must not raise
```
(If `MetadataSink` declares more abstract methods than these three, implement them the same way `_RecordingSink` does — check `backbone/core/interfaces.py`.)

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_mqtt_sink.py tests/test_udp_sink.py tests/test_publisher.py -k etagere -v`
Expected: FAIL — `AttributeError: ... has no attribute 'publish_etagere_state'`

- [ ] **Step 3: Implement**

`backbone/core/interfaces.py` — after `publish_zone_state`:
```python
    def publish_etagere_state(self, msg: object) -> None:
        """Publish an ``EtagereStateMessage`` — one shelf rack's cell matrix.

        Non-abstract default no-op (same rationale as ``publish_zone_state``);
        MQTT sinks override to publish retained on ``{prefix}/etagere/{zone_id}``.
        """
        return None
```
`backbone/comms/publisher.py` — after `publish_zone_state`:
```python
    def publish_etagere_state(self, msg: object) -> None:
        """Fan-out an ``EtagereStateMessage`` (MQTT publishes it retained)."""
        if self._closed:
            return
        for sink in self._sinks:
            try:
                sink.publish_etagere_state(msg)
            except Exception:
                logger.warning(
                    "sink %s failed on publish_etagere_state", type(sink).__name__, exc_info=True
                )
```
`backbone/comms/udp_sink.py` — import `EtagereStateMessage` with the other schema imports; after `publish_zone_state`:
```python
    def publish_etagere_state(self, msg: object) -> None:
        assert isinstance(msg, EtagereStateMessage)
        self._send(msg.model_dump_json().encode("utf-8"))
```
`backbone/comms/mqtt_sink.py` — import `EtagereStateMessage`; add ctor kwarg `etagere_topic: str = "{prefix}/etagere/{zone_id}",` right after `zone_state_qos`; store `self._etagere_topic = etagere_topic` next to `self._zone_state_qos`; add after `publish_zone_state`:
```python
    def publish_etagere_state(self, msg: object) -> None:
        """Publish an ``EtagereStateMessage`` retained on ``{prefix}/etagere/{zone_id}``
        at ``zone_state_qos`` — the same late-joiner hygiene as zone_state."""
        assert isinstance(msg, EtagereStateMessage)
        topic = self._etagere_topic.format(prefix=self._prefix,
                                           zone_id=self._sanitize(msg.zone_id))
        payload = msg.model_dump_json().encode("utf-8")
        try:
            self._client.publish(topic, payload, qos=self._zone_state_qos, retain=True)
        except Exception:
            logger.warning("MqttSink.publish_etagere_state failed on topic %r", topic,
                           exc_info=True)
```
If the file has no `_sanitize` helper, reuse whatever `publish_zone_state` uses to make the zone segment topic-safe (`_zone_segment` → look at how it sanitises the name and call the same underlying function; a topic segment must not contain `/`, `+`, `#`).

- [ ] **Step 4: Run tests + lint**

Run: `pytest tests/test_mqtt_sink.py tests/test_udp_sink.py tests/test_publisher.py tests/test_registry.py -v && ruff check backbone`
Expected: PASS (incl. `test_five_seams_present`).

- [ ] **Step 5: Commit**

```bash
git add backbone/core/interfaces.py backbone/comms/publisher.py backbone/comms/udp_sink.py backbone/comms/mqtt_sink.py tests/test_publisher.py tests/test_udp_sink.py tests/test_mqtt_sink.py
git commit -m "feat(comms): publish_etagere_state fan-out; UDP + MQTT (retained {prefix}/etagere/{zone_id})"
```

---

### Task 4: Per-cell hysteresis — `EtagereStateTracker`

**Files:**
- Create: `backbone/shared/etagere_state.py`
- Test: `tests/test_etagere_state.py`

**Interfaces:**
- Produces:
  ```python
  class EtagereStateTracker:
      def __init__(self, *, window: int = 15, flip_ratio: float = 0.7,
                   unknown_after_s: float = 5.0, heartbeat_s: float = 5.0) -> None
      def update(self, msg: EtagereStateMessage, now: float | None = None) -> EtagereStateMessage | None
          # returns the stabilised message (stabilized=True) when any cell's held state changed
          # since the last emit for that zone OR heartbeat_s elapsed since the last emit; else None
      def forget_zone(self, zone_id: str) -> None
  ```

- [ ] **Step 1: Write the failing tests** (`tests/test_etagere_state.py`)

```python
from __future__ import annotations

from backbone.comms.schemas import EtagereCellState, EtagereStateMessage
from backbone.shared.etagere_state import EtagereStateTracker


def _msg(states: list[str], ts: float, conf: float = 0.9) -> EtagereStateMessage:
    cells = tuple(EtagereCellState(r=i // 3 + 1, c=i % 3 + 1, state=s,
                                   confidence=(conf if s != "unknown" else 0.0))
                  for i, s in enumerate(states))
    return EtagereStateMessage(ts=ts, camera_id="cam_a", zone_id="et_1", name="A",
                               cells=cells, seq=int(ts * 10))


def _grid(msg) -> list[str]:
    return [c.state for c in msg.cells]


def test_first_observation_emits_immediately() -> None:
    tr = EtagereStateTracker()
    out = tr.update(_msg(["filled"] * 9, 0.0), now=0.0)
    assert out is not None and out.stabilized is True
    assert _grid(out) == ["filled"] * 9 and out.zone_id == "et_1"


def test_held_state_needs_supermajority_to_flip() -> None:
    tr = EtagereStateTracker(window=10, flip_ratio=0.7, heartbeat_s=1e9)
    tr.update(_msg(["filled"] * 9, 0.0), now=0.0)
    # 5 challenger votes in a window of 10 → 50% < 70% → hold
    for k in range(1, 6):
        assert tr.update(_msg(["empty"] * 9, float(k)), now=float(k)) is None
    # 7th challenger vote → 7/10 ≥ 70% → flip, emitted once
    tr.update(_msg(["empty"] * 9, 6.0), now=6.0)
    out = tr.update(_msg(["empty"] * 9, 7.0), now=7.0)
    assert out is not None and _grid(out) == ["empty"] * 9


def test_unknown_does_not_vote_but_decays_after_timeout() -> None:
    tr = EtagereStateTracker(unknown_after_s=5.0, heartbeat_s=1e9)
    tr.update(_msg(["filled"] * 9, 0.0), now=0.0)
    assert tr.update(_msg(["unknown"] * 9, 1.0), now=1.0) is None      # still filled (held)
    out = tr.update(_msg(["unknown"] * 9, 6.0), now=6.0)             # >5 s without a vote
    assert out is not None and _grid(out) == ["unknown"] * 9


def test_heartbeat_re_emits_unchanged_state() -> None:
    tr = EtagereStateTracker(heartbeat_s=5.0)
    tr.update(_msg(["empty"] * 9, 0.0), now=0.0)
    assert tr.update(_msg(["empty"] * 9, 1.0), now=1.0) is None
    out = tr.update(_msg(["empty"] * 9, 5.5), now=5.5)
    assert out is not None and _grid(out) == ["empty"] * 9


def test_per_cell_independence_and_confidence_carried() -> None:
    tr = EtagereStateTracker(heartbeat_s=1e9)
    states = ["filled"] * 9
    tr.update(_msg(states, 0.0, conf=0.8), now=0.0)
    states[4] = "empty"
    out = None
    for k in range(1, 15):
        out = tr.update(_msg(states, float(k), conf=0.8), now=float(k)) or out
    assert out is not None
    g = _grid(out)
    assert g[4] == "empty" and g[0] == "filled" and g[8] == "filled"
    assert out.cells[4].confidence == 0.8


def test_forget_zone_resets_history() -> None:
    tr = EtagereStateTracker(heartbeat_s=1e9)
    tr.update(_msg(["filled"] * 9, 0.0), now=0.0)
    tr.forget_zone("et_1")
    out = tr.update(_msg(["empty"] * 9, 1.0), now=1.0)
    assert out is not None and _grid(out) == ["empty"] * 9
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_etagere_state.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `backbone/shared/etagere_state.py`**

```python
"""Temporal stabilisation of étagère cell states.

Mirrors ``OccupancyStabilizer`` (backbone/homography/pallet_occupancy.py) but
keyed per (zone_id, r, c) over the two-state alphabet filled/empty: the first
observation sets a cell immediately; a held state flips only when the
challenger wins ≥ ``flip_ratio`` of the vote window. ``unknown`` never votes
(a hand over the bin must not flip it) but a cell with no vote for
``unknown_after_s`` decays to ``unknown`` (fail honestly). ``update`` returns
a message only on change or heartbeat — retained MQTT + heartbeat is the
same late-joiner hygiene zone_state uses.
"""

from __future__ import annotations

import math
import time
from collections import Counter, deque

from backbone.comms.schemas import EtagereCellState, EtagereStateMessage

_VOTING = ("filled", "empty")


class EtagereStateTracker:
    def __init__(self, *, window: int = 15, flip_ratio: float = 0.7,
                 unknown_after_s: float = 5.0, heartbeat_s: float = 5.0) -> None:
        self._window = int(window)
        self._flip = float(flip_ratio)
        self._unknown_after = float(unknown_after_s)
        self._heartbeat = float(heartbeat_s)
        self._hist: dict[tuple[str, int, int], deque] = {}
        self._held: dict[tuple[str, int, int], str] = {}
        self._conf: dict[tuple[str, int, int], float] = {}
        self._last_vote_t: dict[tuple[str, int, int], float] = {}
        self._last_emit_t: dict[str, float] = {}
        self._last_emit_grid: dict[str, tuple[str, ...]] = {}

    def forget_zone(self, zone_id: str) -> None:
        for d in (self._hist, self._held, self._conf, self._last_vote_t):
            for k in [k for k in d if k[0] == zone_id]:
                d.pop(k, None)
        self._last_emit_t.pop(zone_id, None)
        self._last_emit_grid.pop(zone_id, None)

    def _vote(self, key, state: str, conf: float, now: float) -> str:
        held = self._held.get(key)
        if state in _VOTING:
            hist = self._hist.setdefault(key, deque(maxlen=self._window))
            hist.append(state)
            self._last_vote_t[key] = now
            counts = Counter(hist)
            if held not in _VOTING:
                new = counts.most_common(1)[0][0]
            else:
                challenger = "empty" if held == "filled" else "filled"
                need = math.ceil(self._flip * len(hist))
                new = challenger if counts.get(challenger, 0) >= need else held
            self._held[key] = new
            if new == state:
                self._conf[key] = conf
            return new
        # unknown observation: hold, unless the hold is stale
        last = self._last_vote_t.get(key)
        if held in _VOTING and last is not None and now - last <= self._unknown_after:
            return held
        self._held[key] = "unknown"
        self._conf[key] = 0.0
        self._hist.pop(key, None)
        return "unknown"

    def update(self, msg: EtagereStateMessage, now: float | None = None) -> EtagereStateMessage | None:
        t = time.time() if now is None else float(now)
        out_cells = []
        for cell in msg.cells:
            key = (msg.zone_id, cell.r, cell.c)
            state = self._vote(key, cell.state, cell.confidence, t)
            out_cells.append(EtagereCellState(r=cell.r, c=cell.c, state=state,
                                              confidence=self._conf.get(key, 0.0)))
        grid = tuple(c.state for c in out_cells)
        last_t = self._last_emit_t.get(msg.zone_id)
        changed = grid != self._last_emit_grid.get(msg.zone_id)
        due = last_t is None or (t - last_t) >= self._heartbeat
        if not (changed or due):
            return None
        self._last_emit_t[msg.zone_id] = t
        self._last_emit_grid[msg.zone_id] = grid
        return EtagereStateMessage(
            ts=msg.ts, camera_id=msg.camera_id, zone_id=msg.zone_id, name=msg.name,
            rows=msg.rows, cols=msg.cols, cells=tuple(out_cells), seq=msg.seq,
            producer_id=msg.producer_id, config_fingerprint=msg.config_fingerprint,
            stabilized=True,
        )
```

- [ ] **Step 4: Run tests + lint**

Run: `pytest tests/test_etagere_state.py -v && ruff check backbone/shared/etagere_state.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backbone/shared/etagere_state.py tests/test_etagere_state.py
git commit -m "feat(shared): EtagereStateTracker — per-cell flip hysteresis, unknown decay, heartbeat"
```

---

### Task 5: Backbone ingest routing + orchestrator wiring

**Files:**
- Modify: `backbone/ingestion/points_in.py` (ctor 118-135, `_handle_payload` 190-237)
- Modify: `backbone/runtime/orchestrator.py` (points block ~205-227; publisher available as `self._publisher` — confirm the attribute name used at the `publish_zone_state` call sites ~667/1050 and use the same)
- Test: `tests/test_points_in_etagere.py`

**Interfaces:**
- Consumes: `EtagereStateMessage` (Task 1), `EtagereStateTracker` (Task 4), `Publisher.publish_etagere_state` (Task 3).
- Produces: `DetectionIngest(..., on_etagere: Callable[[EtagereStateMessage], None] | None = None)`; counter `self.etagere_by_zone: dict[str, int]`.

- [ ] **Step 1: Write the failing test** (`tests/test_points_in_etagere.py`)

```python
from __future__ import annotations

import json
import socket
import time

from backbone.comms.schemas import EtagereCellState, EtagereStateMessage
from backbone.ingestion.points_in import DetectionIngest


def _msg(cam="cam_a"):
    return EtagereStateMessage(ts=1.0, camera_id=cam, zone_id="et_1",
                               cells=(EtagereCellState(r=1, c=1, state="filled"),),
                               rows=1, cols=1)


def _send(port: int, msg) -> None:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.sendto(msg.model_dump_json().encode("utf-8"), ("127.0.0.1", port))
    s.close()


def _wait(pred, timeout=2.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.02)
    return False


def test_etagere_routed_to_on_etagere_not_on_set() -> None:
    sets, shelves = [], []
    ing = DetectionIngest(["cam_a"], port=0, on_set=sets.append, on_etagere=shelves.append)
    ing.start()
    try:
        _send(ing.port, _msg())
        assert _wait(lambda: len(shelves) == 1)
        assert isinstance(shelves[0], EtagereStateMessage) and shelves[0].zone_id == "et_1"
        assert sets == []
        assert ing.etagere_by_zone.get("et_1") == 1
    finally:
        ing.stop()


def test_etagere_unknown_camera_ignored_and_no_callback_counts_dropped() -> None:
    shelves = []
    ing = DetectionIngest(["cam_a"], port=0, on_set=lambda s: None, on_etagere=shelves.append)
    ing.start()
    try:
        _send(ing.port, _msg(cam="cam_zzz"))
        time.sleep(0.2)
        assert shelves == []
    finally:
        ing.stop()
    ing2 = DetectionIngest(["cam_a"], port=0, on_set=lambda s: None)   # no on_etagere
    ing2.start()
    try:
        _send(ing2.port, _msg())
        assert _wait(lambda: ing2.dropped_malformed >= 1)
    finally:
        ing2.stop()
```
If `DetectionIngest` has no `.port` property exposing the bound port when constructed with `port=0`, add one (`@property def port(self) -> int: return self._sock.getsockname()[1]` after `start()`); check the existing test `tests/test_points_mode.py` for how it discovers the port and follow it.

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_points_in_etagere.py -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'on_etagere'`

- [ ] **Step 3: Implement**

`backbone/ingestion/points_in.py`:
- Import `EtagereStateMessage` next to `DetectionSetMessage`.
- Ctor: add `on_etagere=None,` after `on_set,`; store `self._on_etagere = on_etagere`; add `self.etagere_by_zone: dict[str, int] = {}` beside the other counters.
- In `_handle_payload`, replace the block
  ```python
  if not isinstance(msg, DetectionSetMessage):
      with self._lock:
          self.dropped_malformed += 1
      return
  ```
  with
  ```python
  if isinstance(msg, EtagereStateMessage):
      if msg.camera_id not in self._camera_ids:
          logger.debug("points ingest: etagere for unknown camera %r ignored", msg.camera_id)
          return
      if self._on_etagere is None:
          with self._lock:
              self.dropped_malformed += 1
          return
      with self._lock:
          self.etagere_by_zone[msg.zone_id] = self.etagere_by_zone.get(msg.zone_id, 0) + 1
      try:
          self._on_etagere(msg)
      except Exception:
          logger.warning("points ingest: etagere delivery failed", exc_info=True)
      return
  if not isinstance(msg, DetectionSetMessage):
      with self._lock:
          self.dropped_malformed += 1
      return
  ```

`backbone/runtime/orchestrator.py`, inside the `if self._ingest_mode == "points":` block, before constructing `DetectionIngest`:
```python
            from backbone.shared.etagere_state import EtagereStateTracker
            etagere_cfg = dict(cfg.get("etagere", {}) or {})
            self._etagere_tracker = EtagereStateTracker(
                window=int(etagere_cfg.get("stabilize_window", 15)),
                flip_ratio=float(etagere_cfg.get("stabilize_flip_ratio", 0.7)),
                unknown_after_s=float(etagere_cfg.get("unknown_after_s", 5.0)),
                heartbeat_s=float(etagere_cfg.get("heartbeat_s", 5.0)),
            )

            def _deliver_etagere(msg) -> None:
                out = self._etagere_tracker.update(msg)
                if out is not None and self._publisher is not None:
                    self._publisher.publish_etagere_state(out)
```
and pass `on_etagere=_deliver_etagere,` to `DetectionIngest(...)`. (Confirm the publisher attribute name — grep `publish_zone_state(` in the orchestrator and use the same object; if the publisher is built AFTER this block, keep the closure lazy as written — it reads `self._publisher` at call time.) Initialise `self._etagere_tracker = None` in `__init__` next to the other stack attributes.

- [ ] **Step 4: Run tests + lint**

Run: `pytest tests/test_points_in_etagere.py tests/test_points_mode.py tests/test_orchestrator.py -v && ruff check backbone`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backbone/ingestion/points_in.py backbone/runtime/orchestrator.py tests/test_points_in_etagere.py
git commit -m "feat(ingest): route etagere_state through the points ingest → tracker → publisher"
```

---

### Task 6: isistream `EtagereDetector` + tick integration

**Files:**
- Create: `isistream/etagere.py`
- Modify: `isistream/core.py` (`IsistreamCore.__init__` 188-232, `tick` 282-358, `build_isistream_core` 360-419)
- Modify: `isistream/__main__.py` (pass the backbone.yaml path so the étagère config can be resolved — see step 3)
- Test: `tests/test_isistream_etagere.py`

**Interfaces:**
- Consumes: `EtagereConfig/EtagereZone/EtagereCell`, `load_etagere_config`, `resolve_config_path` (Task 2); `EtagereStateMessage` (Task 1); `detector_registry` + `Detection` (`backbone.core`), `Frame`, `FramePair` (`backbone.core.types`).
- Produces:
  ```python
  class EtagereDetector:
      def __init__(self, cfg: EtagereConfig, detector, *, producer_id: str = "isistream",
                   fingerprint: str | None = None) -> None
          # detector: any object with .detect(FramePair) -> dict[str, list[Detection]]
      def due_zones(self, frames: dict[str, Frame], now: float) -> list[EtagereZone]
      def run(self, frames: dict[str, Frame], now: float) -> list[EtagereStateMessage]
          # crops due zones' cells (rect+margin, scaled frame_wh→actual), ONE detect() call,
          # per-cell decision, one message per zone (seq per zone)
  def build_etagere_detector(cfg: EtagereConfig, *, producer_id, fingerprint) -> EtagereDetector | None
      # None when not cfg.enabled; builds detector_registry.create("yolo_onnx", onnx_path=...,
      # class_names=..., confidence_threshold=..., input_size=(imgsz, imgsz), providers=<"auto"|cfg>)
  @staticmethod decide(dets: list[Detection], threshold: float) -> tuple[str, float]  # ("filled"|"empty"|"unknown", conf)
  ```
  `IsistreamCore(..., etagere_detector=None)`; per tick after pose: `for msg in self._etagere.run(fresh, now()): send_json_datagram(...)`; `stage_ms["etagere"]`; counter `self.etagere_sent: dict[str, int]`.

- [ ] **Step 1: Write the failing tests** (`tests/test_isistream_etagere.py`)

```python
from __future__ import annotations

import numpy as np

from backbone.core.types import Detection, Frame, FramePair
from backbone.shared.etagere import EtagereCell, EtagereConfig, EtagereModel, EtagereZone
from isistream.etagere import EtagereDetector


class _FakeDet:
    """Returns, per crop key, the detections queued for it; records the batch."""
    def __init__(self, by_key):
        self.by_key = by_key
        self.pairs: list[FramePair] = []

    def detect(self, pair: FramePair):
        self.pairs.append(pair)
        return {k: self.by_key.get(k, []) for k in pair.frames}


def _det(cls: str, conf: float, key: str) -> Detection:
    return Detection(camera_id=key, capture_ts=0.0, cls=cls, confidence=conf,
                     bbox_xyxy=(10, 10, 100, 100), foot_uv=(55, 100), keypoints_uv=None)


def _cfg(max_fps=2.0, margin=0.08) -> EtagereConfig:
    cells = [EtagereCell(r=r, c=c, rect=(c * 100, r * 100, c * 100 + 80, r * 100 + 80))
             for r in (1, 2, 3) for c in (1, 2, 3)]
    return EtagereConfig(
        model=EtagereModel(onnx_path="x.onnx", crop_margin=margin, max_fps=max_fps),
        zones=(EtagereZone(id="et_1", name="A", camera="cam_a", frame_wh=(640, 480),
                           cells=tuple(cells)),),
    )


def _frame(w=1280, h=960) -> Frame:
    return Frame(camera_id="cam_a", capture_ts=1.0, frame_idx=0,
                 image=np.zeros((h, w, 3), dtype=np.uint8))


def test_run_batches_nine_crops_and_maps_scale_and_margin() -> None:
    fake = _FakeDet({})
    ed = EtagereDetector(_cfg(), fake, producer_id="p", fingerprint="fp")
    msgs = ed.run({"cam_a": _frame()}, now=10.0)
    assert len(fake.pairs) == 1 and len(fake.pairs[0].frames) == 9
    crop = fake.pairs[0].frames["et_1:1:1"]
    # rect (100,100,180,180) in 640x480 → x2 in 1280x960 → (200,200,360,360), +8% margin (12.8 px each side)
    assert crop.image.shape[0] == crop.image.shape[1]
    assert 176 <= crop.image.shape[0] <= 190
    assert len(msgs) == 1 and msgs[0].zone_id == "et_1" and msgs[0].camera_id == "cam_a"
    assert msgs[0].producer_id == "p" and msgs[0].config_fingerprint == "fp"
    assert [c.state for c in msgs[0].cells] == ["unknown"] * 9
    assert msgs[0].ts == 1.0 and msgs[0].stabilized is False


def test_decision_per_cell() -> None:
    fake = _FakeDet({
        "et_1:1:1": [_det("filled_box", 0.9, "k"), _det("empty_box", 0.4, "k")],
        "et_1:1:2": [_det("empty_box", 0.8, "k")],
        "et_1:1:3": [_det("filled_box", 0.2, "k")],       # below 0.3 → unknown
    })
    ed = EtagereDetector(_cfg(), fake)
    msg = ed.run({"cam_a": _frame()}, now=0.0)[0]
    st = {(c.r, c.c): (c.state, c.confidence) for c in msg.cells}
    assert st[(1, 1)] == ("filled", 0.9)
    assert st[(1, 2)] == ("empty", 0.8)
    assert st[(1, 3)][0] == "unknown"
    assert st[(3, 3)][0] == "unknown"


def test_max_fps_gate_and_seq() -> None:
    fake = _FakeDet({})
    ed = EtagereDetector(_cfg(max_fps=2.0), fake)
    assert len(ed.run({"cam_a": _frame()}, now=0.0)) == 1
    assert ed.run({"cam_a": _frame()}, now=0.1) == []          # 0.5 s interval not elapsed
    m = ed.run({"cam_a": _frame()}, now=0.6)
    assert len(m) == 1 and m[0].seq == 1


def test_zone_without_fresh_frame_skipped() -> None:
    fake = _FakeDet({})
    ed = EtagereDetector(_cfg(), fake)
    assert ed.run({"cam_b": _frame()}, now=0.0) == []
    assert fake.pairs == []
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_isistream_etagere.py -v`
Expected: FAIL — `ModuleNotFoundError: isistream.etagere`

- [ ] **Step 3: Implement `isistream/etagere.py`**

```python
"""Étagère (bin-rack) cell inference — the producer-side half.

For every configured zone whose camera has a fresh frame (and whose max_fps
interval elapsed): scale its cell rects from ``frame_wh`` to the actual frame,
crop rect + margin (the same margin used at training time), key each crop
``"{zone_id}:{r}:{c}"``, batch ALL due crops into ONE ``FramePair`` → one
detector call (the yolo_onnx plugin letterboxes to its input size), then per
cell take the top-confidence detection ≥ threshold: filled_box → "filled",
empty_box → "empty", none → "unknown". Emits one raw ``EtagereStateMessage``
per zone; the Backbone stabilises and publishes.
"""

from __future__ import annotations

import logging

from backbone.comms.schemas import EtagereCellState, EtagereStateMessage
from backbone.core.types import Detection, Frame, FramePair
from backbone.shared.etagere import EtagereConfig, EtagereZone

logger = logging.getLogger(__name__)

_SEP = ":"


def decide(dets: list[Detection], threshold: float) -> tuple[str, float]:
    best = None
    for d in dets:
        if d.confidence >= threshold and (best is None or d.confidence > best.confidence):
            best = d
    if best is None:
        return "unknown", 0.0
    if best.cls == "filled_box":
        return "filled", float(best.confidence)
    if best.cls == "empty_box":
        return "empty", float(best.confidence)
    return "unknown", 0.0


class EtagereDetector:
    def __init__(self, cfg: EtagereConfig, detector, *, producer_id: str = "isistream",
                 fingerprint: str | None = None) -> None:
        assert cfg.model is not None
        self._cfg = cfg
        self._det = detector
        self._producer_id = producer_id
        self._fingerprint = fingerprint
        self._seq: dict[str, int] = {z.id: 0 for z in cfg.zones}
        self._last_run: dict[str, float] = {}

    def due_zones(self, frames: dict[str, Frame], now: float) -> list[EtagereZone]:
        out = []
        for z in self._cfg.zones:
            if z.camera not in frames:
                continue
            fps = z.max_fps or self._cfg.model.max_fps
            last = self._last_run.get(z.id)
            if last is not None and (now - last) < 1.0 / fps:
                continue
            out.append(z)
        return out

    def _crop(self, frame: Frame, zone: EtagereZone, rect):
        h, w = frame.image.shape[:2]
        sx = w / float(zone.frame_wh[0])
        sy = h / float(zone.frame_wh[1])
        x0, y0, x1, y1 = rect[0] * sx, rect[1] * sy, rect[2] * sx, rect[3] * sy
        m = self._cfg.model.crop_margin
        mx, my = (x1 - x0) * m, (y1 - y0) * m
        cx0, cy0 = max(int(x0 - mx), 0), max(int(y0 - my), 0)
        cx1, cy1 = min(int(x1 + mx), w), min(int(y1 + my), h)
        if cx1 - cx0 < 4 or cy1 - cy0 < 4:
            return None
        return frame.image[cy0:cy1, cx0:cx1]

    def run(self, frames: dict[str, Frame], now: float) -> list[EtagereStateMessage]:
        due = self.due_zones(frames, now)
        if not due:
            return []
        crops: dict[str, Frame] = {}
        for z in due:
            frame = frames[z.camera]
            for cell in z.cells:
                img = self._crop(frame, z, cell.rect)
                if img is None:
                    continue
                key = f"{z.id}{_SEP}{cell.r}{_SEP}{cell.c}"
                crops[key] = Frame(camera_id=key, capture_ts=frame.capture_ts,
                                   frame_idx=frame.frame_idx, image=img)
        results = self._det.detect(FramePair(frames=crops)) if crops else {}
        thr = self._cfg.model.confidence_threshold
        out = []
        for z in due:
            self._last_run[z.id] = now
            frame = frames[z.camera]
            cells = []
            for cell in z.cells:
                key = f"{z.id}{_SEP}{cell.r}{_SEP}{cell.c}"
                state, conf = decide(list(results.get(key, [])), thr)
                cells.append(EtagereCellState(r=cell.r, c=cell.c, state=state, confidence=conf))
            out.append(EtagereStateMessage(
                ts=frame.capture_ts, camera_id=z.camera, zone_id=z.id, name=z.name,
                rows=z.rows, cols=z.cols, cells=tuple(cells), seq=self._seq[z.id],
                producer_id=self._producer_id, config_fingerprint=self._fingerprint,
            ))
            self._seq[z.id] += 1
        return out


def build_etagere_detector(cfg: EtagereConfig, *, producer_id: str = "isistream",
                           fingerprint: str | None = None) -> EtagereDetector | None:
    if not cfg.enabled:
        return None
    import backbone.detection  # noqa: F401  (registers yolo_onnx)
    from backbone.core.interfaces import detector_registry
    m = cfg.model
    kwargs = dict(onnx_path=m.onnx_path, class_names=list(m.class_names),
                  confidence_threshold=m.confidence_threshold,
                  input_size=(m.imgsz, m.imgsz))
    if m.providers:
        kwargs["providers"] = m.providers
    det = detector_registry.create("yolo_onnx", **kwargs)
    logger.info("isistream: étagère detector ready (%d zone(s), imgsz %d)", len(cfg.zones), m.imgsz)
    return EtagereDetector(cfg, det, producer_id=producer_id, fingerprint=fingerprint)
```
Check `FramePair`'s constructor signature in `backbone/core/types.py` (it may require `pair_ts`/`ts` — mirror how `zone_scope.py:_detect_padded` builds its crop pair) and check `Frame`'s field names; adjust the two constructors accordingly. Check `yolo_onnx`'s ctor for the exact `providers` kwarg type (string like "auto" or list) — pass what `tools/detection_smoke.py` passes.

`isistream/core.py`:
- `IsistreamCore.__init__`: add kwarg `etagere_detector=None,`; store `self._etagere = etagere_detector`; add `self.etagere_sent: dict[str, int] = {}`.
- `tick()`: after the pose stage (`self.stage_ms["pose"] = ...`) and before the emit stage add:
```python
        if self._etagere is not None:
            t = now()
            try:
                for msg in self._etagere.run(fresh, now()):
                    send_json_datagram(self._sock, self._addr,
                                       msg.model_dump_json().encode("utf-8"))
                    self.etagere_sent[msg.zone_id] = self.etagere_sent.get(msg.zone_id, 0) + 1
            except Exception:
                self.last_error = "etagere"
                logger.warning("isistream: étagère stage failed", exc_info=True)
            self.stage_ms["etagere"] = (now() - t) * 1000.0
```
- `build_isistream_core(cfg, frame_provider, *, producer_id="isistream", config_path=None)`: add the `config_path` kwarg; before constructing the core:
```python
    from backbone.shared.etagere import load_etagere_config, resolve_config_path
    from isistream.etagere import build_etagere_detector
    etagere = None
    if config_path is not None:
        try:
            et_cfg = load_etagere_config(resolve_config_path(cfg, config_path))
            etagere = build_etagere_detector(et_cfg, producer_id=producer_id,
                                             fingerprint=config_fingerprint(cfg))
        except Exception:
            logger.warning("isistream: étagère config/model failed — feature off", exc_info=True)
```
  and pass `etagere_detector=etagere` to `IsistreamCore(...)`.
- `isistream/__main__.py:182`: `build_isistream_core(cfg, frame_provider, producer_id="isistream", config_path=args.config)`. Also update the in-process caller in `monitor_web/monitor_web/isistream_host.py` if it calls `build_isistream_core` directly (grep; pass the backbone config path it already knows).

- [ ] **Step 4: Run tests + lint**

Run: `pytest tests/test_isistream_etagere.py tests/test_points_mode.py -v && ruff check isistream tests/test_isistream_etagere.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add isistream/etagere.py isistream/core.py isistream/__main__.py tests/test_isistream_etagere.py
git commit -m "feat(isistream): étagère per-cell crop batch inference → EtagereStateMessage"
```

---

### Task 7: isicomms — cache, REST, /ui matrix card

**Files:**
- Modify: `isicomms/isicomms/mqtt_subscriber.py` (`NodeState` ~52-62, `update_from_message` ~319-378)
- Create: `isicomms/isicomms/api/routes_etagere.py`
- Modify: `isicomms/isicomms/app.py` (router list ~20 + include ~111-119)
- Modify: `isicomms/isicomms/api/ui_page.py` (cards ~163-174, JS render ~240, tick ~512-526)
- Test: `isicomms/tests/test_routes_etagere.py`

**Interfaces:**
- Produces: `NodeState.etagere_by_zone: dict[str, EtagereStateMessage]`; `GET /etagere` → `{"etageres": [{node_id, zone_id, name, camera_id, rows, cols, cells:[{r,c,state,confidence}], matrix: [[state,...],...], ts}], "count": N}`; `GET /etagere/{zone_id}` → one entry or 404; both also under the version prefix like the other routers.

- [ ] **Step 1: Write the failing tests** (`isicomms/tests/test_routes_etagere.py`)

```python
from backbone.comms.schemas import EtagereCellState, EtagereStateMessage


def _msg(zone_id="et_1"):
    cells = tuple(EtagereCellState(r=r, c=c, state="filled" if c == 1 else "empty",
                                   confidence=0.9) for r in (1, 2, 3) for c in (1, 2, 3))
    return EtagereStateMessage(ts=100.0, camera_id="cam_a", zone_id=zone_id, name="A",
                               cells=cells, stabilized=True)


def test_etagere_empty(client):
    r = client.get("/etagere")
    assert r.status_code == 200 and r.json() == {"etageres": [], "count": 0}


def test_etagere_listed_with_matrix(client):
    client.app.state.subscriber.update_from_message("node_a", _msg())
    r = client.get("/etagere")
    body = r.json()
    assert body["count"] == 1
    e = body["etageres"][0]
    assert e["node_id"] == "node_a" and e["zone_id"] == "et_1" and e["name"] == "A"
    assert e["matrix"] == [["filled", "empty", "empty"]] * 3
    assert len(e["cells"]) == 9 and e["ts"] == 100.0


def test_etagere_by_id_and_404(client):
    client.app.state.subscriber.update_from_message("node_a", _msg())
    assert client.get("/etagere/et_1").json()["zone_id"] == "et_1"
    assert client.get("/etagere/nope").status_code == 404
    assert client.get("/v1/etagere").status_code == 200
```

- [ ] **Step 2: Run to verify failure**

Run: `cd isicomms && pytest tests/test_routes_etagere.py -v`
Expected: FAIL — 404 on `/etagere` (and/or `update_from_message` ignoring the type).

- [ ] **Step 3: Implement**

`mqtt_subscriber.py`: import `EtagereStateMessage`; add `etagere_by_zone: dict[str, EtagereStateMessage] = field(default_factory=dict)` to `NodeState`; in `update_from_message` add `elif isinstance(msg, EtagereStateMessage): node.etagere_by_zone[msg.zone_id] = msg` before the `DiagnosticsMessage` branch.

`api/routes_etagere.py`:
```python
"""GET /etagere — every node's latest étagère (bin-rack) cell matrices."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from .auth import require_token

router = APIRouter()


def _entry(node_id: str, msg) -> dict:
    matrix = [["unknown"] * msg.cols for _ in range(msg.rows)]
    for c in msg.cells:
        if 1 <= c.r <= msg.rows and 1 <= c.c <= msg.cols:
            matrix[c.r - 1][c.c - 1] = c.state
    return {
        "node_id": node_id, "zone_id": msg.zone_id, "name": msg.name,
        "camera_id": msg.camera_id, "rows": msg.rows, "cols": msg.cols,
        "cells": [c.model_dump() for c in msg.cells], "matrix": matrix, "ts": msg.ts,
    }


def _all(request: Request) -> list[dict]:
    out = []
    for node_id, st in request.app.state.subscriber.snapshot_nodes().items():
        for msg in st.etagere_by_zone.values():
            out.append(_entry(node_id, msg))
    return out


@router.get("/etagere", dependencies=[Depends(require_token)])
async def etagere(request: Request) -> JSONResponse:
    items = _all(request)
    return JSONResponse({"etageres": items, "count": len(items)})


@router.get("/etagere/{zone_id}", dependencies=[Depends(require_token)])
async def etagere_one(zone_id: str, request: Request) -> JSONResponse:
    for e in _all(request):
        if e["zone_id"] == zone_id:
            return JSONResponse(e)
    raise HTTPException(status_code=404, detail=f"unknown étagère {zone_id!r}")
```
`app.py`: import `routes_etagere` with the others and add `routes_etagere.router` to the list that is included under both the version prefix and bare (mirror `routes_zones`).

`api/ui_page.py`: add a card `<div class="card"><h2>Étagères <span class="n" id="n-etag"></span></h2><div id="etag"></div></div>` after the Zones card; add JS:
```js
function renderEtagere(d){
  if(bad(d))return;
  $("n-etag").textContent=d.count;
  const cell=s=>'<td class="et-'+s+'" title="'+s+'"></td>';
  $("etag").innerHTML=(d.etageres||[]).map(e=>
    '<div class="etag"><b>'+esc(e.name||e.zone_id)+'</b> <span class="mut">'+esc(e.node_id)+
    ' · '+esc(e.camera_id)+' · '+ago(e.ts)+'</span><table class="etgrid">'+
    e.matrix.map(row=>'<tr>'+row.map(cell).join('')+'</tr>').join('')+'</table></div>'
  ).join('')||'<div class="empty">— none —</div>';
}
```
CSS in the page's `<style>`: `.etgrid td{width:22px;height:22px;border:1px solid var(--line)} .et-filled{background:#2ea043} .et-empty{background:#444} .et-unknown{background:repeating-linear-gradient(45deg,#333,#333 4px,#555 4px,#555 8px)}`; in `tick()` add `j("/etagere")` to the `Promise.all` list and call `renderEtagere(etag)`.

- [ ] **Step 4: Run tests + lint**

Run: `cd isicomms && pytest -q && ruff check isicomms`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add isicomms/isicomms/mqtt_subscriber.py isicomms/isicomms/api/routes_etagere.py isicomms/isicomms/app.py isicomms/isicomms/api/ui_page.py isicomms/tests/test_routes_etagere.py
git commit -m "feat(isicomms): étagère cache + GET /etagere + /ui matrix card"
```

---

### Task 8: monitor_web — wire cache + `/api/etagere` GET/POST

**Files:**
- Modify: `monitor_web/monitor_web/bus_subscriber.py` (`BusState` ~61-85, dispatch ~268-280, `snapshot` ~171-190)
- Create: `monitor_web/monitor_web/api/routes_etagere.py`
- Modify: `monitor_web/monitor_web/app.py` (include router next to `routes_zone_patches` ~199)
- Test: `monitor_web/tests/test_routes_etagere.py`

**Interfaces:**
- Produces: `BusState.etagere_by_zone: dict[str, EtagereStateMessage]`; `GET /api/etagere` → `{"model": {...}|null, "zones": [...]}` (the YAML as-is, cells included); `POST /api/etagere` body `{"model": {...}|null, "zones": [{id,name,camera,frame_wh,corners,rows,cols,cells}]}` → validates via `EtagereConfig`, atomic write to `resolve_config_path(backbone_cfg, backbone_config_path)`, hot-restarts a running isistream (same closure as `routes_config.py:1094-1112`), returns the saved config; `POST /api/etagere/autosplit` body `{"corners": [[u,v]x4], "rows": 3, "cols": 3}` → `{"cells": [...]}` (server-side `cells_from_corners`, so the JS never re-implements the math); `GET /api/etagere/state` → `{"states": {zone_id: {matrix, cells, ts, name}}}` from the bus.

- [ ] **Step 1: Write the failing tests** (`monitor_web/tests/test_routes_etagere.py`)

```python
from __future__ import annotations

from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from monitor_web.app import create_app
from monitor_web.settings import Settings


def _app(tmp_path: Path):
    backbone_yaml = tmp_path / "backbone.yaml"
    backbone_yaml.write_text(yaml.safe_dump({"cameras": {}, "metadata": {"sinks": []}}))
    return create_app(Settings(backbone_config_path=backbone_yaml, udp_port=0, port=0)), tmp_path


def _cells9():
    return [{"r": r, "c": c, "rect": [c * 10, r * 10, c * 10 + 8, r * 10 + 8]}
            for r in (1, 2, 3) for c in (1, 2, 3)]


def test_get_empty_when_no_file(tmp_path):
    app, _ = _app(tmp_path)
    with TestClient(app) as c:
        r = c.get("/api/etagere")
        assert r.status_code == 200 and r.json() == {"model": None, "zones": []}


def test_autosplit_returns_nine_cells(tmp_path):
    app, _ = _app(tmp_path)
    with TestClient(app) as c:
        r = c.post("/api/etagere/autosplit",
                   json={"corners": [[0, 0], [90, 0], [90, 90], [0, 90]], "rows": 3, "cols": 3})
        cells = r.json()["cells"]
        assert len(cells) == 9 and cells[4]["rect"] == [30.0, 30.0, 60.0, 60.0]


def test_post_writes_yaml_and_roundtrips(tmp_path):
    app, root = _app(tmp_path)
    body = {"model": {"onnx_path": "m.onnx"},
            "zones": [{"id": "et_1", "name": "A", "camera": "cam_a", "frame_wh": [640, 480],
                       "corners": [[0, 0], [90, 0], [90, 90], [0, 90]], "cells": _cells9()}]}
    with TestClient(app) as c:
        r = c.post("/api/etagere", json=body)
        assert r.status_code == 200, r.text
        assert (root / "etagere.yaml").exists()
        saved = yaml.safe_load((root / "etagere.yaml").read_text())
        assert saved["zones"][0]["id"] == "et_1" and len(saved["zones"][0]["cells"]) == 9
        assert c.get("/api/etagere").json()["zones"][0]["name"] == "A"


def test_post_invalid_grid_rejected(tmp_path):
    app, _ = _app(tmp_path)
    body = {"model": {"onnx_path": "m.onnx"},
            "zones": [{"id": "z", "camera": "cam_a", "frame_wh": [640, 480], "cells": _cells9()[:8]}]}
    with TestClient(app) as c:
        assert c.post("/api/etagere", json=body).status_code == 422


def test_state_from_bus(tmp_path):
    from backbone.comms.schemas import EtagereCellState, EtagereStateMessage
    app, _ = _app(tmp_path)
    with TestClient(app) as c:
        bus = c.app.state.bus   # confirm the attribute name used by other route tests (e.g. routes_ws / zone tests)
        msg = EtagereStateMessage(ts=1.0, camera_id="cam_a", zone_id="et_1", name="A",
                                  cells=tuple(EtagereCellState(r=r, c=cc, state="empty")
                                              for r in (1, 2, 3) for cc in (1, 2, 3)),
                                  stabilized=True)
        bus._handle_message(msg)   # use the subscriber's real dispatch entry (see how test_bus_subscriber.py injects)
        st = c.get("/api/etagere/state").json()["states"]["et_1"]
        assert st["matrix"] == [["empty"] * 3] * 3 and st["name"] == "A"
```
Look at `monitor_web/tests/test_bus_subscriber.py` for the exact way messages are injected into the subscriber and the `app.state` attribute name of the bus, and adjust the last test to match.

- [ ] **Step 2: Run to verify failure**

Run: `cd monitor_web && pytest tests/test_routes_etagere.py -v`
Expected: FAIL — 404 on `/api/etagere`.

- [ ] **Step 3: Implement**

`bus_subscriber.py`: import `EtagereStateMessage`; `BusState.etagere_by_zone: dict[str, EtagereStateMessage] = field(default_factory=dict)`; dispatch `elif isinstance(msg, EtagereStateMessage): self._state.etagere_by_zone[msg.zone_id] = msg`; include `etagere_by_zone=dict(self._state.etagere_by_zone)` in `snapshot()`; clear it where `observations_by_camera.clear()` is called (~165).

`api/routes_etagere.py`:
```python
"""Étagère zones — config authoring (GET/POST /api/etagere) + live cell state.

The dashboard AUTHORS config/etagere.yaml; isistream CONSUMES it (per-cell
crop inference). No detector here. A save hot-restarts a running isistream
exactly like the detection-model save does.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ValidationError

from backbone.shared.etagere import (
    EtagereConfig, cells_from_corners, load_etagere_config, resolve_config_path,
)

from .routes_config import _write_yaml_atomic

logger = logging.getLogger(__name__)
router = APIRouter()


def _path(request: Request) -> Path:
    cfg = request.app.state.settings
    by = Path(cfg.backbone_config_path)
    try:
        backbone_cfg = yaml.safe_load(by.read_text()) or {}
    except (OSError, yaml.YAMLError):
        backbone_cfg = {}
    return resolve_config_path(backbone_cfg, by)


@router.get("/api/etagere")
def get_etagere(request: Request) -> dict[str, Any]:
    cfg = load_etagere_config(_path(request))
    return cfg.model_dump(mode="json")


class AutosplitBody(BaseModel):
    corners: list[list[float]]
    rows: int = 3
    cols: int = 3


@router.post("/api/etagere/autosplit")
def autosplit(body: AutosplitBody) -> dict[str, Any]:
    if len(body.corners) != 4:
        raise HTTPException(status_code=422, detail="corners must have exactly 4 points")
    cells = cells_from_corners(body.corners, body.rows, body.cols)
    return {"cells": [c.model_dump(mode="json") for c in cells]}


@router.post("/api/etagere")
def post_etagere(body: dict[str, Any], request: Request) -> dict[str, Any]:
    try:
        cfg = EtagereConfig.model_validate(body)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors()) from e
    _write_yaml_atomic(_path(request), cfg.model_dump(mode="json"))
    host = getattr(request.app.state, "isistream", None)
    if host is not None and host.points_mode() and host.status().get("running"):
        def _restart() -> None:
            try:
                host.stop()
                host.start()
            except Exception:
                logger.warning("etagere: isistream restart failed", exc_info=True)
        threading.Thread(target=_restart, name="etagere-isistream-restart", daemon=True).start()
    return cfg.model_dump(mode="json")


@router.get("/api/etagere/state")
def etagere_state(request: Request) -> dict[str, Any]:
    bus = request.app.state.bus    # same attribute other routes use for the BusSubscriber
    snap = bus.snapshot()
    states: dict[str, Any] = {}
    for zone_id, msg in snap.etagere_by_zone.items():
        matrix = [["unknown"] * msg.cols for _ in range(msg.rows)]
        for c in msg.cells:
            if 1 <= c.r <= msg.rows and 1 <= c.c <= msg.cols:
                matrix[c.r - 1][c.c - 1] = c.state
        states[zone_id] = {"name": msg.name, "camera_id": msg.camera_id, "rows": msg.rows,
                           "cols": msg.cols, "matrix": matrix,
                           "cells": [c.model_dump() for c in msg.cells], "ts": msg.ts}
    return {"states": states}
```
`_write_yaml_atomic` is defined in `routes_config.py:434`; if importing it creates a circular import, move it to `monitor_web/monitor_web/yaml_io.py` and import from there in both files. `app.py`: `app.include_router(routes_etagere.router)` next to `routes_zone_patches`. Confirm the bus attribute name (`request.app.state.bus`?) by grepping `app.state.` in `app.py` for the `BusSubscriber` instance and use that name.

- [ ] **Step 4: Run tests + lint**

Run: `cd monitor_web && pytest tests/test_routes_etagere.py tests/test_bus_subscriber.py -v && ruff check monitor_web`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add monitor_web/monitor_web/bus_subscriber.py monitor_web/monitor_web/api/routes_etagere.py monitor_web/monitor_web/app.py monitor_web/tests/test_routes_etagere.py
git commit -m "feat(monitor_web): /api/etagere config authoring + autosplit + live cell state"
```

---

### Task 9: monitor_web UI — Settings editor, cam overlay, matrix widget

**Files:**
- Create: `monitor_web/monitor_web/static/js/etagere.js`
- Modify: `monitor_web/monitor_web/templates/dashboard.html` (Zones tab: after the "Camera zones" section ~451-456; script tags where `zone_patch.js` is loaded)
- Modify: `monitor_web/monitor_web/static/js/live_overlay.js` (`drawForCam` ~128-147)
- Modify: `monitor_web/monitor_web/static/js/comms_nodes.js` (COMMUNICATION cards; find where zone-patch cards are rendered and add a matrix widget per étagère)
- Modify: `monitor_web/monitor_web/static/css/*.css` (whichever holds `.config-zone-list` styles) — add `.etag-grid`, `.etag-cell`, `.et-filled/.et-empty/.et-unknown`
- Test: `monitor_web/tests/test_etagere_js.mjs` (Node-free: run with `node` like `test_draw_rect_snap.mjs`; check how that file is executed in the suite — if it's run manually, do the same)

**Interfaces:**
- Consumes: `GET/POST /api/etagere`, `POST /api/etagere/autosplit`, `GET /api/etagere/state` (Task 8); `startDraw({target: cam, mode:"raw", minPoints:4, maxPoints:4, onDone})` from `draw_mode.js`.
- Produces: `window.__etagere = { startEtagereDraw(camId), deleteZone(id), getZones(camId), getStates() }`; `etagere.js` exports `getZones(camId)` and `getStates()` for `live_overlay.js`; pure helpers exported for tests: `hitTest(zone, x, y) -> {cellIdx, handle: "move"|"tl"|"tr"|"br"|"bl"|null}` and `applyDrag(rect, handle, dx, dy) -> rect`.

- [ ] **Step 1: Write the failing JS unit test** (`monitor_web/tests/test_etagere_js.mjs`, run with `node`)

```js
import assert from "node:assert/strict";
import { applyDrag, hitTest } from "../monitor_web/static/js/etagere.js";

// move: whole rect translates
assert.deepEqual(applyDrag([10, 10, 50, 50], "move", 5, -5), [15, 5, 55, 45]);
// corner drags: only that corner moves; never inverted (min 4 px)
assert.deepEqual(applyDrag([10, 10, 50, 50], "br", 10, 10), [10, 10, 60, 60]);
assert.deepEqual(applyDrag([10, 10, 50, 50], "tl", 100, 100), [46, 46, 50, 50]);
// hit-test: corner handle within 8 px wins over move
const zone = { cells: [{ r: 1, c: 1, rect: [10, 10, 50, 50] }, { r: 1, c: 2, rect: [60, 10, 100, 50] }] };
assert.deepEqual(hitTest(zone, 49, 49, 8), { cellIdx: 0, handle: "br" });
assert.deepEqual(hitTest(zone, 30, 30, 8), { cellIdx: 0, handle: "move" });
assert.deepEqual(hitTest(zone, 80, 30, 8), { cellIdx: 1, handle: "move" });
assert.deepEqual(hitTest(zone, 200, 200, 8), { cellIdx: -1, handle: null });
console.log("etagere.js helpers OK");
```
`etagere.js` must therefore keep `applyDrag`/`hitTest` free of DOM access at import time (guard all DOM setup behind `if (typeof document !== "undefined")`).

- [ ] **Step 2: Run to verify failure**

Run: `cd monitor_web && node tests/test_etagere_js.mjs`
Expected: FAIL — cannot find module.

- [ ] **Step 3: Implement**

`static/js/etagere.js` (module):
```js
// Étagère zones — Settings editor (4 corners → auto-split → drag-adjust),
// overlay data source, live cell states. Authoring only: NO inference here.
import { startDraw } from "/static/js/draw_mode.js";

const HANDLE_PX = 8;
let cfg = { model: null, zones: [] };
let states = {};              // zone_id → {matrix, cells, ts, name}
let editing = null;           // {zoneId} while the drag-adjust overlay is active

export function getZones(camId) { return (cfg.zones || []).filter((z) => !camId || z.camera === camId); }
export function getStates() { return states; }

export function applyDrag(rect, handle, dx, dy) {
  let [x0, y0, x1, y1] = rect;
  if (handle === "move") return [x0 + dx, y0 + dy, x1 + dx, y1 + dy];
  if (handle === "tl") { x0 = Math.min(x0 + dx, x1 - 4); y0 = Math.min(y0 + dy, y1 - 4); }
  if (handle === "tr") { x1 = Math.max(x1 + dx, x0 + 4); y0 = Math.min(y0 + dy, y1 - 4); }
  if (handle === "br") { x1 = Math.max(x1 + dx, x0 + 4); y1 = Math.max(y1 + dy, y0 + 4); }
  if (handle === "bl") { x0 = Math.min(x0 + dx, x1 - 4); y1 = Math.max(y1 + dy, y0 + 4); }
  return [x0, y0, x1, y1];
}

export function hitTest(zone, x, y, tol = HANDLE_PX) {
  const cells = zone.cells || [];
  for (let i = cells.length - 1; i >= 0; i--) {          // top-most first
    const [x0, y0, x1, y1] = cells[i].rect;
    const corners = { tl: [x0, y0], tr: [x1, y0], br: [x1, y1], bl: [x0, y1] };
    for (const [h, [cx, cy]] of Object.entries(corners)) {
      if (Math.abs(cx - x) <= tol && Math.abs(cy - y) <= tol) return { cellIdx: i, handle: h };
    }
  }
  for (let i = cells.length - 1; i >= 0; i--) {
    const [x0, y0, x1, y1] = cells[i].rect;
    if (x >= x0 && x <= x1 && y >= y0 && y <= y1) return { cellIdx: i, handle: "move" };
  }
  return { cellIdx: -1, handle: null };
}

async function load() {
  const r = await fetch("/api/etagere");
  if (r.ok) cfg = await r.json();
  renderSettingsList();
}
async function save() {
  const r = await fetch("/api/etagere", { method: "POST",
    headers: { "Content-Type": "application/json" }, body: JSON.stringify(cfg) });
  if (!r.ok) window.alert("Étagère save failed: " + (await r.text()));
  renderSettingsList();
}
async function pollStates() {
  try { const r = await fetch("/api/etagere/state"); if (r.ok) states = (await r.json()).states || {}; }
  catch (_) { /* keep last */ }
}

export function startEtagereDraw(camId) {
  const cam = camId || (document.querySelector(".zm-draw-target-btn.active")?.dataset.target) || "cam_a";
  const img = document.getElementById(`${cam}-img`);
  startDraw({
    target: cam, mode: "raw", minPoints: 4, maxPoints: 4,
    label: "Étagère — click the rack's 4 outer corners (TL, TR, BR, BL)",
    onDone: async (points) => {
      if (!points || points.length !== 4) return;
      const name = window.prompt("Étagère name", `Étagère ${cfg.zones.length + 1}`);
      if (!name) return;
      const r = await fetch("/api/etagere/autosplit", { method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ corners: points, rows: 3, cols: 3 }) });
      const cells = (await r.json()).cells;
      cfg.zones.push({ id: "et_" + Date.now().toString(36), name, camera: cam,
        frame_wh: [img?.naturalWidth || 0, img?.naturalHeight || 0],
        corners: points, rows: 3, cols: 3, cells });
      await save();
      startAdjust(cfg.zones[cfg.zones.length - 1].id);
    },
  });
}

// Drag-adjust: mouse events on the cam overlay canvas, source-px via the same
// letterbox mapping live_overlay.js uses (exposed as window.__displayToSource).
export function startAdjust(zoneId) {
  const zone = cfg.zones.find((z) => z.id === zoneId);
  if (!zone) return;
  editing = { zoneId };
  const canvas = document.getElementById(`${zone.camera}-overlay`);
  if (!canvas) return;
  let drag = null;
  const toSrc = (ev) => window.__displayToSource(canvas, zone.camera, ev.offsetX, ev.offsetY, zone.frame_wh);
  const down = (ev) => { const p = toSrc(ev); const h = hitTest(zone, p[0], p[1]);
    if (h.cellIdx >= 0) drag = { ...h, last: p }; };
  const move = (ev) => { if (!drag) return; const p = toSrc(ev);
    zone.cells[drag.cellIdx].rect = applyDrag(zone.cells[drag.cellIdx].rect, drag.handle,
      p[0] - drag.last[0], p[1] - drag.last[1]); drag.last = p; };
  const up = () => { drag = null; };
  canvas.addEventListener("mousedown", down); canvas.addEventListener("mousemove", move);
  window.addEventListener("mouseup", up);
  window.__etagereStopAdjust = async () => {
    canvas.removeEventListener("mousedown", down); canvas.removeEventListener("mousemove", move);
    window.removeEventListener("mouseup", up); editing = null; await save();
  };
  canvas.style.pointerEvents = "auto";
}
export function isEditing(zoneId) { return editing && editing.zoneId === zoneId; }

export function deleteZone(id) { cfg.zones = cfg.zones.filter((z) => z.id !== id); save(); }

function renderSettingsList() {
  const el = document.getElementById("zm-etagere-list");
  if (!el) return;
  el.innerHTML = (cfg.zones || []).map((z) => `
    <div class="config-zone-row">
      <span class="zone-name">${z.name}</span><span class="cam-badge">${z.camera}</span>
      <span class="mut">${z.rows}×${z.cols}</span>
      <button class="glass-btn btn-small" data-adjust="${z.id}">Adjust cells</button>
      <button class="glass-btn btn-small" data-done="${z.id}">Done</button>
      <button class="glass-btn btn-icon" data-del="${z.id}" aria-label="Delete">✕</button>
    </div>`).join("") || '<div class="mut">No étagère yet — draw one on a camera.</div>';
  el.querySelectorAll("[data-adjust]").forEach((b) => b.onclick = () => startAdjust(b.dataset.adjust));
  el.querySelectorAll("[data-done]").forEach((b) => b.onclick = () => window.__etagereStopAdjust?.());
  el.querySelectorAll("[data-del]").forEach((b) => b.onclick = () => deleteZone(b.dataset.del));
}

if (typeof document !== "undefined") {
  document.addEventListener("DOMContentLoaded", () => {
    load();
    document.getElementById("zm-etagere-add")?.addEventListener("click", () => startEtagereDraw());
    setInterval(pollStates, 1000);
  });
  window.__etagere = { startEtagereDraw, deleteZone, getZones, getStates, startAdjust };
}
```
`live_overlay.js`: export a helper `window.__displayToSource(canvas, camId, dx, dy, frameWh)` that inverts `sourceToDisplay` (using `naturalSize` for the cam and scaling from natural to `frame_wh`); import `{ getZones, getStates, isEditing }` from `/static/js/etagere.js`; add `drawEtagere(ctx, box, img, camId)` called from `drawForCam` after `drawPatchGhosts`: for each zone of `camId`, map each cell rect (frame_wh → natural → display via `sourceToDisplay`) and stroke it; fill 25 % alpha by state from `getStates()[zone.id].matrix[r-1][c-1]` (`filled` green `#2ea043`, `empty` grey `#9aa0a6`, `unknown` orange dashed); label `r{r}c{c}`; when `isEditing(zone.id)` draw 6-px corner handles.
`templates/dashboard.html`: after the "Camera zones" section add
```html
<section class="config-section">
  <h3 data-i18n="etagere_section">{{ t.etagere_section or 'Étagères (bin racks, drawn on CAM)' }}</h3>
  <div id="zm-etagere-list" class="config-zone-list"></div>
  <button id="zm-etagere-add" class="glass-btn config-link-add" type="button"
          data-i18n="etagere_add">{{ t.etagere_add or '+ Draw étagère (4 corners) on camera' }}</button>
</section>
```
and load `<script type="module" src="/static/js/etagere.js"></script>` next to `zone_patch.js`. Add `etagere_section` / `etagere_add` keys to both i18n dictionaries (GB/FR — find where `cam_zones_section` is defined and add beside it).
`comms_nodes.js`: where the zone-patch COMMUNICATION cards are rendered, append a card per entry of `window.__etagere?.getStates()` with a 3×3 `<table class="etag-grid">` of `<td class="et-{state}">`.
CSS: `.etag-grid td{width:20px;height:20px;border:1px solid rgba(255,255,255,.25)} .et-filled{background:#2ea043} .et-empty{background:#666} .et-unknown{background:repeating-linear-gradient(45deg,#444,#444 4px,#777 4px,#777 8px)}`.

- [ ] **Step 4: Run tests**

Run: `cd monitor_web && node tests/test_etagere_js.mjs && pytest -q`
Expected: `etagere.js helpers OK` and the Python suite green (template still renders — `tests/test_i18n.py` covers key parity; add the two keys to both languages).

- [ ] **Step 5: Manual check** (dev instance, `MONITOR_WEB_PORT=8100`): open Settings → Zones → "+ Draw étagère" → click 4 corners on CAM 1 → 9 cells appear → "Adjust cells" → drag a corner → "Done" → `cat config/etagere.yaml` shows the edited rects.

- [ ] **Step 6: Commit**

```bash
git add monitor_web/monitor_web/static/js/etagere.js monitor_web/monitor_web/static/js/live_overlay.js monitor_web/monitor_web/static/js/comms_nodes.js monitor_web/monitor_web/templates/dashboard.html monitor_web/monitor_web/static/css monitor_web/monitor_web/i18n* monitor_web/tests/test_etagere_js.mjs
git commit -m "feat(monitor_web): étagère Settings editor (4 corners → auto-split → drag-adjust), cam overlay, matrix widget"
```

---

### Task 10: Hermetic end-to-end + docs

**Files:**
- Test: `tests/test_etagere_e2e.py`
- Modify: `CLAUDE.md` (Zone-scoped detection paragraph — add an "Étagère zones" note), `docs/REUSE.md` (wire contract: `etagere_state`), `config/backbone.yaml` — add a commented `etagere:` block (`config_path`, `stabilize_window`, `stabilize_flip_ratio`, `unknown_after_s`, `heartbeat_s`) — **do not** otherwise touch the live values in that file.

**Interfaces:** consumes everything above.

- [ ] **Step 1: Write the e2e test** (`tests/test_etagere_e2e.py`) — no ONNX, no CUDA: drive `EtagereDetector` with a fake detector into a real `DetectionIngest` + `EtagereStateTracker` + `Publisher([UdpSink])`, and read the stabilised message off a loopback UDP socket.

```python
from __future__ import annotations

import json
import socket
import time

import numpy as np

from backbone.comms.publisher import Publisher
from backbone.comms.schemas import parse_envelope
from backbone.comms.udp_sink import UdpSink
from backbone.core.types import Detection, Frame
from backbone.ingestion.points_in import DetectionIngest
from backbone.shared.etagere import EtagereCell, EtagereConfig, EtagereModel, EtagereZone
from backbone.shared.etagere_state import EtagereStateTracker
from isistream.etagere import EtagereDetector


class _AllFilled:
    def detect(self, pair):
        return {k: [Detection(camera_id=k, capture_ts=0.0, cls="filled_box", confidence=0.95,
                              bbox_xyxy=(0, 0, 10, 10), foot_uv=(5, 10), keypoints_uv=None)]
                for k in pair.frames}


def test_producer_to_bus_roundtrip() -> None:
    out_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    out_sock.bind(("127.0.0.1", 0)); out_sock.settimeout(3.0)
    out_port = out_sock.getsockname()[1]
    sink = UdpSink(host="127.0.0.1", port=out_port)     # match UdpSink's ctor kwargs
    pub = Publisher([sink])
    tracker = EtagereStateTracker()
    ingest = DetectionIngest(["cam_a"], port=0, on_set=lambda s: None,
                             on_etagere=lambda m: (lambda o: pub.publish_etagere_state(o) if o else None)(tracker.update(m)))
    ingest.start()
    try:
        cells = [EtagereCell(r=r, c=c, rect=(c * 50, r * 50, c * 50 + 40, r * 50 + 40))
                 for r in (1, 2, 3) for c in (1, 2, 3)]
        cfg = EtagereConfig(model=EtagereModel(onnx_path="x"),
                            zones=(EtagereZone(id="et_1", name="A", camera="cam_a",
                                               frame_wh=(320, 240), cells=tuple(cells)),))
        det = EtagereDetector(cfg, _AllFilled(), producer_id="test")
        frame = Frame(camera_id="cam_a", capture_ts=1.0, frame_idx=0,
                      image=np.zeros((240, 320, 3), np.uint8))
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        for msg in det.run({"cam_a": frame}, now=0.0):
            s.sendto(msg.model_dump_json().encode(), ("127.0.0.1", ingest.port))
        data, _ = out_sock.recvfrom(65535)
        got = parse_envelope(json.loads(data.decode()))
        assert got.type.value == "etagere_state" and got.stabilized is True
        assert [c.state for c in got.cells] == ["filled"] * 9
        assert got.zone_id == "et_1" and got.producer_id == "test"
    finally:
        ingest.stop(); pub.close(); out_sock.close()
```
Adjust `UdpSink(...)` kwargs and `Publisher(...)` ctor to what `tests/test_udp_sink.py` / `tests/test_publisher.py` use.

- [ ] **Step 2: Run** — `pytest tests/test_etagere_e2e.py -v` → PASS.

- [ ] **Step 3: Docs**
- `CLAUDE.md`, in the "Zone-scoped detection" bullet area, add:
  > **Étagère zones (2026-08-17).** `config/etagere.yaml` (dashboard-authored, isistream-consumed) defines per-camera image-space 3×3 cell grids for bin racks. isistream crops each cell (rect + 8 % margin), batches through the 2-class `yolo26n@320` (`empty_box`/`filled_box`, end-to-end head — see `is_end2end_detect_output`) and emits raw `etagere_state` messages on the points ingest port; the Backbone stabilises per cell (`EtagereStateTracker`, flip ≥ 70 % of a 15-vote window, unknown decay 5 s, heartbeat 5 s) and publishes on UDP + MQTT (`{prefix}/etagere/{zone_id}`, retained). Not floor-based: never in `zones.yaml`. Dashboard: Settings → Étagères (4 corners → auto-split → drag-adjust), cam overlay + COMMUNICATION matrix; isicomms `GET /etagere` + /ui card. Tools: `trainer/isidet/scripts/{grid_click,etagere_dataset}.py`, config `configs/train_etagere.yaml`.
- `docs/REUSE.md`: add `etagere_state` to the wire-contract table (topic, retained, payload summary).
- `config/backbone.yaml`: append a commented block:
  ```yaml
  # etagere:                       # bin-rack cell occupancy (config/etagere.yaml beside this file)
  #   config_path: config/etagere.yaml
  #   stabilize_window: 15
  #   stabilize_flip_ratio: 0.7
  #   unknown_after_s: 5.0
  #   heartbeat_s: 5.0
  ```
  (Note: `backbone.yaml` is rewritten by the Settings modal and comments do not survive a save — that's acceptable; the defaults are in code.)

- [ ] **Step 4: Full suites + lint**

Run: `pytest -q && (cd monitor_web && pytest -q) && (cd isicomms && pytest -q) && ruff check backbone calibration tests isistream monitor_web isicomms`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add tests/test_etagere_e2e.py CLAUDE.md docs/REUSE.md config/backbone.yaml
git commit -m "test(etagere): hermetic producer→ingest→tracker→UDP round-trip; docs"
```

---

### Task 11: Live verification (manual, on the rig)

- [ ] Rebuild isicomms if it runs in docker: `docker compose -p on-prem build gateway` (rebuild, not restart).
- [ ] Dashboard (`3d`, :8000): Settings → Zones → "+ Draw étagère" on CAM 1 → 4 corners → adjust → Done. Confirm `config/etagere.yaml` has 9 cells and isistream restarted (`/api/status` shows the producer running; log line "étagère detector ready").
- [ ] `mosquitto_sub -h <broker> -t 'isiMonitor3D/v1/+/etagere/#' -v` shows a retained matrix; move a box in/out of a bin → the cell flips within ~ one vote window (≤ 15 raw ticks at max_fps 2 ⇒ ≤ ~8 s; lower `max_fps`/`stabilize_window` if too slow).
- [ ] isicomms `/ui` Étagères card shows the same matrix; `GET /etagere` returns it.
- [ ] Note the measured `stage_ms["etagere"]` from isistream's status; expect single-digit ms on GPU for 9 crops.
