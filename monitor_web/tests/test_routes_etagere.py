"""Étagère config authoring (GET/POST /api/etagere) + autosplit + live state.

Mirrors the conventions in test_routes_zone_patches-style route tests: a
hermetic FastAPI app pointed at a tmp_path backbone.yaml, no live Backbone or
isistream required.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from monitor_web.app import create_app
from monitor_web.config import Settings


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


def test_autosplit_rejects_bad_corner_count(tmp_path):
    app, _ = _app(tmp_path)
    with TestClient(app) as c:
        r = c.post("/api/etagere/autosplit",
                   json={"corners": [[0, 0], [90, 0], [90, 90]], "rows": 3, "cols": 3})
        assert r.status_code == 422


def test_autosplit_rejects_malformed_point(tmp_path):
    # A point with the wrong element count (3 instead of 2) must 422 via
    # pydantic validation, not reach cells_from_corners and bare-500 on a
    # ValueError ("too many values to unpack").
    app, _ = _app(tmp_path)
    with TestClient(app) as c:
        r = c.post("/api/etagere/autosplit",
                   json={"corners": [[0, 0], [90, 0], [90, 90], [0, 90, 5]],
                         "rows": 3, "cols": 3})
        assert r.status_code == 422


def test_autosplit_rejects_zero_rows(tmp_path):
    app, _ = _app(tmp_path)
    with TestClient(app) as c:
        r = c.post("/api/etagere/autosplit",
                   json={"corners": [[0, 0], [90, 0], [90, 90], [0, 90]],
                         "rows": 0, "cols": 3})
        assert r.status_code == 422


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
        bus = c.app.state.bus   # attribute name confirmed via app.py / test_bus_subscriber.py
        msg = EtagereStateMessage(ts=1.0, camera_id="cam_a", zone_id="et_1", name="A",
                                  cells=tuple(EtagereCellState(r=r, c=cc, state="empty")
                                              for r in (1, 2, 3) for cc in (1, 2, 3)),
                                  stabilized=True)
        # BusSubscriber's real dispatch entry point is _handle_payload(bytes) —
        # see test_bus_subscriber.py, which injects the same way.
        bus._handle_payload(msg.model_dump_json().encode("utf-8"))
        st = c.get("/api/etagere/state").json()["states"]["et_1"]
        assert st["matrix"] == [["empty"] * 3] * 3 and st["name"] == "A"


def test_state_empty_when_no_messages(tmp_path):
    app, _ = _app(tmp_path)
    with TestClient(app) as c:
        assert c.get("/api/etagere/state").json() == {"states": {}}
