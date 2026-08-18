"""Étagère ZONE-panel stream: rows x cols mosaic of the warped cell crops."""
from __future__ import annotations

import numpy as np
import pytest
from backbone.shared.etagere import EtagereCell, EtagereZone

from monitor_web.api.routes_video import _ETAGERE_TILE_PX, _etagere_mosaic


def _zone() -> EtagereZone:
    cells = [EtagereCell(r=r, c=c, rect=(c * 100, r * 100, c * 100 + 80, r * 100 + 60))
             for r in (1, 2, 3) for c in (1, 2, 3)]
    return EtagereZone(id="et_1", camera="cam_a", frame_wh=(640, 480), cells=tuple(cells))


def test_mosaic_shape_and_state_borders() -> None:
    img = np.zeros((480, 640, 3), np.uint8)
    img[:, :, 1] = 200                                   # greenish scene
    states = [["filled", "empty", "unknown"]] * 3
    m = _etagere_mosaic(img, _zone(), 0.08, states)
    t = _ETAGERE_TILE_PX
    assert m.shape == (3 * t, 3 * t, 3)
    # top-left tile border = filled (BGR 67,160,46); top-middle = empty grey
    assert tuple(m[1, t // 2]) == (67, 160, 46)
    assert tuple(m[1, t + t // 2]) == (166, 160, 154)
    # interior of a tile shows the letterboxed crop (scene colour), not border
    assert m[t // 2, t // 2, 1] == 200


def test_mosaic_without_states_and_with_rotated_cell() -> None:
    img = np.zeros((480, 640, 3), np.uint8)
    z = _zone()
    cells = list(z.cells)
    cells[4] = EtagereCell(r=2, c=2, rect=cells[4].rect, angle_deg=15.0)
    z = EtagereZone(id="et_1", camera="cam_a", frame_wh=(640, 480), cells=tuple(cells))
    m = _etagere_mosaic(img, z, 0.08, None)
    t = _ETAGERE_TILE_PX
    assert m.shape == (3 * t, 3 * t, 3)
    assert tuple(m[1, t // 2]) == (40, 160, 240)         # unknown (orange) when no states


def test_ws_stream_id_dispatch(monkeypatch) -> None:
    from monitor_web.api import routes_ws_video as ws
    called = {}
    monkeypatch.setattr(ws, "build_etagere_stream", lambda state, zid: called.setdefault("zid", zid))
    ws._build_stream(object(), "etagere:et_9")
    assert called["zid"] == "et_9"
    with pytest.raises(LookupError):
        ws._build_stream(object(), "etagere:")
