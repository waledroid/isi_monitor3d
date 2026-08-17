from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from backbone.shared.etagere import (
    EtagereCell,
    EtagereConfig,
    cells_from_corners,
    load_etagere_config,
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
    swapped = _cells9()
    swapped[0], swapped[1] = swapped[1], swapped[0]
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
