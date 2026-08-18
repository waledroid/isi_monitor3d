from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from backbone.shared.etagere import (
    EtagereCell,
    EtagereConfig,
    EtagereModel,
    EtagereZone,
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


# ---- C1: relative onnx_path resolves against the repo root ----

def test_relative_onnx_path_resolves_against_repo_root(tmp_path: Path) -> None:
    # <root>/config/etagere.yaml + <root>/models/x.onnx (mirrors backbone.yaml's
    # own layout convention: config/ sits directly under the repo root).
    root = tmp_path
    (root / "config").mkdir()
    (root / "models").mkdir()
    (root / "models" / "x.onnx").write_bytes(b"")
    yaml_path = root / "config" / "etagere.yaml"
    yaml_path.write_text(yaml.safe_dump({
        "model": {"onnx_path": "models/x.onnx"},
        "zones": [{"id": "et_1", "camera": "cam_a", "frame_wh": [1920, 1080],
                   "cells": _cells9()}],
    }))
    cfg = load_etagere_config(yaml_path)
    assert cfg.model.onnx_path == str((root / "models" / "x.onnx").resolve())
    assert Path(cfg.model.onnx_path).is_absolute()


def test_absolute_onnx_path_untouched(tmp_path: Path) -> None:
    abs_path = "/some/absolute/models/x.onnx"
    cfg = load_etagere_config(_cfg(tmp_path, model={"onnx_path": abs_path}))
    assert cfg.model.onnx_path == abs_path


def test_committed_config_onnx_path_is_absolute() -> None:
    # Do NOT assert the file exists — CI machines won't have the trained
    # artifact; only that a relative path in the repo's shipped config can
    # never again silently no-op a live isistream (C1).
    repo_root = Path(__file__).resolve().parent.parent
    cfg = load_etagere_config(repo_root / "config" / "etagere.yaml")
    assert cfg.model is not None
    assert Path(cfg.model.onnx_path).is_absolute()


# ---- I2: a 0x0 frame_wh must be rejected, not ZeroDivisionError at inference ----

def test_zero_frame_wh_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        EtagereZone(id="z", camera="cam_a", frame_wh=(0, 0), cells=tuple(_cells9_models()))
    with pytest.raises(ValidationError):
        EtagereZone(id="z", camera="cam_a", frame_wh=(1920, 0), cells=tuple(_cells9_models()))
    with pytest.raises(ValidationError):
        load_etagere_config(_cfg(tmp_path, zones=[{
            "id": "z", "camera": "cam_a", "frame_wh": [0, 0], "cells": _cells9()}]))


def _cells9_models():
    return [EtagereCell(**c) for c in _cells9()]


# ---- I4: class_names must cover the labels decide() matches on ----

def test_class_names_must_include_decide_labels(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        EtagereModel(onnx_path="x.onnx", class_names=["palette", "carton"])
    with pytest.raises(ValidationError):
        load_etagere_config(_cfg(tmp_path, model={
            "onnx_path": "x.onnx", "class_names": ["empty_box"]}))
    # both present, in any order/superset, is fine
    EtagereModel(onnx_path="x.onnx", class_names=["filled_box", "empty_box", "extra"])


def test_cell_angle_deg_defaults_zero_and_is_bounded(tmp_path: Path) -> None:
    from backbone.shared.etagere import EtagereCell
    assert EtagereCell(r=1, c=1, rect=(0, 0, 10, 10)).angle_deg == 0.0
    assert EtagereCell(r=1, c=1, rect=(0, 0, 10, 10), angle_deg=-12.5).angle_deg == -12.5
    with pytest.raises(ValidationError):
        EtagereCell(r=1, c=1, rect=(0, 0, 10, 10), angle_deg=200)
    cells = _cells9()
    cells[4]["angle_deg"] = 7
    cfg = load_etagere_config(_cfg(tmp_path, zones=[{
        "id": "z", "camera": "cam_a", "frame_wh": [640, 480], "cells": cells}]))
    assert cfg.zones[0].cells[4].angle_deg == 7.0 and cfg.zones[0].cells[0].angle_deg == 0.0
