"""Zone-panel fill-dim overlay (`show_zone_fill` UI pref).

The producer blanks crop pixels outside the zone polygon before inference
(`detection.zone_crop_polygon_fill`); the panels render raw bus pixels, so
that blind area is invisible to the operator. With the Display pref ON the
zone panel darkens exactly that region — the panel then shows the detector's
true field of view. Off by default; display-only.
"""

from types import SimpleNamespace

import numpy as np
import yaml

from monitor_web.api.routes_video import _zone_render_iter

_GRAY = 200


def _run_one(tmp_path, *, pref_on: bool) -> np.ndarray:
    ui = tmp_path / f"ui_{pref_on}.yaml"
    ui.write_text(yaml.safe_dump({"show_zone_fill": bool(pref_on)}))
    cfg = SimpleNamespace(ui_settings_path=str(ui))
    frame = np.full((200, 200, 3), _GRAY, dtype=np.uint8)
    # Crop rect (40,40)-(160,160); diamond zone polygon inside it — the crop's
    # corner triangles are outside the polygon (what the producer gray-fills).
    diamond = np.array([[100, 40], [160, 100], [100, 160], [40, 100]], float)
    it = _zone_render_iter(
        iter([frame]), cfg, "cam_a", [40, 40, 160, 160], (200, 200),
        display_px=320, is_running=lambda: True, get_dets=lambda: [],
        fill_poly_calib=diamond, calib_wh=(200, 200))
    return next(it)


def test_pref_on_dims_outside_polygon_keeps_inside(tmp_path) -> None:
    crop = _run_one(tmp_path, pref_on=True)
    assert crop.shape[:2] == (120, 120)
    # Crop corner = outside the diamond → darkened (200 * 0.45 = 90).
    assert (crop[2, 2] == int(_GRAY * 0.45)).all()
    # Diamond centre (frame 100,100 → crop 60,60) → untouched.
    assert (crop[60, 60] == _GRAY).all()


def test_pref_off_leaves_panel_untouched(tmp_path) -> None:
    crop = _run_one(tmp_path, pref_on=False)
    assert (crop == _GRAY).all()
