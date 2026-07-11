"""One palette, two renderers.

Zone panels are drawn server-side (Python/OpenCV); the big cam views are drawn
client-side (JS/canvas). When the two tables disagree, the SAME pallet shows
green in one view and blue in another — which is exactly what happened. The
server owns the palette and serves it on /api/ui-settings; the JS fallback
must be identical.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from monitor_web.overlay import (
    CLASS_COLORS_HEX,
    DEFAULT_CLASS_COLOR_HEX,
    _color_for,
    _hex_to_bgr,
)

_JS = (Path(__file__).resolve().parents[1]
       / "monitor_web" / "static" / "js" / "passthrough_player.js")


def _js_fallback_palette() -> dict[str, str]:
    src = _JS.read_text()
    block = re.search(r"const FALLBACK_CLASS_COLORS = \{(.*?)\};", src, re.S)
    assert block, "FALLBACK_CLASS_COLORS table not found in passthrough_player.js"
    return {m.group(1): m.group(2)
            for m in re.finditer(r'(\w+):\s*"(#[0-9a-fA-F]{6})"', block.group(1))}


def _js_default_color() -> str:
    src = _JS.read_text()
    m = re.search(r'const DEFAULT_CLASS_COLOR = "(#[0-9a-fA-F]{6})";', src)
    assert m, "DEFAULT_CLASS_COLOR not found"
    return m.group(1)


def test_js_fallback_palette_matches_python():
    assert _js_fallback_palette() == CLASS_COLORS_HEX
    assert _js_default_color() == DEFAULT_CLASS_COLOR_HEX


def test_python_draws_the_hex_palette_as_bgr():
    # cv2 is BGR: the green pallet must be (b, g, r) = (0x50, 0xdc, 0x50).
    assert _color_for("palette") == _hex_to_bgr(CLASS_COLORS_HEX["palette"])
    assert _color_for("PALETTE") == _color_for("palette")     # case-insensitive
    assert _hex_to_bgr("#ff0000") == (0, 0, 255)              # red → BGR


def test_pallet_alias_shares_one_colour():
    assert CLASS_COLORS_HEX["pallet"] == CLASS_COLORS_HEX["palette"]


def test_ui_settings_serves_the_palette(tmp_path):
    from fastapi.testclient import TestClient

    from monitor_web.app import create_app
    from monitor_web.config import Settings

    cfg = Settings(backbone_config_path=tmp_path / "b.yaml",
                   ui_settings_path=tmp_path / "ui.yaml", udp_port=0, port=0)
    with TestClient(create_app(cfg)) as client:
        body = json.loads(client.get("/api/ui-settings").text)
    assert body["class_colors"] == CLASS_COLORS_HEX
    assert body["class_color_default"] == DEFAULT_CLASS_COLOR_HEX
