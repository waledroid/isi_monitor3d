"""Floor-anchor wording: the ChArUco floor board must be told to lay FLAT.

The floor ChArUco defines the world ground plane (Z=0). Leaning it tilts that
plane and breaks the metric homography, so every floor-shot instruction must say
FLAT and must NOT tell the operator to lean it at an angle.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from isical.capture import session as sess_mod

_BASE = Path(__file__).resolve().parent.parent
_CAPTURE_HTML = _BASE / "templates" / "capture.html"


def _floor_strings() -> str:
    """All floor-relevant text: the capture template + the session module source."""
    return _CAPTURE_HTML.read_text() + "\n" + inspect.getsource(sess_mod)


def test_floor_tips_say_flat():
    text = _floor_strings().lower()
    assert "flat on the floor" in text


def test_floor_tips_do_not_say_lean_or_angle():
    text = _floor_strings().lower()
    assert "lean" not in text
    en_dash = chr(0x2013)                              # U+2013, avoid the literal in source
    for ang in ("20-40", f"20{en_dash}40", "~20"):     # hyphen + en-dash + tilde variants
        assert ang not in text
