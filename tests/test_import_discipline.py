"""Lean-image import discipline for the consumer-side backbone surface.

The isicomms (gateway) Docker image is deliberately lean — no CUDA / OpenCV /
GStreamer / calibration stack. All it needs from the backbone is
``backbone.comms.schemas`` and ``backbone.shared.zones.Zone``. This test pins
that those imports succeed *without* ``cv2`` or ``calibration`` importable, so a
heavyweight dependency can never silently creep back into the gateway's path.

Run in a subprocess with a meta_path finder that blocks ``cv2``/``calibration``
so the assertion holds even if the parent test process already imported them.
"""

from __future__ import annotations

import subprocess
import sys

_BLOCKED_IMPORT_SCRIPT = r"""
import importlib.abc
import importlib.machinery
import sys

BLOCKED = {"cv2", "calibration"}


class _Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        top = fullname.split(".", 1)[0]
        if top in BLOCKED:
            raise ModuleNotFoundError(f"blocked for lean-image test: {fullname}")
        return None


sys.meta_path.insert(0, _Blocker())

# The two imports the lean gateway relies on.
from backbone.shared.zones import Zone
from backbone.comms.schemas import Track2DMessage  # noqa: F401

# Exercise the numpy point-in-polygon path (the cv2 user we removed).
z = Zone("z", "danger", __import__("numpy").array(
    [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]]
))
assert z.contains((1.0, 1.0)) is True
assert z.contains((5.0, 5.0)) is False
assert z.contains((2.0, 1.0)) is True   # on edge

# Prove the block is real: importing cv2/calibration must fail.
for name in ("cv2", "calibration"):
    try:
        __import__(name)
    except ModuleNotFoundError:
        pass
    else:
        raise AssertionError(f"{name} was importable — block did not engage")

print("OK")
"""


def test_zones_import_without_cv2_or_calibration() -> None:
    """``backbone.shared.zones`` + ``backbone.comms.schemas`` import lean."""
    result = subprocess.run(
        [sys.executable, "-c", _BLOCKED_IMPORT_SCRIPT],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"lean import failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert result.stdout.strip().endswith("OK"), result.stdout
