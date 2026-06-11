"""Concrete, single-implementation utilities used across the Backbone.

No ABCs live here — see ``backbone.core.interfaces`` for the five seams.
"""

from .camera_rig import CameraRig
from .timestamps import LatencyMeter, elapsed_ms, now

__all__ = ["CameraRig", "LatencyMeter", "elapsed_ms", "now"]
