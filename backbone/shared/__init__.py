"""Concrete, single-implementation utilities used across the Backbone.

No ABCs live here — see ``backbone.core.interfaces`` for the five seams.
"""

from .timestamps import LatencyMeter, elapsed_ms, now

__all__ = ["CameraRig", "LatencyMeter", "elapsed_ms", "now"]


def __getattr__(name):
    # Lazy import: ``CameraRig`` pulls in ``calibration`` (and transitively the
    # heavy geometry stack), which the lean isi-gateway image deliberately omits.
    # PEP 562 keeps ``from backbone.shared import CameraRig`` working while letting
    # the light ``backbone.shared.zones`` submodule import without that cost.
    if name == "CameraRig":
        from .camera_rig import CameraRig

        return CameraRig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
