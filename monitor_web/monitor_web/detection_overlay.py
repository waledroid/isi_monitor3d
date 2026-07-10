"""Compatibility facade — the implementation now lives in three stage
modules (the modular pipeline layout):

    model_store.py  — model discovery/resolution (which .onnx exists / is configured)
    engines.py      — inference-session lifecycle + GPU guard + reset_detector
    overlay.py      — frame annotation (drawing, occupancy, display prefs)

Import from those modules in new code; this facade re-exports the full old
surface so existing imports keep working unchanged.
"""

from __future__ import annotations

from .engines import *  # noqa: F403

# Private names some consumers import through this module.
from .engines import (  # noqa: F401
    _ASYNC_POSE,
    _gpu_free_mb,
)
from .model_store import *  # noqa: F403
from .model_store import _REPO_ROOT, _RUNS_GLOB, _RUNS_ROOT  # noqa: F401
from .overlay import *  # noqa: F403
from .overlay import _PALLET_CLASSES, _color_for  # noqa: F401
