"""Ingestion layer: RTSP / replay / V4L2 sources, frame pairing, in-process bus.

Importing this package auto-registers the ``rtsp``, ``replay``, ``shm``, and
``v4l2`` ``FrameSource`` plugins. After ``import backbone.ingestion``,
``frame_source_registry.names()`` returns ``["replay", "rtsp", "shm", "v4l2"]``.
"""

from . import replay as _replay  # noqa: F401  — registers "replay"
from . import rtsp as _rtsp  # noqa: F401  — registers "rtsp"
from . import shm_source as _shm  # noqa: F401  — registers "shm"
from . import v4l2 as _v4l2  # noqa: F401  — registers "v4l2"
from .frame_bus import FrameBus
from .frame_sync import FrameSynchronizer
from .replay import ReplayFrameSource
from .rtsp import RtspFrameSource
from .shm_source import ShmFrameSource
from .v4l2 import V4l2FrameSource

__all__ = [
    "FrameBus",
    "FrameSynchronizer",
    "ReplayFrameSource",
    "RtspFrameSource",
    "ShmFrameSource",
    "V4l2FrameSource",
]
