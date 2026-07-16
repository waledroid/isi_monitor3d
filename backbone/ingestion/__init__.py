"""Ingestion layer: RTSP / replay / V4L2 sources, frame pairing, in-process bus.

Importing this package auto-registers the ``rtsp``, ``replay``, and ``v4l2``
``FrameSource`` plugins. After ``import backbone.ingestion``,
``frame_source_registry.names()`` returns ``["replay", "rtsp", "v4l2"]``.
"""

from . import replay as _replay  # noqa: F401  — registers "replay"
from . import rtsp as _rtsp  # noqa: F401  — registers "rtsp"
from . import v4l2 as _v4l2  # noqa: F401  — registers "v4l2"
from .frame_bus import FrameBus
from .frame_sync import FrameSynchronizer
from .replay import ReplayFrameSource
from .rtsp import RtspFrameSource
from .v4l2 import V4l2FrameSource

__all__ = [
    "FrameBus",
    "FrameSynchronizer",
    "ReplayFrameSource",
    "RtspFrameSource",
    "V4l2FrameSource",
]
