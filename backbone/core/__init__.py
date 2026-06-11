"""Core contracts: types, registry, and the five plugin seams.

This subpackage is the only one any other Backbone module is allowed to import
unconditionally. Everything else is wired by the runtime orchestrator from YAML.
"""

from .interfaces import (
    Detector,
    FrameSource,
    MetadataSink,
    Tracker,
    Triangulator,
    detector_registry,
    frame_source_registry,
    metadata_sink_registry,
    tracker_registry,
    triangulator_registry,
)
from .registry import Registry, RegistryError
from .types import Detection, Frame, FramePair, Track2D, Track3D

__all__ = [
    "Detection",
    "Detector",
    "Frame",
    "FramePair",
    "FrameSource",
    "MetadataSink",
    "Registry",
    "RegistryError",
    "Track2D",
    "Track3D",
    "Tracker",
    "Triangulator",
    "detector_registry",
    "frame_source_registry",
    "metadata_sink_registry",
    "tracker_registry",
    "triangulator_registry",
]
