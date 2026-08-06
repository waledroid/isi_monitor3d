"""Homography layer: detections → metric tracks (always-on default output).

Importing this package auto-registers the ``bytetrack`` ``Tracker`` plugin.
After ``import backbone.homography``, ``tracker_registry.names()`` includes
``"bytetrack"``.

Pipeline (wired by the orchestrator in S6):

    Detection(per cam) ──► FootProjector ──► (det, floor_xy) pairs
                                                  │
                                                  ▼
                                            CrossCamFusion
                                                  │
                                                  ▼
                                          DisagreementGate
                                                  │
                                                  ▼
                                          ByteTrackMeters  →  Track2D list
                                                  │
                                                  ▼
                                       TemporalStabilizer  →  Track2D list (published)
"""

from . import bytetrack as _bytetrack  # noqa: F401  — registers "bytetrack"
from .bytetrack import ByteTrackMeters
from .cross_cam_fusion import CrossCamFusion, FusedObservation
from .disagreement_gate import DisagreementGate
from .foot_projector import FootProjector
from .pallet_occupancy import OccupancyStabilizer, PalletOccupancy
from .pallet_state_manager import PalletStateManager, ZoneDecision
from .temporal_stabilizer import TemporalStabilizer
from .track import InternalTrack, TrackConfig, TrackState

__all__ = [
    "ByteTrackMeters",
    "CrossCamFusion",
    "DisagreementGate",
    "FootProjector",
    "FusedObservation",
    "InternalTrack",
    "OccupancyStabilizer",
    "PalletOccupancy",
    "PalletStateManager",
    "TemporalStabilizer",
    "TrackConfig",
    "TrackState",
    "ZoneDecision",
]
