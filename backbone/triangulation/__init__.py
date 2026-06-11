"""Triangulation layer: subscription-driven 3D, identity inherited from S4.

Importing this package auto-registers the ``opencv_dlt`` ``Triangulator``
plugin. After ``import backbone.triangulation``, ``triangulator_registry``
includes ``"opencv_dlt"``.

Pipeline (wired by the orchestrator in S6):

    Track2D list ──► SubscriptionManager ──► matched tracks
                                                │
                                                ▼
                                       KeypointAssociator   (per-cam foot pixels)
                                                │
                                                ▼
                                       OpencvDltTriangulator
                                                │
                                                ▼
                                       ReprojectionGate     (≤ 5 px max)
                                                │
                                                ▼
                                            Tracker3D       (3D Kalman, keyed by track_id)
                                                │
                                                ▼
                                            Track3D list (published)
"""

from . import opencv_dlt as _opencv_dlt  # noqa: F401  — registers "opencv_dlt"
from .keypoint_associator import KeypointAssociator
from .opencv_dlt import OpencvDltTriangulator
from .reprojection_gate import ReprojectionGate
from .subscription_manager import MatchRule, SubscriptionManager, SubscriptionRule
from .tracker_3d import Track3DConfig, Tracker3D

__all__ = [
    "KeypointAssociator",
    "MatchRule",
    "OpencvDltTriangulator",
    "ReprojectionGate",
    "SubscriptionManager",
    "SubscriptionRule",
    "Track3DConfig",
    "Tracker3D",
]
