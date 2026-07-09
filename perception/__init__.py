"""isi perception — the Pixels side of Direction 1.

Owns RTSP capture (via an injected frame provider or its own sources),
zone-scoped object detection, and pose, and publishes per-camera
``DetectionSetMessage``s to the Backbone's points-mode ingest port. The
Backbone stays a pure metric engine; this package is the only place (besides
the dashboard's display overlays) that touches pixels and CUDA.

Deliberately FastAPI-free: ``monitor_web`` hosts :class:`~perception.core.PerceptionCore`
in-process on the dev box (sharing the camera hub — one decode per camera),
while headless deployments run ``python -m perception`` as its own service.
"""

from perception.core import PerceptionCore, build_perception_core

__all__ = ["PerceptionCore", "build_perception_core"]
