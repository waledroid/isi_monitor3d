"""isistream — the Pixels side of Direction 1.

Owns RTSP capture (via an injected frame provider or its own sources),
zone-scoped object detection, and pose, and publishes per-camera
``DetectionSetMessage``s to the Backbone's points-mode ingest port. The
Backbone stays a pure metric engine; this package is the only place (besides
the dashboard's display overlays) that touches pixels and CUDA.

Deliberately FastAPI-free: ``monitor_web`` supervises :class:`~isistream.core.IsistreamCore`
in-process on the dev box (sharing the camera hub — one decode per camera),
while headless deployments run ``python -m isistream`` as its own service.
"""

from isistream.core import IsistreamCore, build_isistream_core

__all__ = ["IsistreamCore", "build_isistream_core"]
