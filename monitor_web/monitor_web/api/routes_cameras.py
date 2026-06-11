"""Camera discovery — enumerate locally-attached V4L2 (USB/UVC) devices.

``GET /api/cameras/available`` lets the zone manager offer attached cameras as
an alternative to typing an RTSP URL. Devices are discovered by reading
``/sys/class/video4linux/video*/name`` — dependency-free (no ``v4l2-ctl``
binary required) and present on any Linux box with the V4L2 subsystem.

On WSL2, USB cameras are not exposed to Linux unless attached via
``usbipd-win``, so this returns an empty list there — by design, never an error.
"""

from __future__ import annotations

import logging
from pathlib import Path

from backbone.core.interfaces import frame_source_registry
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter()

_SYSFS_V4L2_ROOT = Path("/sys/class/video4linux")


def discover_v4l2_devices(sysfs_root: Path = _SYSFS_V4L2_ROOT) -> list[dict[str, str]]:
    """Return ``[{path: '/dev/video0', name: 'HD Webcam'}, …]`` for attached cameras.

    Reads each ``videoN/name`` from sysfs. Sorted by numeric device index so
    the capture node (usually the lowest index per physical camera) comes first.
    Never raises — a missing sysfs tree (WSL2, container) yields ``[]``.
    """
    if not sysfs_root.exists():
        return []
    devices: list[tuple[int, dict[str, str]]] = []
    try:
        entries = list(sysfs_root.iterdir())
    except OSError as exc:
        logger.warning("camera discovery: cannot read %s (%s)", sysfs_root, exc)
        return []
    for entry in entries:
        if not entry.name.startswith("video"):
            continue
        try:
            index = int(entry.name.removeprefix("video"))
        except ValueError:
            continue
        name = ""
        name_file = entry / "name"
        try:
            name = name_file.read_text().strip()
        except OSError:
            pass
        devices.append((index, {"path": f"/dev/{entry.name}", "name": name or entry.name}))
    devices.sort(key=lambda t: t[0])
    return [d for _, d in devices]


@router.get("/api/cameras/available")
async def cameras_available(request: Request) -> JSONResponse:
    """List attached V4L2 devices + the registered FrameSource plugin names."""
    # Read the module global at call time so tests can monkeypatch the root.
    devices = discover_v4l2_devices(_SYSFS_V4L2_ROOT)
    return JSONResponse(
        {
            "devices": devices,
            "plugins": sorted(frame_source_registry.names()),
        }
    )
