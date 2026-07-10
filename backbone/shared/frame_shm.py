"""Shared-memory frame bus — decode once, share everywhere.

The perception process (the single camera owner in Direction 1) publishes
every decoded frame into ``/dev/shm/isi3d_frame_<camera_id>``; any local
consumer (the dashboard's camera hub, tools) reads the latest frame with a
memory copy instead of opening its own RTSP session. This is the CPU
equivalent of DeepStream's one-pipeline fan-out: ONE ingest, ONE decode,
frames + their ``capture_ts`` shared to every consumer — the panels display
the exact pixels the models saw.

Layout (little-endian): a fixed header
``magic(8s) version(u32) width(u32) height(u32) channels(u32) latest_slot(u32)``
followed by TWO slots, each ``seq(u64) capture_ts(f64)`` + the raw BGR frame
bytes. Double-buffered seqlock: the writer bumps the slot's ``seq`` to odd,
memcpys the frame, stamps ``capture_ts``, bumps ``seq`` to even, then flips
``latest_slot`` — a reader that sees an odd or changed ``seq`` retries on the
other side of the flip. No locks, no syscalls per frame, ~1 ms per 720p write.

Staleness is the liveness signal: a reader treats ``capture_ts`` older than
its ``max_age_s`` as "writer gone" and returns ``None`` — consumers then fall
back to their own source (the hub reopens RTSP).
"""

from __future__ import annotations

import logging
import mmap
import os
import struct

import numpy as np

logger = logging.getLogger(__name__)

_MAGIC = b"ISI3DFRM"
_VERSION = 1
_HEADER = struct.Struct("<8sIIIII")          # magic, version, w, h, c, latest_slot
_SLOT_HDR = struct.Struct("<Qd")             # seq, capture_ts
def _default_dir() -> str:
    """Bus directory — /dev/shm, overridable for tests via ISI3D_SHM_DIR."""
    return os.environ.get("ISI3D_SHM_DIR", "/dev/shm")


def shm_path(camera_id: str, directory: str | None = None) -> str:
    directory = directory or _default_dir()
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in camera_id)
    return os.path.join(directory, f"isi3d_frame_{safe}")


def _layout(w: int, h: int, c: int) -> tuple[int, int, int]:
    """(slot_span, frame_bytes, total_size) for the given dims."""
    frame_bytes = w * h * c
    slot_span = _SLOT_HDR.size + frame_bytes
    return slot_span, frame_bytes, _HEADER.size + 2 * slot_span


class FrameShmWriter:
    """Publish the latest frame for one camera. Single writer per camera."""

    def __init__(self, camera_id: str, directory: str | None = None) -> None:
        self._path = shm_path(camera_id, directory)
        self._camera_id = camera_id
        self._mm: mmap.mmap | None = None
        self._dims: tuple[int, int, int] | None = None
        self._latest = 0
        self._seq = 0

    def write(self, image: np.ndarray, capture_ts: float) -> None:
        h, w = image.shape[:2]
        c = 1 if image.ndim == 2 else image.shape[2]
        if self._mm is None or self._dims != (w, h, c):
            self._create(w, h, c)
        assert self._mm is not None
        slot = 1 - self._latest
        slot_span, frame_bytes, _ = _layout(w, h, c)
        off = _HEADER.size + slot * slot_span
        self._seq += 1                                       # odd = mid-write
        _SLOT_HDR.pack_into(self._mm, off, self._seq, 0.0)
        start = off + _SLOT_HDR.size
        self._mm[start:start + frame_bytes] = \
            np.ascontiguousarray(image, dtype=np.uint8).tobytes()
        self._seq += 1                                       # even = stable
        _SLOT_HDR.pack_into(self._mm, off, self._seq, float(capture_ts))
        self._latest = slot
        _HEADER.pack_into(self._mm, 0, _MAGIC, _VERSION, w, h, c, slot)

    def close(self) -> None:
        if self._mm is not None:
            try:
                self._mm.close()
            except (BufferError, ValueError):
                pass
            self._mm = None

    def unlink(self) -> None:
        """Remove the bus file (on deliberate shutdown, so readers see
        'absent' instantly instead of waiting out the staleness window)."""
        self.close()
        try:
            os.unlink(self._path)
        except OSError:
            pass

    # ---- internals ----

    def _create(self, w: int, h: int, c: int) -> None:
        self.close()
        _, _, total = _layout(w, h, c)
        fd = os.open(self._path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            os.ftruncate(fd, total)
            self._mm = mmap.mmap(fd, total)
        finally:
            os.close(fd)
        self._dims = (w, h, c)
        self._latest = 0
        self._seq = 0
        _HEADER.pack_into(self._mm, 0, _MAGIC, _VERSION, w, h, c, 0)
        _SLOT_HDR.pack_into(self._mm, _HEADER.size, 0, 0.0)
        slot_span, _, _ = _layout(w, h, c)
        _SLOT_HDR.pack_into(self._mm, _HEADER.size + slot_span, 0, 0.0)
        logger.info("frame bus: %s → %s (%dx%dx%d)", self._camera_id, self._path, w, h, c)


class FrameShmReader:
    """Read the latest frame for one camera. Any number of readers."""

    def __init__(self, camera_id: str, directory: str | None = None,
                 max_age_s: float = 2.0) -> None:
        self._path = shm_path(camera_id, directory)
        self._max_age_s = float(max_age_s)
        self._mm: mmap.mmap | None = None
        self._size = 0
        self._last_ts = 0.0

    def latest(self, *, now: float | None = None) -> tuple[np.ndarray, float] | None:
        """Newest ``(frame_copy, capture_ts)``, or ``None`` when the bus is
        absent, stale (writer dead), or mid-resize."""
        import time
        now = time.time() if now is None else now
        mm = self._map()
        if mm is None:
            return None
        try:
            magic, version, w, h, c, latest = _HEADER.unpack_from(mm, 0)
        except struct.error:
            return None
        if magic != _MAGIC or version != _VERSION or latest not in (0, 1):
            return None
        slot_span, frame_bytes, total = _layout(w, h, c)
        if self._size != total:
            self._remap()                    # resolution changed → remap once
            return None
        off = _HEADER.size + latest * slot_span
        for _ in range(3):                   # seqlock retry on torn reads
            seq0, ts = _SLOT_HDR.unpack_from(mm, off)
            if seq0 == 0 or seq0 % 2 == 1:
                return None                  # never written / mid-write
            if now - ts > self._max_age_s:
                return None                  # writer dead → fall back
            # Copy straight into a WRITABLE array. `np.frombuffer` over the
            # mmap (or over immutable `bytes`) yields a READ-ONLY view, and
            # every display consumer draws overlays IN PLACE — cv2 then dies
            # with "dst marked as output argument, but provided NumPy array
            # marked as readonly". One copy either way; this one is writable.
            frame = np.empty(frame_bytes, dtype=np.uint8)
            start = off + _SLOT_HDR.size
            frame[:] = np.frombuffer(mm, dtype=np.uint8,
                                     count=frame_bytes, offset=start)
            seq1, _ = _SLOT_HDR.unpack_from(mm, off)
            if seq1 == seq0:
                shape = (h, w) if c == 1 else (h, w, c)
                self._last_ts = ts
                return frame.reshape(shape), ts
        return None

    def fresh(self, *, now: float | None = None) -> bool:
        return self.latest(now=now) is not None

    def close(self) -> None:
        if self._mm is not None:
            try:
                self._mm.close()
            except (BufferError, ValueError):
                pass
            self._mm = None
            self._size = 0

    # ---- internals ----

    def _map(self) -> mmap.mmap | None:
        try:
            size = os.path.getsize(self._path)
        except OSError:
            self.close()
            return None
        if self._mm is not None and self._size == size:
            return self._mm
        self.close()
        try:
            fd = os.open(self._path, os.O_RDONLY)
            try:
                self._mm = mmap.mmap(fd, size, prot=mmap.PROT_READ)
            finally:
                os.close(fd)
            self._size = size
        except (OSError, ValueError):
            self._mm = None
            self._size = 0
        return self._mm

    def _remap(self) -> None:
        self.close()
