"""Memory hygiene around phase jobs.

isical is CPU-only (capture + cv2.aruco + Multical subprocess), so unlike isiGen
there's no GPU/VRAM to reap — just run the GC and hand any freed host heap back to
the OS (glibc ``malloc_trim``) after each job. The JobRunner calls these before/
after every job, with the same hook names isiGen's JobRunner expects.
"""

from __future__ import annotations

import gc
import logging

logger = logging.getLogger(__name__)


def _trim_host_heap() -> None:
    """Return freed host RAM to the OS (glibc only; best-effort no-op elsewhere)."""
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def free_memory(label: str = "") -> None:
    """Post-job: GC + return freed heap to the kernel."""
    gc.collect()
    _trim_host_heap()


def prepare_for_gpu(label: str = "") -> None:
    """Pre-job hook (named for JobRunner compatibility; no GPU here) — just GC."""
    gc.collect()
