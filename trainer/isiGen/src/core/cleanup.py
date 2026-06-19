"""Memory hygiene around GPU-heavy phase jobs.

isiGen runs every phase *in the Studio process* (no per-phase subprocesses), so
the equivalent of the dashboard's orphan-reaping is: release Python + CUDA
memory between jobs, and reap any **orphaned** prior isiGen process that is still
holding the GPU (its launcher died → reparented to init). The running Studio is
never touched.

Used by the JobRunner before and after each job.
"""

from __future__ import annotations

import gc
import logging
import os
import signal

logger = logging.getLogger(__name__)


def vram_used_total_mb() -> tuple[int, int] | None:
    """(used, total) MB across the whole GPU, or None if no CUDA."""
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return (total - free) // (1024 * 1024), total // (1024 * 1024)
    except Exception:
        pass
    return None


def _trim_host_heap() -> None:
    """Return freed HOST RAM to the OS (glibc only). ``gc.collect`` frees the Python
    objects (e.g. the SDXL weights, which live in CPU RAM under ``cpu_offload``), but
    glibc keeps the pages in its malloc arena, so the process RSS doesn't drop. A
    ``malloc_trim(0)`` hands the freed arenas back to the kernel — the difference
    between an idle Studio sitting at ~9 GB vs ~2-3 GB after a heavy mint phase.
    Best-effort: no-op on non-glibc libc."""
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def free_memory(label: str = "") -> tuple[int, int] | None:
    """Run the garbage collector, release cached CUDA VRAM back to the driver, AND
    return freed host RAM to the OS. Logs VRAM after. Safe with no GPU."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    _trim_host_heap()                          # hand freed host heap back to the kernel
    v = vram_used_total_mb()
    if v is not None:
        logger.info("memory[%s]: VRAM %d/%d MB used after cleanup", label or "-", v[0], v[1])
    return v


def _gpu_pids() -> dict[int, int]:
    """{pid: used_MB} for processes currently on the GPU (read-only, via pynvml)."""
    out: dict[int, int] = {}
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            for i in range(pynvml.nvmlDeviceGetCount()):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                for p in pynvml.nvmlDeviceGetComputeRunningProcesses(h):
                    mb = (p.usedGpuMemory or 0) // (1024 * 1024) if p.usedGpuMemory else 0
                    out[int(p.pid)] = out.get(int(p.pid), 0) + mb
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        pass
    return out


def _cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            return fh.read().replace(b"\x00", b" ").decode("utf-8", "ignore").lower()
    except Exception:
        return ""


def _ppid(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/stat") as fh:
            return int(fh.read().split(") ", 1)[1].split()[1])
    except Exception:
        return -1


def reap_orphans() -> list[int]:
    """SIGKILL orphaned (reparented to init) isiGen processes that are still
    holding the GPU. Never targets the current process. Returns reaped PIDs.

    Disabled when ``ISIGEN_DISABLE_REAP=1`` (set in tests so the suite never
    kills a real Studio)."""
    if os.environ.get("ISIGEN_DISABLE_REAP") == "1":
        return []
    self_pid = os.getpid()
    reaped: list[int] = []
    for pid, mb in _gpu_pids().items():
        if pid == self_pid:
            continue
        cmd = _cmdline(pid)
        is_isigen = ("isigen" in cmd or "run_studio.py" in cmd
                     or "/scripts/run_" in cmd)
        if is_isigen and _ppid(pid) == 1:              # orphan: launcher is gone
            try:
                os.kill(pid, signal.SIGKILL)
                reaped.append(pid)
                logger.warning("reaped orphan GPU process pid=%d (%d MB): %s",
                               pid, mb, cmd[:120])
            except ProcessLookupError:
                pass
            except Exception as exc:
                logger.warning("could not reap pid=%d (%s)", pid, exc)
    # warn about any OTHER live GPU users (not reaped) so the operator knows
    others = {p: m for p, m in _gpu_pids().items() if p != self_pid and p not in reaped}
    if others:
        logger.info("note: other live GPU processes present: %s",
                    ", ".join(f"pid {p} ({m} MB)" for p, m in others.items()))
    return reaped


def prepare_for_gpu(label: str = "") -> None:
    """Pre-job: reap orphaned isiGen GPU stragglers, then free memory."""
    reap_orphans()
    free_memory(f"before:{label}")
