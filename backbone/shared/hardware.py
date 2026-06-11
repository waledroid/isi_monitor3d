"""High-level hardware probe — is a CUDA GPU available?

The Backbone's ONNX detector already auto-selects the accelerator at the ORT
level (``yolo_onnx`` lists ``CUDAExecutionProvider`` then ``CPUExecutionProvider``,
so ONNX Runtime uses the GPU when present and silently falls back to CPU). This
helper makes that hardware fact explicit and reusable for the *choice between
backends* (ONNX-on-GPU vs OpenVINO-on-CPU) — used by the operator dashboard to
pick the detector automatically. It is a consumer-side helper, safe to import
from ``monitor_web`` (like ``backbone.shared.zones``).
"""

from __future__ import annotations

import shutil
import subprocess
from functools import lru_cache


@lru_cache(maxsize=1)
def gpu_available() -> bool:
    """Return True if an NVIDIA CUDA GPU is usable. Cached (hardware is static).

    Primary signal: ``nvidia-smi -L`` lists at least one GPU. Fallback (when
    ``nvidia-smi`` isn't on PATH): ONNX Runtime exposes ``CUDAExecutionProvider``.
    Any error → False (treat as CPU-only).
    """
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            out = subprocess.run(
                [smi, "-L"], capture_output=True, text=True, timeout=5, check=False
            )
            if out.returncode == 0 and "GPU" in out.stdout:
                return True
        except (OSError, subprocess.SubprocessError):
            pass
    try:
        import onnxruntime as ort

        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


def gpu_memory_mb() -> tuple[int, int] | None:
    """Return ``(used_mb, total_mb)`` for GPU 0, or ``None`` if unavailable.

    Not cached — free/used VRAM changes as sessions load. Queries
    ``nvidia-smi``; any error (no GPU, smi missing, parse failure) → ``None``.
    """
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    try:
        out = subprocess.run(
            [smi, "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if out.returncode != 0:
            return None
        used, total = out.stdout.strip().splitlines()[0].split(",")
        return int(used), int(total)
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def gpu_utilization_pct() -> int | None:
    """GPU 0 compute utilization (0-100), or ``None`` if unavailable (nvidia-smi)."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    try:
        out = subprocess.run(
            [smi, "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if out.returncode != 0:
            return None
        return int(out.stdout.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def host_memory_mb() -> tuple[int, int] | None:
    """Return ``(used_mb, total_mb)`` of system (CPU) RAM, or ``None``. Parsed from
    ``/proc/meminfo`` — ``used = MemTotal - MemAvailable`` (the kernel's own estimate
    of reclaimable memory, the same number ``free -h`` shows under *used*)."""
    try:
        info: dict[str, int] = {}
        with open("/proc/meminfo") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                if rest:
                    info[key.strip()] = int(rest.split()[0])   # value is in kB
        total = info.get("MemTotal", 0) // 1024
        avail = info.get("MemAvailable", info.get("MemFree", 0)) // 1024
        if total <= 0:
            return None
        return (max(0, total - avail), total)
    except (OSError, ValueError, KeyError):
        return None
