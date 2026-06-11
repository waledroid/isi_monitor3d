import gc
import logging

from src.shared.registry import HOOKS

logger = logging.getLogger(__name__)


def _rss_gb() -> float:
    """Resident set size of this process in GB (no psutil dependency)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1e6  # kB → GB
    except Exception:
        pass
    return 0.0


@HOOKS.register('MemoryCleanup')
class MemoryCleanup:
    """Free accumulated host + GPU memory at each epoch boundary.

    Long runs on a memory-constrained box (the 12 GB WSL VM here) creep up in
    host RSS and fragmented CUDA cache across epochs. Left unchecked that creep
    can end in a silent kernel OOM-kill mid-epoch (SIGKILL, no Python traceback —
    exactly the symptom that took down this run twice around epoch ~90).

    Forcing ``gc.collect()`` plus ``torch.cuda.empty_cache()`` after every epoch
    releases dead Python objects and returns cached-but-unused GPU blocks to the
    allocator, flattening the creep. It logs RSS + reserved-GPU before/after so
    the effect (and any remaining leak) is observable in the training log.
    """

    def after_epoch(self, trainer):
        import torch
        cuda = torch.cuda.is_available()
        rss_before = _rss_gb()
        gpu_before = torch.cuda.memory_reserved(0) / 1e9 if cuda else 0.0

        collected = gc.collect()
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        rss_after = _rss_gb()
        gpu_after = torch.cuda.memory_reserved(0) / 1e9 if cuda else 0.0
        logger.info(
            "🧹 MemoryCleanup: gc freed %d objs | RSS %.2fG→%.2fG | CUDA reserved %.2fG→%.2fG",
            collected, rss_before, rss_after, gpu_before, gpu_after,
        )
