"""One shared ONNX Runtime session builder with memory-safe defaults.

Every InferenceSession the Backbone creates — and the dashboard's pose overlay —
goes through :func:`build_onnx_session` so they all share the same memory options
instead of each call site duplicating the setup.

The option that matters is ``arena_extend_strategy="kSameAsRequested"`` on the CUDA
provider. ORT's default ``kNextPowerOfTwo`` rounds every allocation up to the next
power of two, which over-reserves VRAM and OOMs on the 12 GB card when several
models are resident at once (e.g. an 840x840 RF-DETR plus a pose model).
``kSameAsRequested`` extends the arena by exactly what each allocation needs.

CPU (and any non-CUDA) provider passes through unchanged, so CPU-only deployments
and the hermetic CPU test suite are unaffected by the CUDA-only option.
"""
from __future__ import annotations

import onnxruntime as ort

DEFAULT_PROVIDERS: tuple[str, ...] = ("CUDAExecutionProvider", "CPUExecutionProvider")

# Memory-safe CUDA provider options. kSameAsRequested keeps the arena tight so
# concurrent sessions fit on a 12 GB card.
_CUDA_OPTIONS = {
    "arena_extend_strategy": "kSameAsRequested",
    # HEURISTIC, not the default EXHAUSTIVE: exhaustive benchmarks every conv
    # algorithm on the FIRST inference of a session — measured as a 10-30 s
    # stall per fresh session on the RTX 5070, during which a lazily-built
    # pose/zone session answers nothing (skeletons appear ~30 s after a
    # restart; a passer-by is gone before pose shows). Heuristic picks in
    # milliseconds at near-identical steady-state throughput.
    "cudnn_conv_algo_search": "HEURISTIC",
}


def resolve_providers(providers):
    """Return an ORT providers list with memory-safe options on the CUDA provider.

    Each ``CUDAExecutionProvider`` entry becomes a ``(name, options)`` tuple; every
    other provider passes through unchanged.
    """
    resolved = []
    for p in providers:
        if p == "CUDAExecutionProvider":
            resolved.append((p, dict(_CUDA_OPTIONS)))
        else:
            resolved.append(p)
    return resolved


def make_session_options(*, for_cuda: bool = False) -> ort.SessionOptions:
    """SessionOptions shared by every session: full graph optimization, quiet logs
    (provider fallback is already logged by each caller).

    ``for_cuda=True`` additionally caps the intra-op CPU pool at 2 threads and
    disables its busy-spinning. A CUDA session uses CPU threads only for
    pre/post + kernel launch, but ORT's defaults give every session one
    **spinning** thread per core — two resident models on the 8-core edge PC
    meant 16 busy-wait threads starving the GStreamer RTSP decode callbacks:
    frame pairing collapsed from ~16 pairs/s to ~0.5/s while isolated inference
    looked perfectly healthy. The cap costs nothing measurable on GPU inference.
    CPU-only sessions keep ORT's defaults (there, the pool does the real work).
    """
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.log_severity_level = 3
    if for_cuda:
        opts.intra_op_num_threads = 2
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return opts


def build_onnx_session(onnx_path, providers=None):
    """Create an ORT ``InferenceSession`` with memory-safe defaults.

    Attaches ``arena_extend_strategy=kSameAsRequested`` to the CUDA provider so
    sessions don't over-reserve VRAM, and (for CUDA sessions) a small
    non-spinning intra-op pool so resident models don't starve the RTSP decode
    threads; CPU/other providers pass through unchanged.
    """
    providers = list(providers) if providers else list(DEFAULT_PROVIDERS)
    for_cuda = "CUDAExecutionProvider" in providers
    return ort.InferenceSession(
        str(onnx_path),
        sess_options=make_session_options(for_cuda=for_cuda),
        providers=resolve_providers(providers),
    )
