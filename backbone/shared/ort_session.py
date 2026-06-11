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
_CUDA_OPTIONS = {"arena_extend_strategy": "kSameAsRequested"}


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


def make_session_options() -> ort.SessionOptions:
    """SessionOptions shared by every session: full graph optimization, quiet logs
    (provider fallback is already logged by each caller)."""
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.log_severity_level = 3
    return opts


def build_onnx_session(onnx_path, providers=None):
    """Create an ORT ``InferenceSession`` with memory-safe defaults.

    Attaches ``arena_extend_strategy=kSameAsRequested`` to the CUDA provider so
    sessions don't over-reserve VRAM; CPU/other providers pass through unchanged.
    """
    providers = list(providers) if providers else list(DEFAULT_PROVIDERS)
    return ort.InferenceSession(
        str(onnx_path),
        sess_options=make_session_options(),
        providers=resolve_providers(providers),
    )
