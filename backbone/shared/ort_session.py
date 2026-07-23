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

import logging
import os
from pathlib import Path

import onnxruntime as ort

logger = logging.getLogger(__name__)

DEFAULT_PROVIDERS: tuple[str, ...] = ("CUDAExecutionProvider", "CPUExecutionProvider")

# TensorRT acceleration (opt-in, Settings ▸ Detection ▸ "TensorRT"). Enabled
# via the ISI3D_TRT=1 env var, which the process entry points set from
# `detection.trt_enabled` — an env var (not a kwarg) so every call site
# (plugins, pose overlay) inherits the decision without signature churn.
# Requires an onnxruntime build that ships TensorrtExecutionProvider (the
# conda-forge build does NOT; the pip onnxruntime-gpu wheel does) + TensorRT
# 10.x libs for Blackwell. When unavailable, requests degrade to CUDA with a
# one-time warning — the system NEVER fails to start over a missing TRT.
_TRT_CACHE_DIR = os.environ.get(
    "ISI3D_TRT_CACHE", str(Path(__file__).resolve().parents[2] / "models" / ".trt_cache"))
_trt_warned = False


def trt_available() -> bool:
    return "TensorrtExecutionProvider" in ort.get_available_providers()


def trt_requested() -> bool:
    return os.environ.get("ISI3D_TRT", "").strip() in ("1", "true", "yes")


def _trt_options() -> dict:
    Path(_TRT_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    return {
        # fp16 engines (matches the deployed model precision); int8 QDQ models
        # bring their own scales and are honored automatically by TRT.
        "trt_fp16_enable": True,
        # The engine + timing caches are NON-NEGOTIABLE here: a cold engine
        # build takes minutes per model/shape, and isistream hot-restarts on
        # every Settings save. Cached engines load in seconds.
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": _TRT_CACHE_DIR,
        "trt_timing_cache_enable": True,
        "trt_timing_cache_path": _TRT_CACHE_DIR,
    }

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
    other provider passes through unchanged. When TensorRT is requested
    (``ISI3D_TRT=1``) and this ORT build ships the EP, it is prepended ahead of
    the first CUDA entry (CUDA stays as the in-graph fallback for nodes TRT
    can't take); requested-but-unavailable degrades to CUDA with one warning.
    """
    global _trt_warned
    resolved = []
    want_trt = trt_requested() and any(p == "CUDAExecutionProvider" for p in providers)
    if want_trt and not trt_available():
        if not _trt_warned:
            _trt_warned = True
            logger.warning(
                "TensorRT requested (ISI3D_TRT=1) but this onnxruntime build has "
                "no TensorrtExecutionProvider — running on CUDA. Install the pip "
                "onnxruntime-gpu wheel + TensorRT 10 to enable it.")
        want_trt = False
    for p in providers:
        if p == "CUDAExecutionProvider":
            if want_trt:
                # Preload the pip TensorRT libs into the process so ORT's EP
                # dlopens them without LD_LIBRARY_PATH plumbing through every
                # supervisor/systemd unit. Harmless if already loaded.
                try:
                    import tensorrt  # noqa: F401
                except ImportError:
                    pass
                resolved.append(("TensorrtExecutionProvider", _trt_options()))
                want_trt = False   # once, ahead of the first CUDA entry
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
    """Create an inference session with memory-safe defaults.

    Dispatches on the file suffix: a ``.engine`` path loads a prebuilt native
    TensorRT engine (``backbone.shared.trt_session`` — seconds to deserialize,
    no lazy TRT-EP build), anything else gets an ORT ``InferenceSession``.
    Both expose the same three surfaces the detector plugins consume
    (``get_inputs/get_outputs``, ``get_providers``, ``run``), so callers never
    know which backend they got.

    A ``.engine`` that refuses to load (wrong GPU / TRT version — engines are
    per-machine artifacts) falls back to its sidecar-recorded source ``.onnx``
    when that file exists, honouring the never-fail-to-start contract; without
    a fallback the honest error propagates.

    For ORT sessions: attaches ``arena_extend_strategy=kSameAsRequested`` to
    the CUDA provider so sessions don't over-reserve VRAM, and (for CUDA
    sessions) a small non-spinning intra-op pool so resident models don't
    starve the RTSP decode threads; CPU/other providers pass through unchanged.
    """
    if str(onnx_path).endswith(".engine"):
        from backbone.shared.trt_session import TrtEngineSession, read_sidecar

        try:
            return TrtEngineSession(onnx_path)
        except Exception as exc:
            meta = read_sidecar(onnx_path) or {}
            source = meta.get("source_onnx")
            if source and Path(source).exists():
                logger.warning(
                    "engine %s unusable (%s) — falling back to its source "
                    "onnx %s", Path(str(onnx_path)).name, exc, source)
                onnx_path = source
            else:
                raise

    providers = list(providers) if providers else list(DEFAULT_PROVIDERS)
    for_cuda = "CUDAExecutionProvider" in providers
    return ort.InferenceSession(
        str(onnx_path),
        sess_options=make_session_options(for_cuda=for_cuda),
        providers=resolve_providers(providers),
    )
