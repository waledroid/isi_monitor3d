"""Unit tests for the shared ORT session builder (backbone/shared/ort_session.py).

These are pure (no model load): they prove the CUDA-only memory option is attached
to CUDAExecutionProvider and that CPU/other providers pass through untouched — so
the hermetic CPU test suite and CPU-only deployments are unaffected.
"""
import onnxruntime as ort

from backbone.shared.ort_session import (
    DEFAULT_PROVIDERS,
    make_session_options,
    resolve_providers,
)


def test_cuda_provider_gets_arena_option():
    resolved = resolve_providers(["CUDAExecutionProvider", "CPUExecutionProvider"])
    # CUDA → (name, options) tuple with the tight arena strategy and the
    # realtime-friendly conv algo search (EXHAUSTIVE = 10-30 s first-call stall).
    assert resolved[0] == (
        "CUDAExecutionProvider",
        {"arena_extend_strategy": "kSameAsRequested",
         "cudnn_conv_algo_search": "HEURISTIC"},
    )
    # CPU passes through as a bare string.
    assert resolved[1] == "CPUExecutionProvider"


def test_cpu_only_is_untouched():
    # No CUDA entry → no tuples, no options injected.
    resolved = resolve_providers(["CPUExecutionProvider"])
    assert resolved == ["CPUExecutionProvider"]


def test_default_providers_shape():
    assert DEFAULT_PROVIDERS == ("CUDAExecutionProvider", "CPUExecutionProvider")


def test_session_options_enables_full_graph_optimization():
    opts = make_session_options()
    assert opts.graph_optimization_level == ort.GraphOptimizationLevel.ORT_ENABLE_ALL


def test_cuda_session_options_cap_intra_op_pool():
    """CUDA sessions get a tiny non-spinning intra-op pool — ORT's default
    (one spinning thread per core, per session) starves the RTSP decode
    threads on the edge PC and collapses frame pairing."""
    opts = make_session_options(for_cuda=True)
    assert opts.intra_op_num_threads == 2
    assert opts.graph_optimization_level == ort.GraphOptimizationLevel.ORT_ENABLE_ALL


def test_cpu_session_options_keep_default_pool():
    """CPU-only sessions keep ORT's defaults — there the pool does the real work."""
    opts = make_session_options()
    assert opts.intra_op_num_threads == 0   # 0 = ORT default (per-core)
