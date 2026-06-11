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
    # CUDA → (name, options) tuple with the tight arena strategy.
    assert resolved[0] == (
        "CUDAExecutionProvider",
        {"arena_extend_strategy": "kSameAsRequested"},
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
