"""Native TensorRT `.engine` support (backbone/shared/trt_session.py + the
suffix dispatch in build_onnx_session).

Hermetic — no GPU, no TensorRT: the engine-session construction is
monkeypatched; what's pinned is the CONTRACT: suffix dispatch, sidecar
validation semantics, honest fallback to the sidecar's source onnx, and the
class-names metadata surface.
"""

from __future__ import annotations

import json

import pytest

from backbone.shared import ort_session
from backbone.shared.trt_session import (
    read_sidecar,
    sidecar_path,
    validate_sidecar,
)

TRT_V = "10.16.1.11"


# ------------------------------------------------------------ sidecar rules

def test_validate_no_sidecar_is_warning_not_fatal():
    problems = validate_sidecar(None, TRT_V, "NVIDIA GeForce RTX 5070")
    assert len(problems) == 1
    assert "rebuild" not in problems[0]          # soft warning only


def test_validate_trt_major_minor_mismatch_is_fatal():
    problems = validate_sidecar(
        {"tensorrt_version": "10.4.0", "gpu_name": "NVIDIA GeForce RTX 5070"},
        TRT_V, "NVIDIA GeForce RTX 5070")
    assert any("rebuild" in p for p in problems)


def test_validate_patch_version_difference_is_fine():
    problems = validate_sidecar(
        {"tensorrt_version": "10.16.0.5", "gpu_name": "G"}, TRT_V, "G")
    assert problems == []


def test_validate_gpu_mismatch_is_fatal():
    problems = validate_sidecar(
        {"tensorrt_version": TRT_V, "gpu_name": "Orin"}, TRT_V, "RTX 5070")
    assert any("rebuild" in p for p in problems)


def test_read_sidecar_roundtrip(tmp_path):
    eng = tmp_path / "m.engine"
    eng.write_bytes(b"x")
    sidecar_path(eng).write_text(json.dumps({"gpu_name": "G"}))
    assert read_sidecar(eng) == {"gpu_name": "G"}
    assert read_sidecar(tmp_path / "missing.engine") is None


# ------------------------------------------------------- suffix dispatch

def test_engine_suffix_dispatches_to_trt_session(monkeypatch, tmp_path):
    sentinel = object()
    import backbone.shared.trt_session as ts
    monkeypatch.setattr(ts, "TrtEngineSession", lambda p: sentinel)
    eng = tmp_path / "m.engine"
    eng.write_bytes(b"x")
    assert ort_session.build_onnx_session(str(eng)) is sentinel


def test_engine_load_failure_falls_back_to_source_onnx(monkeypatch, tmp_path):
    """A foreign engine (wrong GPU/TRT) must not kill startup when its
    sidecar-recorded source onnx exists — the honest fallback loads that."""
    import backbone.shared.trt_session as ts

    def boom(_p):
        raise RuntimeError("engine built for 'Orin', this machine differs")

    monkeypatch.setattr(ts, "TrtEngineSession", boom)
    src = tmp_path / "model.onnx"
    src.write_bytes(b"onnx-bytes")
    eng = tmp_path / "model.engine"
    eng.write_bytes(b"x")
    sidecar_path(eng).write_text(json.dumps({"source_onnx": str(src)}))

    captured: dict = {}

    def fake_ort_session(path, sess_options=None, providers=None):
        captured["path"] = str(path)
        return "ORT_SESSION"

    monkeypatch.setattr(ort_session.ort, "InferenceSession", fake_ort_session)
    out = ort_session.build_onnx_session(str(eng))
    assert out == "ORT_SESSION"
    assert captured["path"] == str(src)


def test_engine_load_failure_without_fallback_raises(monkeypatch, tmp_path):
    import backbone.shared.trt_session as ts
    monkeypatch.setattr(
        ts, "TrtEngineSession",
        lambda p: (_ for _ in ()).throw(RuntimeError("no good")))
    eng = tmp_path / "m.engine"
    eng.write_bytes(b"x")                       # no sidecar → no fallback
    with pytest.raises(RuntimeError):
        ort_session.build_onnx_session(str(eng))


def test_onnx_suffix_still_builds_ort_session(monkeypatch, tmp_path):
    monkeypatch.setattr(ort_session.ort, "InferenceSession",
                        lambda path, sess_options=None, providers=None: "ORT")
    assert ort_session.build_onnx_session(str(tmp_path / "m.onnx")) == "ORT"


def test_onnx_never_gets_trt_ep_even_with_legacy_env(monkeypatch):
    """The retired lazy TRT-EP path must stay dead: a stale ISI3D_TRT=1 in the
    environment injects NOTHING — `.onnx` means plain ONNX Runtime, TensorRT
    happens only through native `.engine` files."""
    monkeypatch.setenv("ISI3D_TRT", "1")
    resolved = ort_session.resolve_providers(
        ["CUDAExecutionProvider", "CPUExecutionProvider"])
    names = [p[0] if isinstance(p, tuple) else p for p in resolved]
    assert names == ["CUDAExecutionProvider", "CPUExecutionProvider"]


# ------------------------------------------------- class-names metadata

def test_modelmeta_names_from_sidecar(tmp_path):
    """TrtEngineSession.get_modelmeta must surface sidecar class_names in the
    exact shape read_embedded_class_names parses."""
    from backbone.detection.onnx_meta import read_embedded_class_names
    from backbone.shared.trt_session import TrtEngineSession

    eng = tmp_path / "m.engine"
    eng.write_bytes(b"x")
    sidecar_path(eng).write_text(
        json.dumps({"class_names": ["palette", "carton", "polybag"]}))
    fake = object.__new__(TrtEngineSession)      # skip GPU __init__
    fake._path = eng
    assert read_embedded_class_names(fake) == ["palette", "carton", "polybag"]
