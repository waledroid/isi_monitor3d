"""``backbone.shared.hardware.gpu_available`` — the high-level CUDA-GPU probe."""

from __future__ import annotations

import backbone.shared.hardware as hw


def test_returns_bool() -> None:
    hw.gpu_available.cache_clear()
    assert isinstance(hw.gpu_available(), bool)
    hw.gpu_available.cache_clear()


def test_gpu_true_when_nvidia_smi_lists_a_gpu(monkeypatch) -> None:
    hw.gpu_available.cache_clear()
    monkeypatch.setattr(hw.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")

    class _Result:
        returncode = 0
        stdout = "GPU 0: NVIDIA GeForce RTX 5070 (UUID: GPU-abc)"

    monkeypatch.setattr(hw.subprocess, "run", lambda *a, **k: _Result())
    assert hw.gpu_available() is True
    hw.gpu_available.cache_clear()


def test_cpu_when_no_nvidia_and_no_ort_cuda(monkeypatch) -> None:
    hw.gpu_available.cache_clear()
    monkeypatch.setattr(hw.shutil, "which", lambda _name: None)  # nvidia-smi absent
    import onnxruntime as ort

    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    assert hw.gpu_available() is False
    hw.gpu_available.cache_clear()
