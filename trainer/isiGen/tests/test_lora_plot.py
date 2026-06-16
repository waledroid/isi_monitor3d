"""LoRA loss-curve plotting — the EMA smoother that makes the (intrinsically
jagged) diffusion loss readable. Hermetic: pure math + headless matplotlib."""

import statistics

from src.stages.lora.diffusers_sdxl import DiffusersSdxlLoraTrainer as L

JAGGED = [0.30, 0.0, 0.25, 0.05, 0.20, 0.10, 0.15, 0.12, 0.18, 0.08]


def test_ema_is_a_bounded_smoother():
    ema = L._ema(JAGGED, alpha=0.1)
    assert len(ema) == len(JAGGED)
    assert abs(ema[0] - JAGGED[0]) < 1e-9                      # seeded at first sample
    # stays within the data range (it's a weighted average) and is smoother
    assert min(JAGGED) - 1e-9 <= min(ema) and max(ema) <= max(JAGGED) + 1e-9
    assert statistics.pstdev(ema) < statistics.pstdev(JAGGED)


def test_ema_empty_and_singleton():
    assert L._ema([]) == []
    assert L._ema([0.42]) == [0.42]


def test_plot_renders_png(tmp_path):
    out = tmp_path / "loss_curve.png"
    L._plot_losses(JAGGED, out)
    assert out.exists() and out.stat().st_size > 1000
    # never fails training, even with no data
    L._plot_losses([], tmp_path / "empty.png")
