"""Re-export the trained RF-DETR-seg-medium checkpoint at higher fixed resolutions.

RF-DETR cannot produce a clean dynamic ONNX (windowed attention ties the input to a
fixed ÷56 grid; the package hardcodes dynamic_axes=None). The practical lever for
better far/small-pallet detection is to re-export at a higher *fixed* resolution —
the DINOv2 backbone interpolates its position embeddings to the new size.

Produces, under models/rfdetr/:
    rfdetr_seg_medium_672.sim.onnx   (12 x 56 = 672)
    rfdetr_seg_medium_784.sim.onnx   (14 x 56 = 784)
"""
import shutil
import tempfile
from pathlib import Path

from rfdetr import RFDETRSegMedium

RUN_DIR = Path(__file__).resolve().parents[1] / "models" / "rfdetr" / "07-06-2026_0909"
CHECKPOINT = RUN_DIR / "checkpoint_best_ema.pth"
OUT_DIR = Path(__file__).resolve().parents[1] / "models" / "rfdetr"

# (resolution, output stem). TWO stacked constraints: the exporter requires the
# shape ÷14, and the backbone requires input ÷ block_size (= patch_size(12) ×
# num_windows(2) = 24). LCM(14, 24) = 168, so valid resolutions are multiples of
# 168: …504, 672, 840, 1008. 672 = 168×4 (done). 840 = 168×5 is the nearest valid
# size to the requested 784 (slightly higher-res → better far/small detection).
TARGETS = [
    (840, "rfdetr_seg_medium_840"),
]


def export_one(resolution: int, stem: str) -> Path:
    assert resolution % 168 == 0, f"{resolution} not a multiple of 168 (÷14 export ∧ ÷24 backbone)"
    print(f"\n=== Exporting RF-DETR-seg-medium @ {resolution}x{resolution} ===", flush=True)
    model = RFDETRSegMedium(pretrain_weights=str(CHECKPOINT), resolution=resolution)

    with tempfile.TemporaryDirectory() as tmp:
        model.export(output_dir=tmp, simplify=True, shape=(resolution, resolution))
        src = Path(tmp) / "inference_model.sim.onnx"
        if not src.exists():  # fall back to the un-simplified graph if simplify was skipped
            src = Path(tmp) / "inference_model.onnx"
        dst = OUT_DIR / f"{stem}.sim.onnx"
        shutil.copy2(src, dst)
    print(f"✅ wrote {dst} ({dst.stat().st_size / 1e6:.1f} MB)", flush=True)
    return dst


def main() -> None:
    assert CHECKPOINT.exists(), f"checkpoint not found: {CHECKPOINT}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for resolution, stem in TARGETS:
        export_one(resolution, stem)
    print("\nAll exports complete.", flush=True)


if __name__ == "__main__":
    main()
