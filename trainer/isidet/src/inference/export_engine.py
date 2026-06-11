"""Model Export Pipeline — converts trained weights to optimized deployment formats.

Usage:
    # From a model directory (auto-discovers ONNX)
    python -m src.inference.export_engine --model-dir models/rfdetr/31-03-2026_1117 --format all
    python -m src.inference.export_engine --model-dir runs/segment/models/yolo/yolo26m_640_200/weights --format onnx openvino

    # From raw weights (exports ONNX first)
    python -m src.inference.export_engine --weights models/yolo/best.pt --format all
"""

import argparse
import contextlib
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

VALID_FORMATS = {'onnx', 'openvino', 'tensorrt', 'all'}


# ── ONNX Discovery & Optimization ───────────────────────────────────────────

def discover_onnx(model_dir: Path) -> Path | None:
    """Find the best available ONNX file in a model directory.

    Prefers .sim.onnx (already optimized) over raw .onnx.
    """
    model_dir = Path(model_dir)

    # Prefer simplified
    sim_files = list(model_dir.rglob('*.sim.onnx'))
    if sim_files:
        logger.info(f"Found simplified ONNX: {sim_files[0]}")
        return sim_files[0]

    # Fall back to raw ONNX
    onnx_files = list(model_dir.rglob('*.onnx'))
    if onnx_files:
        logger.info(f"Found ONNX: {onnx_files[0]}")
        return onnx_files[0]

    return None


def optimize_onnx(onnx_path: Path) -> Path:
    """Run onnxsim on an ONNX model. Returns path to simplified model.

    If the input is already a .sim.onnx, returns it unchanged.
    """
    onnx_path = Path(onnx_path)

    if onnx_path.name.endswith('.sim.onnx'):
        logger.info(f"Already simplified: {onnx_path}")
        return onnx_path

    import onnx
    import onnxsim

    sim_path = onnx_path.with_suffix('.sim.onnx')
    if sim_path.exists():
        logger.info(f"Simplified version already exists: {sim_path}")
        return sim_path

    logger.info(f"Simplifying {onnx_path} ...")
    model = onnx.load(str(onnx_path))
    model_sim, check = onnxsim.simplify(model)
    if not check:
        logger.warning("onnxsim simplification check failed — using original")
        return onnx_path

    onnx.save(model_sim, str(sim_path))
    orig_mb = onnx_path.stat().st_size / 1024**2
    sim_mb = sim_path.stat().st_size / 1024**2
    logger.info(f"Saved {sim_path} ({orig_mb:.1f} MB → {sim_mb:.1f} MB)")
    return sim_path


# ── Export from Raw Weights ──────────────────────────────────────────────────

def export_from_weights(weights_path: Path, imgsz: int = None) -> Path:
    """Export ONNX from native .pt (YOLO) or .pth (RF-DETR) weights.

    Args:
        weights_path: Path to .pt or .pth weights.
        imgsz: Override image size. If None, reads from the model's training config.

    Returns path to the generated ONNX file.
    """
    weights_path = Path(weights_path)
    ext = weights_path.suffix.lower()

    if ext == '.pt':
        return _export_yolo(weights_path, imgsz)
    elif ext == '.pth':
        return _export_rfdetr(weights_path, imgsz)
    else:
        raise ValueError(f"Unsupported weight format: {ext}. Expected .pt or .pth")


def _export_yolo(weights_path: Path, imgsz: int = None) -> Path:
    """Export YOLO .pt → .onnx via Ultralytics."""
    from ultralytics import YOLO

    model = YOLO(str(weights_path))

    # Read imgsz from the model's training config if not provided
    if imgsz is None:
        imgsz = model.overrides.get('imgsz', 640)

    logger.info(f"Exporting YOLO: {weights_path} → ONNX (imgsz={imgsz})")
    export_path = model.export(
        format='onnx',
        imgsz=imgsz,
        opset=12,
        nms=True,
        simplify=True,
        dynamic=False,
    )
    onnx_path = Path(export_path)
    logger.info(f"YOLO ONNX exported: {onnx_path} ({onnx_path.stat().st_size / 1024**2:.1f} MB, imgsz={imgsz})")
    return onnx_path


def _export_rfdetr(weights_path: Path, imgsz: int = None) -> Path:
    """Export RF-DETR .pth → .onnx via rfdetr package."""
    from rfdetr import RFDETRBase

    output_dir = weights_path.parent

    # RF-DETR uses resolution in its constructor
    kwargs = {'pretrain_weights': str(weights_path)}
    if imgsz is not None:
        kwargs['resolution'] = imgsz

    logger.info(f"Exporting RF-DETR: {weights_path} → ONNX" +
                (f" (imgsz={imgsz})" if imgsz else " (default resolution)"))

    model = RFDETRBase(**kwargs)
    model.export(output_dir=str(output_dir), simplify=True)

    onnx_path = output_dir / 'inference_model.onnx'
    if not onnx_path.exists():
        raise FileNotFoundError(f"RF-DETR export did not produce {onnx_path}")

    logger.info(f"RF-DETR ONNX exported: {onnx_path} ({onnx_path.stat().st_size / 1024**2:.1f} MB)")
    return onnx_path


# ── Format Converters ────────────────────────────────────────────────────────

@contextlib.contextmanager
def _silence_torch_export_pt2_probe():
    """Mute ``torch.export``'s pt2-archive probe failure.

    ``openvino.convert_model()`` probes the input as a PyTorch pt2
    archive before falling back to the ONNX frontend. On an ONNX file
    the probe fails and logs a multi-line ``PytorchStreamReader`` stack
    trace at WARNING level, even though the fallback path succeeds.
    Scope the level bump to just the conversion call so unrelated
    torch warnings elsewhere in the run stay visible.
    """
    lg = logging.getLogger("torch.export")
    prev_level = lg.level
    lg.setLevel(logging.ERROR)
    try:
        yield
    finally:
        lg.setLevel(prev_level)


def convert_openvino(onnx_path: Path, output_dir: Path = None) -> Path:
    """Convert ONNX → OpenVINO IR (.xml + .bin)."""
    import openvino as ov

    onnx_path = Path(onnx_path)
    if output_dir is None:
        output_dir = onnx_path.parent / 'openvino'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    xml_path = output_dir / 'model.xml'
    if xml_path.exists():
        logger.info(f"OpenVINO IR already exists: {xml_path}")
        return xml_path

    logger.info(f"Converting {onnx_path} → OpenVINO IR ...")
    with _silence_torch_export_pt2_probe():
        ov_model = ov.convert_model(str(onnx_path))
    ov.save_model(ov_model, str(xml_path))

    bin_path = output_dir / 'model.bin'
    xml_mb = xml_path.stat().st_size / 1024**2
    bin_mb = bin_path.stat().st_size / 1024**2 if bin_path.exists() else 0
    logger.info(f"Saved OpenVINO IR: {xml_path} ({xml_mb:.1f} MB) + model.bin ({bin_mb:.1f} MB)")
    return xml_path


def convert_tensorrt(onnx_path: Path, output_dir: Path = None) -> Path | None:
    """Convert ONNX → TensorRT engine via trtexec.

    Returns None if TensorRT is not available (graceful skip).
    """
    import shutil
    import subprocess

    onnx_path = Path(onnx_path)
    if output_dir is None:
        output_dir = onnx_path.parent / 'tensorrt'
    output_dir = Path(output_dir)

    engine_path = output_dir / 'model.engine'
    if engine_path.exists():
        logger.info(f"TensorRT engine already exists: {engine_path}")
        return engine_path

    trtexec = shutil.which('trtexec')
    if trtexec is None:
        logger.warning("trtexec not found — skipping TensorRT conversion. "
                        "Install TensorRT and ensure trtexec is on PATH.")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Converting {onnx_path} → TensorRT engine ...")

    cmd = [
        trtexec,
        f'--onnx={onnx_path}',
        f'--saveEngine={engine_path}',
        '--fp16',
        '--workspace=4096',
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"trtexec failed:\n{result.stderr[-500:]}")
        return None

    logger.info(f"Saved TensorRT engine: {engine_path} ({engine_path.stat().st_size / 1024**2:.1f} MB)")
    return engine_path


# ── Pipeline Orchestrator ────────────────────────────────────────────────────

def run_pipeline(model_dir: Path = None, weights: Path = None, formats: set = None,
                 imgsz: int = None):
    """Run the full export pipeline.

    Args:
        model_dir: Directory containing trained model (searches for ONNX or weights).
        weights: Path to raw .pt or .pth weights (used if no ONNX found).
        formats: Set of target formats: {'onnx', 'openvino', 'tensorrt', 'all'}.
        imgsz: Override image size for ONNX export (reads from model config if None).
    """
    if formats is None or 'all' in formats:
        formats = {'onnx', 'openvino', 'tensorrt'}

    # Step 1: Get an ONNX model
    onnx_path = None
    if model_dir:
        model_dir = Path(model_dir)
        onnx_path = discover_onnx(model_dir)

    if onnx_path is None and weights:
        weights = Path(weights)
        if not weights.exists():
            logger.error(f"Weights file not found: {weights}")
            return
        onnx_path = export_from_weights(weights, imgsz=imgsz)
        model_dir = onnx_path.parent

    if onnx_path is None:
        # Last resort: look for .pt or .pth in model_dir
        if model_dir:
            for ext in ('*.pt', '*.pth'):
                found = list(model_dir.rglob(ext))
                if found:
                    logger.info(f"No ONNX found — exporting from {found[0]}")
                    onnx_path = export_from_weights(found[0], imgsz=imgsz)
                    break

    if onnx_path is None:
        logger.error("No ONNX file found and no weights to export from.")
        return

    # Step 2: Optimize ONNX
    if 'onnx' in formats:
        onnx_path = optimize_onnx(onnx_path)
    else:
        # Even if not requested, prefer .sim.onnx for downstream conversions
        sim = onnx_path.with_suffix('.sim.onnx')
        if sim.exists():
            onnx_path = sim

    # Step 3: Convert to target formats
    results = {'onnx': onnx_path}

    if 'openvino' in formats:
        try:
            results['openvino'] = convert_openvino(onnx_path)
        except Exception as e:
            logger.error(f"OpenVINO conversion failed: {e}")

    if 'tensorrt' in formats:
        try:
            results['tensorrt'] = convert_tensorrt(onnx_path)
        except Exception as e:
            logger.error(f"TensorRT conversion failed: {e}")

    # Summary
    logger.info("── Export Summary ──")
    for fmt, path in results.items():
        status = f"  {fmt}: {path}" if path else f"  {fmt}: SKIPPED"
        logger.info(status)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Export trained models to optimized deployment formats.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--model-dir', type=Path,
                        help='Directory containing trained model (searches for ONNX or weights)')
    parser.add_argument('--weights', type=Path,
                        help='Path to raw .pt or .pth weights')
    parser.add_argument('--format', nargs='+', default=['all'],
                        choices=sorted(VALID_FORMATS),
                        help='Target formats (default: all)')
    parser.add_argument('--imgsz', type=int, default=None,
                        help='Override image size for ONNX export (default: read from model)')

    args = parser.parse_args()

    if not args.model_dir and not args.weights:
        parser.error("Provide --model-dir or --weights")

    formats = set(args.format)
    run_pipeline(model_dir=args.model_dir, weights=args.weights, formats=formats, imgsz=args.imgsz)


if __name__ == '__main__':
    main()
