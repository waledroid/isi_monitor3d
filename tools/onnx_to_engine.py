"""Build a native TensorRT ``.engine`` from an ``.onnx`` — the fast-path
compile step that replaces the lazy TRT-EP minute-long build at first
inference with a one-time offline build.

The engine is a PER-MACHINE artifact (GPU arch + TensorRT version specific);
the portable ``.onnx`` remains the source of truth. Alongside the engine a
sidecar ``<engine>.json`` records provenance (source hash, TRT version, GPU,
profile, output names) — the runtime validates it and refuses/falls back
honestly instead of crashing on a foreign engine.

Usage (GPU required; run while the live stack is stopped):
    python tools/onnx_to_engine.py MODEL.onnx \
        --imgsz 320 --min-batch 1 --opt-batch 8 --max-batch 32
    # → MODEL.engine + MODEL.engine.json beside the source (or --output PATH)

A single dynamic-batch optimization profile spans every isistream batch
bucket (1..32), so ONE engine serves all zone-crop batch sizes — no
per-shape engine zoo. For the pose model use e.g. --imgsz 640 --max-batch 4.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


def _gpu_name() -> str | None:
    try:
        from cuda.bindings import runtime as cudart

        err, props = cudart.cudaGetDeviceProperties(0)
        if int(err) == 0:
            name = props.name
            return name.decode() if isinstance(name, bytes) else str(name)
    except Exception:
        pass
    return None


def _class_names(onnx_path: Path) -> list[str] | None:
    """The Ultralytics-embedded names, read WITHOUT building a session."""
    import ast

    import onnx

    try:
        model = onnx.load(str(onnx_path), load_external_data=False)
        for prop in model.metadata_props:
            if prop.key == "names":
                parsed = ast.literal_eval(prop.value)
                if isinstance(parsed, dict):
                    return [str(parsed[k]) for k in sorted(parsed)]
                if isinstance(parsed, (list, tuple)):
                    return [str(n) for n in parsed]
    except Exception:
        pass
    return None


def build_engine(onnx_path: Path, output: Path, *, imgsz: int | None,
                 min_batch: int, opt_batch: int, max_batch: int,
                 fp16: bool, workspace_gb: float) -> dict:
    """Compile + serialize; returns the sidecar dict. Raises on any failure."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errs = "; ".join(str(parser.get_error(i))
                         for i in range(parser.num_errors))
        raise RuntimeError(f"onnx parse failed: {errs}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                                 int(workspace_gb * (1 << 30)))
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    profile_desc: dict[str, list[list[int]]] = {}
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        dims = list(inp.shape)
        h = w = imgsz
        if imgsz is None:
            if any(int(d) < 0 for d in dims[2:]):
                raise SystemExit(
                    f"input {inp.name} has dynamic spatial dims {dims} — "
                    "pass --imgsz")
            h, w = int(dims[2]), int(dims[3])
        lo = [min_batch, int(dims[1]), h, w]
        op = [opt_batch, int(dims[1]), h, w]
        hi = [max_batch, int(dims[1]), h, w]
        profile.set_shape(inp.name, lo, op, hi)
        profile_desc[inp.name] = [lo, op, hi]
    config.add_optimization_profile(profile)

    t0 = time.time()
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise RuntimeError("TensorRT build_serialized_network returned None")
    output.write_bytes(bytes(blob))

    sidecar = {
        "source_onnx": str(onnx_path.resolve()),
        "source_sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
        "tensorrt_version": trt.__version__,
        "gpu_name": _gpu_name(),
        "fp16": fp16,
        "profile": profile_desc,
        "outputs": [network.get_output(i).name
                    for i in range(network.num_outputs)],
        "class_names": _class_names(onnx_path),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "build_seconds": round(time.time() - t0, 1),
    }
    Path(str(output) + ".json").write_text(json.dumps(sidecar, indent=2))
    return sidecar


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("onnx", type=Path)
    ap.add_argument("--output", type=Path, default=None,
                    help="engine path (default: beside the onnx, .engine)")
    ap.add_argument("--imgsz", type=int, default=None,
                    help="square input size for dynamic-spatial models "
                         "(e.g. 320 for the zone detector, 640 for pose)")
    ap.add_argument("--min-batch", type=int, default=1)
    ap.add_argument("--opt-batch", type=int, default=8)
    ap.add_argument("--max-batch", type=int, default=32,
                    help="cover isistream's largest batch bucket (default 32)")
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--workspace-gb", type=float, default=4.0)
    args = ap.parse_args(argv)

    if not args.onnx.exists():
        ap.error(f"{args.onnx} not found")
    output = args.output or args.onnx.with_suffix(".engine")

    print(f"building {output.name} from {args.onnx.name} "
          f"(imgsz={args.imgsz}, batch {args.min_batch}/{args.opt_batch}/"
          f"{args.max_batch}, fp16={not args.no_fp16}) — takes minutes …",
          flush=True)
    sidecar = build_engine(
        args.onnx, output, imgsz=args.imgsz, min_batch=args.min_batch,
        opt_batch=args.opt_batch, max_batch=args.max_batch,
        fp16=not args.no_fp16, workspace_gb=args.workspace_gb)
    print(f"done in {sidecar['build_seconds']}s → {output} "
          f"({output.stat().st_size / 1e6:.1f} MB) + sidecar "
          f"{output.name}.json [gpu={sidecar['gpu_name']}, "
          f"trt={sidecar['tensorrt_version']}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
