"""Quick ONNX model inspector.

Usage:
    python tools/onnx_inspect.py path/to/yolo11n.onnx

Prints input/output names, shapes, dtypes, and opset — useful before plugging
an unfamiliar export into ``YoloOnnxDetector``. Specifically, you want:

    * Exactly one input named e.g. ``images`` with shape ``[N, 3, 640, 640]``
      (dynamic ``N`` is fine; we batch per FramePair).
    * Exactly one output with the last two dims being ``[4 + nc, A]`` (the
      Ultralytics canonical shape) or ``[A, 4 + nc]`` (the transposed variant).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import onnx


def _format_tv(tv) -> str:
    dtype = onnx.TensorProto.DataType.Name(tv.type.tensor_type.elem_type)
    dims = []
    for d in tv.type.tensor_type.shape.dim:
        if d.dim_param:
            dims.append(d.dim_param)
        elif d.HasField("dim_value"):
            dims.append(str(d.dim_value))
        else:
            dims.append("?")
    return f"{tv.name!r}  dtype={dtype}  shape=[{', '.join(dims)}]"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect an ONNX model.")
    parser.add_argument("onnx_path", type=Path)
    args = parser.parse_args(argv)

    if not args.onnx_path.exists():
        print(f"ONNX file not found: {args.onnx_path}", file=sys.stderr)
        return 1

    model = onnx.load(str(args.onnx_path))
    print(f"file:           {args.onnx_path}")
    print(f"producer:       {model.producer_name} {model.producer_version}".rstrip())
    print(f"ir_version:     {model.ir_version}")
    print(
        "opset_imports: "
        + ", ".join(f"{op.domain or 'ai.onnx'}:{op.version}" for op in model.opset_import)
    )
    print(f"nodes:          {len(model.graph.node)}")
    print()
    print("Inputs:")
    for tv in model.graph.input:
        print(f"  - {_format_tv(tv)}")
    print()
    print("Outputs:")
    for tv in model.graph.output:
        print(f"  - {_format_tv(tv)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
