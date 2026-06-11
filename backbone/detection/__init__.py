"""Detection layer: YOLO11 inference via ONNX Runtime (and OpenVINO).

Importing this package auto-registers four ``Detector`` plugins:
``yolo_onnx`` / ``yolo_onnx_seg`` (ONNX Runtime, GPU via CUDAExecutionProvider)
and ``yolo_openvino`` / ``yolo_openvino_seg`` (OpenVINO IR, CPU/Intel-iGPU).
(``yolo_openvino*`` imports the ``openvino`` package lazily, so this import
succeeds even when OpenVINO isn't installed — only instantiating it requires it.)

Training is **out of scope** for this codebase — the user produces ``.onnx`` /
OpenVINO ``.xml`` artefacts in a separate environment (Ultralytics or equivalent)
and references them from ``config/backbone.yaml``.
"""

from . import rfdetr_onnx_seg as _rfdetr_onnx_seg  # noqa: F401  — registers "rfdetr_onnx_seg"
from . import yolo_onnx as _yolo_onnx  # noqa: F401  — registers "yolo_onnx"
from . import yolo_onnx_pose as _yolo_onnx_pose  # noqa: F401  — registers "yolo_onnx_pose"
from . import yolo_onnx_seg as _yolo_onnx_seg  # noqa: F401  — registers "yolo_onnx_seg"
from . import yolo_openvino as _yolo_openvino  # noqa: F401  — registers "yolo_openvino"
from . import yolo_openvino_seg as _yolo_openvino_seg  # noqa: F401  — registers "yolo_openvino_seg"
from .postprocess import (
    decode_rfdetr_seg,
    decode_yolo11_detect,
    decode_yolo11_pose,
    decode_yolo11_seg,
)
from .preprocess import LetterboxResult, batch_letterbox, invert_letterbox_xyxy, letterbox
from .rfdetr_onnx_seg import RfdetrOnnxSegDetector
from .yolo_onnx import YoloOnnxDetector
from .yolo_onnx_pose import YoloOnnxPoseDetector
from .yolo_onnx_seg import YoloOnnxSegDetector
from .yolo_openvino import YoloOpenvinoDetector
from .yolo_openvino_seg import YoloOpenvinoSegDetector

__all__ = [
    "LetterboxResult",
    "RfdetrOnnxSegDetector",
    "YoloOnnxDetector",
    "YoloOnnxPoseDetector",
    "YoloOnnxSegDetector",
    "YoloOpenvinoDetector",
    "YoloOpenvinoSegDetector",
    "batch_letterbox",
    "decode_rfdetr_seg",
    "decode_yolo11_detect",
    "decode_yolo11_pose",
    "decode_yolo11_seg",
    "invert_letterbox_xyxy",
    "letterbox",
]
