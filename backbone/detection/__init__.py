"""Detection layer: YOLO11 inference via OpenVINO IR (CPU deployment branch).

Importing this package auto-registers three ``Detector`` plugins:
``yolo_openvino`` / ``yolo_openvino_seg`` / ``yolo_openvino_pose`` — all
OpenVINO IR (``model.xml`` + ``model.bin``), CPU/Intel-iGPU. The ``openvino``
package is imported lazily, so this import succeeds even when OpenVINO isn't
installed — only instantiating a detector requires it.

This branch supports **only** OpenVINO IRs: the ONNX Runtime and TensorRT
plugins of the GPU line were removed. Training is out of scope — produce the
``.onnx`` in the training env and convert once with
``ovc model.onnx --output_model model.xml``.
"""

from . import yolo_openvino as _yolo_openvino  # noqa: F401  — registers "yolo_openvino"
from . import (
    yolo_openvino_pose as _yolo_openvino_pose,  # noqa: F401  — registers "yolo_openvino_pose"
)
from . import yolo_openvino_seg as _yolo_openvino_seg  # noqa: F401  — registers "yolo_openvino_seg"
from .postprocess import (
    decode_yolo11_detect,
    decode_yolo11_pose,
    decode_yolo11_seg,
)
from .preprocess import LetterboxResult, batch_letterbox, invert_letterbox_xyxy, letterbox
from .yolo_openvino import YoloOpenvinoDetector
from .yolo_openvino_pose import YoloOpenvinoPoseDetector
from .yolo_openvino_seg import YoloOpenvinoSegDetector

__all__ = [
    "LetterboxResult",
    "YoloOpenvinoDetector",
    "YoloOpenvinoPoseDetector",
    "YoloOpenvinoSegDetector",
    "batch_letterbox",
    "decode_yolo11_detect",
    "decode_yolo11_pose",
    "decode_yolo11_seg",
    "invert_letterbox_xyxy",
    "letterbox",
]
