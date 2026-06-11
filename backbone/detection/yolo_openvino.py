"""``YoloOpenvinoDetector`` — YOLO11 inference via OpenVINO (Intel CPU / iGPU).

Sibling of ``YoloOnnxDetector`` for Intel-CPU edge nodes. The OpenVINO IR
(``model.xml`` + ``model.bin``) is exported alongside the ONNX with ``nms=False``
(see trainer T1.1), so it carries the **same raw YOLO11-detect head**
``(1, 4+nc, 8400)``. This detector therefore reuses ``batch_letterbox`` and
``decode_yolo11_detect`` verbatim — only the inference call differs from the ONNX
plugin.

Hardware note: OpenVINO runs on Intel CPU / iGPU. On the NVIDIA RTX 5070 dev box
it runs on CPU only (it does NOT use the NVIDIA GPU) — there ``yolo_onnx`` with
CUDAExecutionProvider is the fast path. ``yolo_openvino`` is for a future Intel
edge node and for validating the exported IR.

``openvino`` is imported lazily in ``__init__`` so ``import backbone.detection``
(which registers this plugin) succeeds even when OpenVINO isn't installed; only
*instantiating* the detector requires it.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from backbone.core.interfaces import Detector, detector_registry
from backbone.core.types import Detection, FramePair

from .postprocess import decode_yolo11_detect
from .preprocess import batch_letterbox

logger = logging.getLogger(__name__)


@detector_registry.register("yolo_openvino")
class YoloOpenvinoDetector(Detector):
    """Run a YOLO11-detect OpenVINO IR on synchronized camera frames (CPU/iGPU)."""

    def __init__(
        self,
        model_xml: str | Path,
        class_names: list[str],
        *,
        input_size: tuple[int, int] = (640, 640),
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        keep_classes: list[str] | None = None,
        device: str = "AUTO",
    ) -> None:
        xml_file = Path(model_xml)
        if not xml_file.exists():
            raise FileNotFoundError(
                f"YoloOpenvinoDetector: OpenVINO IR not found at {xml_file}. "
                f"Export from the training env with `format=openvino` (alongside the .bin)."
            )
        if not class_names:
            raise ValueError("YoloOpenvinoDetector: class_names must be non-empty")
        if keep_classes is not None:
            unknown = [c for c in keep_classes if c not in class_names]
            if unknown:
                raise ValueError(
                    f"keep_classes contains names not in class_names: {unknown}. "
                    f"Available: {class_names}"
                )

        try:
            import openvino as ov  # lazy — keeps backbone.detection importable without it
        except ImportError as exc:
            raise RuntimeError(
                "YoloOpenvinoDetector requires the 'openvino' package. "
                "Add it to the env: `conda env update -f environment.yml -n monitor3d`."
            ) from exc

        self._model_xml = xml_file
        self._class_names = list(class_names)
        self._input_size = input_size
        self._confidence_threshold = float(confidence_threshold)
        self._iou_threshold = float(iou_threshold)
        self._keep_classes = list(keep_classes) if keep_classes else None

        core = ov.Core()
        model = core.read_model(str(xml_file))
        try:
            self._compiled = core.compile_model(model, device)
            self._device = device
        except Exception:
            logger.warning(
                "YoloOpenvinoDetector: device %r unavailable, falling back to CPU", device
            )
            self._compiled = core.compile_model(model, "CPU")
            self._device = "CPU"
        self._output = self._compiled.output(0)

        logger.info(
            "YoloOpenvinoDetector: loaded %s | device=%s | nc=%d",
            xml_file.name, self._device, len(self._class_names),
        )

    @property
    def class_names(self) -> tuple[str, ...]:
        return tuple(self._class_names)

    @property
    def device(self) -> str:
        return self._device

    def warmup(self) -> None:
        """Run a dummy inference to stabilize timings."""
        dummy = np.zeros((1, 3, self._input_size[1], self._input_size[0]), dtype=np.float32)
        self._compiled([dummy])

    def _to_channels_first(self, raw_one: np.ndarray) -> np.ndarray:
        """Return a (4+nc, A) view, accepting either (4+nc, A) or (A, 4+nc)."""
        expected = 4 + len(self._class_names)
        if raw_one.shape[0] == expected:
            return raw_one
        if raw_one.shape[1] == expected:
            return raw_one.transpose(1, 0)
        raise RuntimeError(
            f"YoloOpenvinoDetector: output channel dim does not match nc={len(self._class_names)}. "
            f"Got shape {raw_one.shape}. Did you pass the right class_names for this model?"
        )

    def detect(self, pair: FramePair) -> dict[str, list[Detection]]:
        if not pair.frames:
            return {}
        # OpenVINO IR is fixed batch=1 (dynamic=False export), so infer per camera.
        result: dict[str, list[Detection]] = {}
        for cam_id, frame in pair.frames.items():
            batch_tensor, lb_results = batch_letterbox([frame.image], target=self._input_size)
            raw = self._compiled([batch_tensor])[self._output]  # (1, 4+nc, A) or (1, A, 4+nc)
            if raw.ndim != 3 or raw.shape[0] != 1:
                raise RuntimeError(
                    f"YoloOpenvinoDetector: unexpected output shape {raw.shape} (expected (1, 4+nc, A))"
                )
            per_image = self._to_channels_first(raw[0])
            result[cam_id] = decode_yolo11_detect(
                per_image,
                camera_id=cam_id,
                capture_ts=frame.capture_ts,
                letterbox_meta=lb_results[0],
                class_names=self._class_names,
                confidence_threshold=self._confidence_threshold,
                iou_threshold=self._iou_threshold,
                keep_classes=self._keep_classes,
            )
        return result
