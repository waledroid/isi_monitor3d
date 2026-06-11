"""``YoloOpenvinoSegDetector`` — YOLO11-seg instance segmentation via OpenVINO.

Sibling of :mod:`backbone.detection.yolo_openvino` (the detect IR variant) and
:mod:`backbone.detection.yolo_onnx_seg` (the ONNX seg variant). Same plugin
contract, same preprocess (letterbox), same ``decode_yolo11_seg`` postprocess.

The seg IR exposes two outputs:

    output 0:  head    (1, 4 + nc + nm, A)      ← bbox + class + nm mask coeffs
    output 1:  protos  (1, nm, mh, mw)          ← mask prototype maps

(Ultralytics exports head first then protos; we identify them by ``ndim`` after
the first inference so a swapped ordering would still work.)

Hardware note: OpenVINO runs on Intel CPU / iGPU. On an NVIDIA GPU box,
``yolo_onnx_seg`` with ``CUDAExecutionProvider`` is the fast path. This plugin
is for an Intel-CPU edge node and for validating the exported IR.

``openvino`` is imported lazily in ``__init__`` so ``import backbone.detection``
succeeds even when OpenVINO isn't installed; only *instantiating* the detector
requires it.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from backbone.core.interfaces import Detector, detector_registry
from backbone.core.types import Detection, FramePair

from .postprocess import decode_yolo11_seg
from .preprocess import batch_letterbox

logger = logging.getLogger(__name__)


@detector_registry.register("yolo_openvino_seg")
class YoloOpenvinoSegDetector(Detector):
    """Run a YOLO11-seg OpenVINO IR on synchronized camera frames (CPU/iGPU)."""

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
        mask_threshold: float = 0.5,
    ) -> None:
        xml_file = Path(model_xml)
        if not xml_file.exists():
            raise FileNotFoundError(
                f"YoloOpenvinoSegDetector: OpenVINO IR not found at {xml_file}."
            )
        if not class_names:
            raise ValueError("YoloOpenvinoSegDetector: class_names must be non-empty")
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
                "YoloOpenvinoSegDetector requires the 'openvino' package. "
                "Add it to the env: `conda env update -f environment.yml -n monitor3d`."
            ) from exc

        self._model_xml = xml_file
        self._class_names = list(class_names)
        self._input_size = input_size                   # (w, h)
        self._confidence_threshold = float(confidence_threshold)
        self._iou_threshold = float(iou_threshold)
        self._keep_classes = list(keep_classes) if keep_classes else None
        self._mask_threshold = float(mask_threshold)

        core = ov.Core()
        model = core.read_model(str(xml_file))
        if len(model.outputs) != 2:
            raise ValueError(
                f"YoloOpenvinoSegDetector: IR has {len(model.outputs)} outputs; expected 2 "
                f"(head + mask prototypes). Did you load a detect-only IR?"
            )
        try:
            self._compiled = core.compile_model(model, device)
            self._device = device
        except Exception:
            logger.warning(
                "YoloOpenvinoSegDetector: device %r unavailable, falling back to CPU", device
            )
            self._compiled = core.compile_model(model, "CPU")
            self._device = "CPU"

        # Identify head vs protos by tensor rank: head is 3D (1, 4+nc+nm, A),
        # protos are 4D (1, nm, mh, mw). Robust to either ONNX-export order.
        out0, out1 = self._compiled.output(0), self._compiled.output(1)
        shape0 = list(out0.partial_shape.get_max_shape())
        shape1 = list(out1.partial_shape.get_max_shape())
        if len(shape0) == 3 and len(shape1) == 4:
            self._head_output, self._protos_output = out0, out1
        elif len(shape0) == 4 and len(shape1) == 3:
            self._head_output, self._protos_output = out1, out0
        else:
            raise RuntimeError(
                f"YoloOpenvinoSegDetector: cannot identify head/protos from output ranks "
                f"{len(shape0)} and {len(shape1)} (expected one 3D, one 4D)."
            )

        logger.info(
            "YoloOpenvinoSegDetector: loaded %s | device=%s | nc=%d",
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

    def _to_head_channels_first(self, raw_one: np.ndarray, nm: int) -> np.ndarray:
        """Return a (4+nc+nm, A) view, accepting either (4+nc+nm, A) or (A, 4+nc+nm)."""
        expected = 4 + len(self._class_names) + nm
        if raw_one.shape[0] == expected:
            return raw_one
        if raw_one.shape[1] == expected:
            return raw_one.transpose(1, 0)
        raise RuntimeError(
            f"YoloOpenvinoSegDetector: head channel dim does not match 4+nc+nm={expected}. "
            f"Got shape {raw_one.shape}. Did class_names match the IR?"
        )

    def detect(self, pair: FramePair) -> dict[str, list[Detection]]:
        if not pair.frames:
            return {}
        # `_input_size` is (w, h); target_hw needed by the decode is (h, w).
        target_hw = (self._input_size[1], self._input_size[0])
        # OpenVINO IR is fixed batch=1 (dynamic=False export), so infer per camera.
        result: dict[str, list[Detection]] = {}
        for cam_id, frame in pair.frames.items():
            batch_tensor, lb_results = batch_letterbox([frame.image], target=self._input_size)
            outs = self._compiled([batch_tensor])
            head_raw = outs[self._head_output]      # (1, 4+nc+nm, A) or (1, A, 4+nc+nm)
            protos_raw = outs[self._protos_output]  # (1, nm, mh, mw)
            if head_raw.ndim != 3 or head_raw.shape[0] != 1:
                raise RuntimeError(
                    f"YoloOpenvinoSegDetector: unexpected head shape {head_raw.shape}"
                )
            if protos_raw.ndim != 4 or protos_raw.shape[0] != 1:
                raise RuntimeError(
                    f"YoloOpenvinoSegDetector: unexpected protos shape {protos_raw.shape}"
                )
            nm = protos_raw.shape[1]
            head_per_image = self._to_head_channels_first(head_raw[0], nm)
            result[cam_id] = decode_yolo11_seg(
                head_per_image,
                protos_raw[0],
                camera_id=cam_id,
                capture_ts=frame.capture_ts,
                letterbox_meta=lb_results[0],
                target_hw=target_hw,
                class_names=self._class_names,
                confidence_threshold=self._confidence_threshold,
                iou_threshold=self._iou_threshold,
                keep_classes=self._keep_classes,
                mask_threshold=self._mask_threshold,
            )
        return result
