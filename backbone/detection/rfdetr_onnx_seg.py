"""``RfdetrOnnxSegDetector`` — RF-DETR instance-segmentation via ONNX Runtime.

A drop-in ``Detector`` plugin that makes an RF-DETR-seg model swappable in
``config/backbone.yaml`` exactly like ``yolo_onnx_seg`` — same registry, same
``Detection`` contract (``bbox_xyxy`` + ``foot_uv`` in source pixels, full-frame
bool ``mask`` in source coords), same CUDA→CPU provider fallback.

RF-DETR differs from YOLO in three ways this plugin must honour, ported faithfully
from the proven trainer inferencer
(``trainer/isidet/src/inference/onnx_inferencer.py``):

1. **Fixed batch-1, square input.** The model input is ``[1, 3, S, S]`` (here
   ``S = 432``). We process **one** image per ``session.run`` — RF-DETR's batch is
   fixed to 1, unlike the YOLO plugin's batched call.
2. **Stretch-resize preprocess.** RGB, plain resize to ``SxS`` (no letterbox / no
   pad), then ImageNet mean/std normalise (DINOv2 backbone). Normalised boxes map
   straight to source pixels by multiplying x/w by ``orig_w`` and y/h by ``orig_h``.
3. **DETR-style, NMS-free head.** ``dets`` (cxcywh-norm), ``labels`` (per-query
   **logits** → sigmoid/focal, COCO-indexed: column 0 = background, columns 1..nc =
   trained classes), ``masks`` (per-query mask logits at ``S/4``). No NMS.

The pure decode lives in :func:`backbone.detection.postprocess.decode_rfdetr_seg`
so it's unit-testable without a real ONNX.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from backbone.core.interfaces import Detector, detector_registry
from backbone.core.types import Detection, FramePair
from backbone.shared.ort_session import build_onnx_session

from .postprocess import decode_rfdetr_seg

logger = logging.getLogger(__name__)

DEFAULT_PROVIDERS: tuple[str, ...] = ("CUDAExecutionProvider", "CPUExecutionProvider")

# RF-DETR's trained classes sit at COCO indices 1=palette, 2=carton, 3=polybag
# (index 0 = background). class_names[i] ↔ logits column i+1.
DEFAULT_CLASS_NAMES: tuple[str, ...] = ("palette", "carton", "polybag")

# ImageNet statistics — required by RF-DETR (DINOv2 backbone). CHW-broadcastable.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


@detector_registry.register("rfdetr_onnx_seg")
class RfdetrOnnxSegDetector(Detector):
    """Run an RF-DETR-seg ONNX model on synchronized camera frames (1 image/run)."""

    def __init__(
        self,
        onnx_path: str | Path,
        class_names: list[str] | None = None,
        *,
        confidence_threshold: float = 0.3,
        mask_threshold: float = 0.5,
        providers: list[str] | None = None,
        input_size: tuple[int, int] | None = None,
    ) -> None:
        onnx_file = Path(onnx_path)
        if not onnx_file.exists():
            raise FileNotFoundError(
                f"RfdetrOnnxSegDetector: ONNX file not found at {onnx_file}."
            )
        self._onnx_path = onnx_file
        self._class_names = list(class_names) if class_names else list(DEFAULT_CLASS_NAMES)
        self._confidence_threshold = float(confidence_threshold)
        self._mask_threshold = float(mask_threshold)
        self._providers = list(providers) if providers else list(DEFAULT_PROVIDERS)

        try:
            self._session = build_onnx_session(onnx_file, providers=self._providers)
        except Exception as exc:
            raise RuntimeError(
                f"RfdetrOnnxSegDetector: failed to create ORT session for {onnx_file} "
                f"with providers {self._providers}: {exc}"
            ) from exc

        inputs = self._session.get_inputs()
        if len(inputs) != 1:
            raise ValueError(
                f"RfdetrOnnxSegDetector: model has {len(inputs)} inputs; expected 1"
            )
        self._input_name = inputs[0].name

        # Read the model's FIXED input HxW from the ONNX (don't hardcode 432).
        # Input shape is [batch, 3, H, W]; H/W are ints when static.
        ishape = inputs[0].shape
        if input_size is not None:
            self._input_size = (int(input_size[0]), int(input_size[1]))   # (w, h)
        elif (len(ishape) == 4 and isinstance(ishape[2], int) and isinstance(ishape[3], int)
                and ishape[2] > 0 and ishape[3] > 0):
            self._input_size = (int(ishape[3]), int(ishape[2]))           # (w, h)
        else:
            raise ValueError(
                f"RfdetrOnnxSegDetector: model input shape {ishape} has no fixed H/W and "
                f"no input_size was supplied. RF-DETR needs a fixed square input."
            )

        self._output_names = [o.name for o in self._session.get_outputs()]
        # Map RF-DETR's named outputs (dets / labels / masks) robustly, not by
        # positional order — exporters may reorder them.
        if "dets" not in self._output_names or "labels" not in self._output_names:
            raise ValueError(
                f"RfdetrOnnxSegDetector: expected RF-DETR outputs 'dets' and 'labels'; "
                f"got {self._output_names}. Is this an RF-DETR ONNX?"
            )
        self._has_masks = "masks" in self._output_names

        self._active_providers = self._session.get_providers()
        if ("CUDAExecutionProvider" in self._providers
                and "CUDAExecutionProvider" not in self._active_providers
                # native .engine sessions ARE the GPU fast path
                and "TensorrtEngineFile" not in self._active_providers):
            logger.warning(
                "RfdetrOnnxSegDetector: CUDA was requested but the session fell back to %s — "
                "inference will be SLOW. Check the onnxruntime-gpu build / CUDA libs.",
                self._active_providers,
            )
        logger.info(
            "RfdetrOnnxSegDetector: loaded %s | providers=%s | nc=%d | input=%dx%d | masks=%s",
            onnx_file.name, self._active_providers, len(self._class_names),
            self._input_size[0], self._input_size[1], self._has_masks,
        )

    @property
    def active_providers(self) -> tuple[str, ...]:
        return tuple(self._active_providers)

    @property
    def class_names(self) -> tuple[str, ...]:
        return tuple(self._class_names)

    @property
    def supports_batch(self) -> bool:
        """RF-DETR exports are fixed batch-1 (``[1,3,S,S]``) and the decode loops
        per-image with a hardcoded ``[0]``, so it can never be batched. Constant
        False — these zones gracefully fall back to per-zone inference."""
        return False

    def warmup(self) -> None:
        dummy = np.zeros((1, 3, self._input_size[1], self._input_size[0]), dtype=np.float32)
        for _ in range(2):
            self._session.run(None, {self._input_name: dummy})

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """BGR (H, W, 3) uint8 → (1, 3, S, S) float32, RF-DETR style.

        Stretch-resize to the model's square input (no letterbox), RGB,
        ImageNet mean/std normalise. Mirrors the trainer inferencer's rfdetr
        preprocess branch.
        """
        w, h = self._input_size
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        chw = resized.transpose(2, 0, 1).astype(np.float32) / 255.0
        chw = (chw - _IMAGENET_MEAN) / _IMAGENET_STD
        return np.ascontiguousarray(chw[np.newaxis, ...])

    def detect(self, pair: FramePair) -> dict[str, list[Detection]]:
        if not pair.frames:
            return {}
        result: dict[str, list[Detection]] = {}
        for cam_id, frame in pair.frames.items():
            image = frame.image
            src_h, src_w = image.shape[:2]
            tensor = self._preprocess(image)
            outputs = self._session.run(None, {self._input_name: tensor})
            out_map = {name: outputs[i] for i, name in enumerate(self._output_names)}

            dets = out_map["dets"][0]        # (num_queries, 4) cxcywh-norm
            labels = out_map["labels"][0]    # (num_queries, head_nc) logits
            masks = out_map["masks"][0] if self._has_masks else None  # (num_queries, mh, mw)

            result[cam_id] = decode_rfdetr_seg(
                dets,
                labels,
                masks,
                camera_id=cam_id,
                capture_ts=frame.capture_ts,
                source_wh=(src_w, src_h),
                class_names=self._class_names,
                confidence_threshold=self._confidence_threshold,
                mask_threshold=self._mask_threshold,
            )
        return result
