"""``YoloOnnxPoseDetector`` — YOLO11-pose (person) via ONNX Runtime.

Sibling of :mod:`backbone.detection.yolo_onnx`. Same plugin contract, same
preprocess (letterbox), same provider auto-fallback (CUDA → CPU). The pose ONNX
emits **one** output:

    head  (N, 4 + nc + K*3, A)   ← bbox + class score(s) + K keypoints (x, y, conf)

``decode_yolo11_pose`` turns it into ``Detection`` objects with ``cls="person"``,
``keypoints_uv`` (K, 3), and ``foot_uv`` at the ankle midpoint — the floor-contact
point the homography layer projects to metres. Used alongside the object detector
so the Backbone emits both person and pallet ``Track2D`` (person↔pallet distance).

Inference-only.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from backbone.core.interfaces import Detector, detector_registry
from backbone.core.types import Detection, FramePair
from backbone.shared.ort_session import build_onnx_session

from .onnx_meta import read_embedded_class_names
from .postprocess import decode_yolo11_pose
from .preprocess import batch_letterbox

logger = logging.getLogger(__name__)

DEFAULT_PROVIDERS: tuple[str, ...] = ("CUDAExecutionProvider", "CPUExecutionProvider")


@detector_registry.register("yolo_onnx_pose")
class YoloOnnxPoseDetector(Detector):
    """Run a YOLO11-pose ONNX model (person) on synchronized camera frames."""

    def __init__(
        self,
        onnx_path: str | Path,
        class_names: list[str] | None = None,
        *,
        input_size: tuple[int, int] = (640, 640),
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        kpt_conf: float = 0.3,
        providers: list[str] | None = None,
    ) -> None:
        onnx_file = Path(onnx_path)
        if not onnx_file.exists():
            raise FileNotFoundError(f"YoloOnnxPoseDetector: ONNX file not found at {onnx_file}.")
        self._onnx_path = onnx_file
        # Pose models are single-class (person); default to ["person"] but prefer
        # the model's embedded names if present.
        self._class_names = list(class_names) if class_names else ["person"]
        self._input_size = input_size                  # (w, h)
        self._confidence_threshold = float(confidence_threshold)
        self._iou_threshold = float(iou_threshold)
        self._kpt_conf = float(kpt_conf)
        self._providers = list(providers) if providers else list(DEFAULT_PROVIDERS)

        try:
            self._session = build_onnx_session(onnx_file, providers=self._providers)
        except Exception as exc:
            raise RuntimeError(
                f"YoloOnnxPoseDetector: failed to create ORT session for {onnx_file} "
                f"with providers {self._providers}: {exc}"
            ) from exc

        inputs = self._session.get_inputs()
        if len(inputs) != 1:
            raise ValueError(f"YoloOnnxPoseDetector: model has {len(inputs)} inputs; expected 1")
        outputs = self._session.get_outputs()
        if len(outputs) != 1:
            raise ValueError(
                f"YoloOnnxPoseDetector: model has {len(outputs)} outputs; expected 1 "
                f"(pose head). Did you load a seg/detect ONNX?"
            )
        self._input_name = inputs[0].name
        # Adopt the model's own input size when it's FIXED (dynamic=False export).
        ishape = inputs[0].shape
        if (len(ishape) == 4 and isinstance(ishape[2], int) and isinstance(ishape[3], int)
                and ishape[2] > 0 and ishape[3] > 0):
            model_wh = (int(ishape[3]), int(ishape[2]))
            if model_wh != self._input_size:
                logger.info("YoloOnnxPoseDetector: model expects fixed %dx%d input — overriding %s",
                            model_wh[0], model_wh[1], self._input_size)
                self._input_size = model_wh

        embedded = read_embedded_class_names(self._session)
        if embedded:
            self._class_names = embedded

        self._active_providers = self._session.get_providers()
        if ("CUDAExecutionProvider" in self._providers
                and "CUDAExecutionProvider" not in self._active_providers):
            logger.warning(
                "YoloOnnxPoseDetector: CUDA requested but session fell back to %s — inference SLOW.",
                self._active_providers,
            )
        logger.info("YoloOnnxPoseDetector: loaded %s | providers=%s | classes=%s | input=%dx%d",
                    onnx_file.name, self._active_providers, self._class_names,
                    self._input_size[0], self._input_size[1])

    @property
    def active_providers(self) -> tuple[str, ...]:
        return tuple(self._active_providers)

    @property
    def class_names(self) -> tuple[str, ...]:
        return tuple(self._class_names)

    def warmup(self) -> None:
        dummy = np.zeros((1, 3, self._input_size[1], self._input_size[0]), dtype=np.float32)
        for _ in range(2):
            self._session.run(None, {self._input_name: dummy})

    def detect(self, pair: FramePair) -> dict[str, list[Detection]]:
        if not pair.frames:
            return {}
        cam_ids = list(pair.frames.keys())
        images = [pair.frames[cid].image for cid in cam_ids]

        batch_tensor, lb_results = batch_letterbox(images, target=self._input_size)
        outputs = self._session.run(None, {self._input_name: batch_tensor})
        head_batch = outputs[0]
        if head_batch.ndim != 3 or head_batch.shape[0] != len(cam_ids):
            raise RuntimeError(
                f"YoloOnnxPoseDetector: unexpected head shape {head_batch.shape} "
                f"(expected (N={len(cam_ids)}, 4+nc+K*3, A))"
            )
        result: dict[str, list[Detection]] = {}
        for i, cam_id in enumerate(cam_ids):
            result[cam_id] = decode_yolo11_pose(
                head_batch[i],
                camera_id=cam_id,
                capture_ts=pair.frames[cam_id].capture_ts,
                letterbox_meta=lb_results[i],
                class_names=self._class_names,
                confidence_threshold=self._confidence_threshold,
                iou_threshold=self._iou_threshold,
                kpt_conf=self._kpt_conf,
            )
        return result
