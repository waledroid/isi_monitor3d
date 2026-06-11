"""``YoloOnnxDetector`` — YOLO11 inference via ONNX Runtime.

Inference-only. The Backbone never trains; the training environment produces
``yolo*.onnx`` artefacts that are dropped in and referenced from
``config/backbone.yaml``. Same artefact runs on the RTX 5070 dev box (Blackwell
sm_120 via CUDAExecutionProvider) and on the Jetson Orin NX (Jetson onnxruntime
wheel) — no engine rebuild required, which is the whole reason this Backbone
picked ORT over TensorRT.

Pipeline per ``FramePair``:

    images BGR ── letterbox ──► (N, 3, 640, 640) RGB float32
                                       │
                                       ▼
                       ort.InferenceSession.run
                                       │
                                       ▼
                   (N, 4 + nc, 8400)  ──► postprocess (per camera)
                                       │
                                       ▼
                       dict[cam_id, list[Detection]]
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from backbone.core.interfaces import Detector, detector_registry
from backbone.core.types import Detection, FramePair
from backbone.shared.ort_session import build_onnx_session

from .onnx_meta import read_embedded_class_names
from .postprocess import decode_yolo11_detect
from .preprocess import batch_letterbox

logger = logging.getLogger(__name__)

DEFAULT_PROVIDERS: tuple[str, ...] = ("CUDAExecutionProvider", "CPUExecutionProvider")


@detector_registry.register("yolo_onnx")
class YoloOnnxDetector(Detector):
    """Run a YOLO11-detect ONNX model on synchronized camera frames."""

    def __init__(
        self,
        onnx_path: str | Path,
        class_names: list[str] | None = None,
        *,
        input_size: tuple[int, int] = (640, 640),
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        keep_classes: list[str] | None = None,
        providers: list[str] | None = None,
    ) -> None:
        onnx_file = Path(onnx_path)
        if not onnx_file.exists():
            raise FileNotFoundError(
                f"YoloOnnxDetector: ONNX file not found at {onnx_file}. "
                f"Export from the training env with `yolo export model=*.pt format=onnx`."
            )
        # class_names / keep_classes are resolved AFTER the session loads so the
        # model's embedded names (if any) can take over — see below.
        self._onnx_path = onnx_file
        self._class_names = list(class_names) if class_names else None
        self._keep_classes_requested = list(keep_classes) if keep_classes else None
        self._keep_classes = None
        self._input_size = input_size
        self._confidence_threshold = float(confidence_threshold)
        self._iou_threshold = float(iou_threshold)
        self._providers = list(providers) if providers else list(DEFAULT_PROVIDERS)

        try:
            self._session = build_onnx_session(onnx_file, providers=self._providers)
        except Exception as exc:
            raise RuntimeError(
                f"YoloOnnxDetector: failed to create ORT session for {onnx_file} "
                f"with providers {self._providers}: {exc}"
            ) from exc

        inputs = self._session.get_inputs()
        if len(inputs) != 1:
            raise ValueError(
                f"YoloOnnxDetector: model has {len(inputs)} inputs; expected 1"
            )
        self._input_name = inputs[0].name
        # ONNX may declare dynamic batch ('N') or fixed; we validate shape at run time.
        self._input_shape = inputs[0].shape
        # Adopt the model's own input spatial size when it's FIXED (e.g. a YOLO
        # exported with dynamic=False imgsz=1024). The shape is [batch, 3, H, W];
        # H/W are ints when static, strings/-1 when dynamic. Letterboxing to a
        # different size triggers "Got invalid dimensions for input ... Got 640
        # Expected 1024" at run().
        ishape = self._input_shape
        if (len(ishape) == 4 and isinstance(ishape[2], int) and isinstance(ishape[3], int)
                and ishape[2] > 0 and ishape[3] > 0):
            model_wh = (int(ishape[3]), int(ishape[2]))   # (w, h)
            if model_wh != self._input_size:
                logger.info(
                    "YoloOnnxDetector: model expects fixed %dx%d input — overriding "
                    "configured input_size %s", model_wh[0], model_wh[1], self._input_size,
                )
                self._input_size = model_wh

        # Resolve class names: prefer the names embedded in the ONNX (self-
        # configuring — immune to config drift), fall back to the configured names.
        embedded = read_embedded_class_names(self._session)
        if embedded:
            if self._class_names and self._class_names != embedded:
                logger.info(
                    "YoloOnnxDetector: using class names embedded in the model %s "
                    "(overriding configured %s)", embedded, self._class_names,
                )
            self._class_names = embedded
        if not self._class_names:
            raise ValueError(
                "YoloOnnxDetector: no class names — the model embeds none and none "
                "were configured."
            )
        # keep_classes is a best-effort filter: drop entries the model doesn't have
        # (stale config); 'nothing valid' means no filter (keep all).
        if self._keep_classes_requested:
            valid = [c for c in self._keep_classes_requested if c in self._class_names]
            dropped = [c for c in self._keep_classes_requested if c not in self._class_names]
            if dropped:
                logger.warning(
                    "YoloOnnxDetector: keep_classes %s not in model classes %s — ignored",
                    dropped, self._class_names,
                )
            self._keep_classes = valid or None

        self._active_providers = self._session.get_providers()
        if ("CUDAExecutionProvider" in self._providers
                and "CUDAExecutionProvider" not in self._active_providers):
            logger.warning(
                "YoloOnnxDetector: CUDA was requested but the session fell back to %s — "
                "inference will be SLOW. Check the onnxruntime-gpu build / CUDA libs.",
                self._active_providers,
            )
        logger.info(
            "YoloOnnxDetector: loaded %s | providers=%s | input=%s shape=%s | nc=%d",
            onnx_file.name,
            self._active_providers,
            self._input_name,
            self._input_shape,
            len(self._class_names),
        )

    @property
    def active_providers(self) -> tuple[str, ...]:
        return tuple(self._active_providers)

    @property
    def class_names(self) -> tuple[str, ...]:
        return tuple(self._class_names)

    def warmup(self) -> None:
        """Run a couple of dummy inferences to JIT-stabilize timings."""
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
        if not outputs:
            raise RuntimeError("YoloOnnxDetector: ORT session returned no outputs")
        raw = outputs[0]
        # YOLO11 detect: (N, 4+nc, A). Some exports emit (N, A, 4+nc); detect either.
        if raw.ndim != 3 or raw.shape[0] != len(cam_ids):
            raise RuntimeError(
                f"YoloOnnxDetector: unexpected output shape {raw.shape} "
                f"(expected (N={len(cam_ids)}, 4+nc, A))"
            )
        expected_channels = 4 + len(self._class_names)
        if raw.shape[1] == expected_channels:
            per_image = raw  # (N, 4+nc, A) — Ultralytics canonical
        elif raw.shape[2] == expected_channels:
            per_image = raw.transpose(0, 2, 1)  # (N, A, 4+nc) → (N, 4+nc, A)
        else:
            raise RuntimeError(
                f"YoloOnnxDetector: output channel dim does not match nc={len(self._class_names)}. "
                f"Got shape {raw.shape}. Did you pass the right class_names for this model?"
            )

        result: dict[str, list[Detection]] = {}
        for i, cam_id in enumerate(cam_ids):
            cam_frame = pair.frames[cam_id]
            result[cam_id] = decode_yolo11_detect(
                per_image[i],
                camera_id=cam_id,
                capture_ts=cam_frame.capture_ts,
                letterbox_meta=lb_results[i],
                class_names=self._class_names,
                confidence_threshold=self._confidence_threshold,
                iou_threshold=self._iou_threshold,
                keep_classes=self._keep_classes,
            )
        return result
