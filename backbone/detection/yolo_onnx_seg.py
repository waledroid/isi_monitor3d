"""``YoloOnnxSegDetector`` — YOLO11-seg instance segmentation via ONNX Runtime.

Sibling of :mod:`backbone.detection.yolo_onnx`. Same plugin contract, same
preprocess (letterbox), same provider auto-fallback (CUDA → CPU). The seg ONNX
emits **two** outputs:

    output 0:  head    (N, 4 + nc + nm, A)      ← bbox + class + nm mask coeffs
    output 1:  protos  (N, nm, mh, mw)          ← mask prototype maps

``decode_yolo11_seg`` combines them into ``Detection`` objects with an extra
``mask`` field (full-frame HxW bool array in source-frame coords).

Inference-only. Detect detectors leave ``Detection.mask = None``; this plugin
populates it. The Backbone's homography pipeline is unaffected — it still
consumes ``foot_uv`` (bbox bottom-centre).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from backbone.core.interfaces import Detector, detector_registry
from backbone.core.types import Detection, FramePair
from backbone.shared.ort_session import build_onnx_session

from .onnx_meta import read_embedded_class_names
from .postprocess import decode_yolo11_seg
from .preprocess import batch_letterbox, pad_batch

logger = logging.getLogger(__name__)

DEFAULT_PROVIDERS: tuple[str, ...] = ("CUDAExecutionProvider", "CPUExecutionProvider")


@detector_registry.register("yolo_onnx_seg")
class YoloOnnxSegDetector(Detector):
    """Run a YOLO11-seg ONNX model on synchronized camera frames."""

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
        mask_threshold: float = 0.5,
        decode_masks: bool = True,
    ) -> None:
        onnx_file = Path(onnx_path)
        if not onnx_file.exists():
            raise FileNotFoundError(
                f"YoloOnnxSegDetector: ONNX file not found at {onnx_file}."
            )
        # class_names / keep_classes are resolved AFTER the session loads, so the
        # model's embedded names (if any) can take over — see below.
        self._onnx_path = onnx_file
        self._class_names = list(class_names) if class_names else None
        self._keep_classes_requested = list(keep_classes) if keep_classes else None
        self._input_size = input_size                  # (w, h)
        self._confidence_threshold = float(confidence_threshold)
        self._iou_threshold = float(iou_threshold)
        self._keep_classes = None
        self._mask_threshold = float(mask_threshold)
        # False = boxes/foot points only (mask=None) — skips the per-detection
        # full-frame mask assembly, which is pure CPU cost for consumers that
        # never read masks (the Backbone pipeline sets this False).
        self._decode_masks = bool(decode_masks)
        self._providers = list(providers) if providers else list(DEFAULT_PROVIDERS)

        try:
            self._session = build_onnx_session(onnx_file, providers=self._providers)
        except Exception as exc:
            raise RuntimeError(
                f"YoloOnnxSegDetector: failed to create ORT session for {onnx_file} "
                f"with providers {self._providers}: {exc}"
            ) from exc

        inputs = self._session.get_inputs()
        if len(inputs) != 1:
            raise ValueError(
                f"YoloOnnxSegDetector: model has {len(inputs)} inputs; expected 1"
            )
        outputs = self._session.get_outputs()
        if len(outputs) != 2:
            raise ValueError(
                f"YoloOnnxSegDetector: model has {len(outputs)} outputs; expected 2 "
                f"(head + mask prototypes). Did you load a detect-only ONNX?"
            )
        self._input_name = inputs[0].name
        # Adopt the model's own input spatial size when it's FIXED (e.g. a YOLO
        # exported with dynamic=False imgsz=1024). The ONNX input shape is
        # [batch, 3, H, W]; H/W are ints when static, strings/-1 when dynamic.
        # Letterboxing to a different size triggers the run() error
        # "Got invalid dimensions for input: images ... Got 640 Expected 1024".
        self._input_shape = inputs[0].shape
        # Largest batch fed so far (sticky) — solo pairs are padded up to this
        # so the CUDA session never sees a shape change (each one costs ~2.5 s).
        self._max_batch_seen = 1
        self._warned_static_batch = False
        ishape = self._input_shape
        if (len(ishape) == 4 and isinstance(ishape[2], int) and isinstance(ishape[3], int)
                and ishape[2] > 0 and ishape[3] > 0):
            model_wh = (int(ishape[3]), int(ishape[2]))   # (w, h)
            if model_wh != self._input_size:
                logger.info(
                    "YoloOnnxSegDetector: model expects fixed %dx%d input — overriding "
                    "configured input_size %s", model_wh[0], model_wh[1], self._input_size,
                )
                self._input_size = model_wh

        # Resolve class names: prefer the names embedded in the ONNX (self-
        # configuring — immune to config drift), fall back to the configured
        # names. This is what stops the "head has N channels; expected M for
        # nc=..." error when the model's class count changes.
        embedded = read_embedded_class_names(self._session)
        if embedded:
            if self._class_names and self._class_names != embedded:
                logger.info(
                    "YoloOnnxSegDetector: using class names embedded in the model %s "
                    "(overriding configured %s)", embedded, self._class_names,
                )
            self._class_names = embedded
        if not self._class_names:
            raise ValueError(
                "YoloOnnxSegDetector: no class names — the model embeds none and none "
                "were configured."
            )
        # keep_classes is a best-effort filter: silently drop entries the model
        # doesn't have (stale config), and treat 'nothing valid' as no filter
        # (keep all) rather than erroring.
        if self._keep_classes_requested:
            valid = [c for c in self._keep_classes_requested if c in self._class_names]
            dropped = [c for c in self._keep_classes_requested if c not in self._class_names]
            if dropped:
                logger.warning(
                    "YoloOnnxSegDetector: keep_classes %s not in model classes %s — ignored",
                    dropped, self._class_names,
                )
            self._keep_classes = valid or None

        self._active_providers = self._session.get_providers()
        if ("CUDAExecutionProvider" in self._providers
                and "CUDAExecutionProvider" not in self._active_providers
                # native .engine sessions ARE the GPU fast path
                and "TensorrtEngineFile" not in self._active_providers):
            logger.warning(
                "YoloOnnxSegDetector: CUDA was requested but the session fell back to %s — "
                "inference will be SLOW. Check the onnxruntime-gpu build / CUDA libs.",
                self._active_providers,
            )
        logger.info(
            "YoloOnnxSegDetector: loaded %s | providers=%s | nc=%d | input=%dx%d",
            onnx_file.name, self._active_providers, len(self._class_names),
            self._input_size[0], self._input_size[1],
        )

    @property
    def active_providers(self) -> tuple[str, ...]:
        return tuple(self._active_providers)

    @property
    def class_names(self) -> tuple[str, ...]:
        return tuple(self._class_names)

    @property
    def supports_batch(self) -> bool:
        """True when the ONNX has a dynamic batch dim, so >1 frame can be fed in
        one ``detect()`` call. The input shape is ``[batch, 3, H, W]``; ``batch`` is
        an int when fixed, a string ('batch'/'N') or None/-1 when dynamic."""
        bdim = self._input_shape[0] if self._input_shape else None
        return (not isinstance(bdim, int)) or bdim <= 0

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
        if self.supports_batch or batch_tensor.shape[0] == 1:
            # Sticky batch: never let the input shape shrink back after an
            # aligned pair — pad solo pairs with zero images instead (see
            # ``pad_batch``).
            if self.supports_batch:
                self._max_batch_seen = max(self._max_batch_seen, batch_tensor.shape[0])
                batch_tensor = pad_batch(batch_tensor, self._max_batch_seen)
            outputs = self._session.run(None, {self._input_name: batch_tensor})
        else:
            # Static-batch export (dynamic=False → batch pinned to 1) fed a
            # multi-image batch (zone scope rides every camera's crops in ONE
            # call): run per-image and re-stitch, same as yolo_onnx_pose —
            # slower than a dynamic export, infinitely better than failing
            # every tick with ORT InvalidArgument (live 2026-08-06).
            if not self._warned_static_batch:
                self._warned_static_batch = True
                logger.warning(
                    "YoloOnnxSegDetector: static-batch ONNX under a batching "
                    "consumer — falling back to per-image inference. "
                    "Re-export with dynamic=True for one-call batching.")
            per = [self._session.run(None, {self._input_name: batch_tensor[i:i + 1]})
                   for i in range(batch_tensor.shape[0])]
            outputs = [np.concatenate([p[j] for p in per], axis=0)
                       for j in range(len(per[0]))]
        if len(outputs) != 2:
            raise RuntimeError(
                f"YoloOnnxSegDetector: expected 2 ORT outputs (head + protos), got {len(outputs)}"
            )
        # Identify head vs protos by tensor rank, not output index — Ultralytics
        # usually exports head (3D) first then protos (4D), but the order isn't
        # guaranteed across versions/export flags. The 3D tensor is always the
        # detection head (N, 4+nc+nm, A); the 4D one is always the mask protos
        # (N, nm, mh, mw). Robust to either ordering.
        out_a, out_b = outputs
        if out_a.ndim == 3 and out_b.ndim == 4:
            head_batch, protos_batch = out_a, out_b
        elif out_a.ndim == 4 and out_b.ndim == 3:
            head_batch, protos_batch = out_b, out_a
        else:
            raise RuntimeError(
                f"YoloOnnxSegDetector: cannot identify head/protos from output ranks "
                f"{out_a.ndim} and {out_b.ndim} (expected one 3D head, one 4D protos)"
            )
        # N is the (possibly padded) batch; only the first len(cam_ids) entries
        # are real frames.
        if head_batch.shape[0] != batch_tensor.shape[0]:
            raise RuntimeError(
                f"YoloOnnxSegDetector: unexpected head shape {head_batch.shape} "
                f"(expected (N={batch_tensor.shape[0]}, 4+nc+nm, A))"
            )
        if protos_batch.shape[0] != batch_tensor.shape[0]:
            raise RuntimeError(
                f"YoloOnnxSegDetector: unexpected protos shape {protos_batch.shape} "
                f"(expected (N={batch_tensor.shape[0]}, nm, mh, mw))"
            )

        # `_input_size` is (w, h); target_hw needed by the decode is (h, w).
        target_hw = (self._input_size[1], self._input_size[0])
        result: dict[str, list[Detection]] = {}
        for i, cam_id in enumerate(cam_ids):
            cam_frame = pair.frames[cam_id]
            result[cam_id] = decode_yolo11_seg(
                head_batch[i],
                protos_batch[i],
                camera_id=cam_id,
                capture_ts=cam_frame.capture_ts,
                letterbox_meta=lb_results[i],
                target_hw=target_hw,
                class_names=self._class_names,
                confidence_threshold=self._confidence_threshold,
                iou_threshold=self._iou_threshold,
                keep_classes=self._keep_classes,
                mask_threshold=self._mask_threshold,
                decode_masks=self._decode_masks,
            )
        return result
