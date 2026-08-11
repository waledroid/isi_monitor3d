"""``YoloOpenvinoPoseDetector`` — YOLO11-pose (person) via OpenVINO (CPU/iGPU).

Sibling of ``YoloOpenvinoDetector``, pose variant. The IR (``model.xml`` +
``model.bin``, converted from the pose ONNX with ``ovc``) carries the same raw
pose head:

    head  (N, 4 + nc + K*3, A)   ← bbox + class score(s) + K keypoints (x, y, conf)

``decode_yolo11_pose`` (pure numpy, backend-agnostic) turns it into
``Detection`` objects with ``cls="person"``, ``keypoints_uv`` (K, 3), and
``foot_uv`` at the ankle midpoint — identical output contract to the retired
ONNX pose plugin, so isistream's pose stage and every wire consumer are
unchanged.

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

from .postprocess import decode_yolo11_pose
from .preprocess import batch_letterbox

logger = logging.getLogger(__name__)


@detector_registry.register("yolo_openvino_pose")
class YoloOpenvinoPoseDetector(Detector):
    """Run a YOLO11-pose OpenVINO IR (person) on synchronized camera frames."""

    def __init__(
        self,
        model_xml: str | Path,
        class_names: list[str] | None = None,
        *,
        input_size: tuple[int, int] = (640, 640),
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        kpt_conf: float = 0.3,
        device: str = "CPU",
    ) -> None:
        xml_file = Path(model_xml)
        if not xml_file.exists():
            raise FileNotFoundError(
                f"YoloOpenvinoPoseDetector: OpenVINO IR not found at {xml_file}. "
                f"Convert the pose ONNX once with `ovc <pose>.onnx --output_model model.xml`."
            )
        try:
            import openvino as ov  # lazy — keeps backbone.detection importable without it
        except ImportError as exc:
            raise RuntimeError(
                "YoloOpenvinoPoseDetector requires the 'openvino' package. "
                "Add it to the env: `conda env update -f environment.yml`."
            ) from exc

        self._model_xml = xml_file
        # Pose models are single-class (person).
        self._class_names = list(class_names) if class_names else ["person"]
        self._input_size = input_size                  # (w, h)
        self._confidence_threshold = float(confidence_threshold)
        self._iou_threshold = float(iou_threshold)
        self._kpt_conf = float(kpt_conf)

        core = ov.Core()
        model = core.read_model(str(xml_file))
        if len(model.outputs) != 1:
            raise ValueError(
                f"YoloOpenvinoPoseDetector: IR has {len(model.outputs)} outputs; expected 1 "
                f"(pose head). Did you convert a seg/detect model?"
            )
        try:
            self._compiled = core.compile_model(model, device)
            self._device = device
        except Exception:
            logger.warning(
                "YoloOpenvinoPoseDetector: device %r unavailable, falling back to CPU", device
            )
            self._compiled = core.compile_model(model, "CPU")
            self._device = "CPU"
        self._output = self._compiled.output(0)

        logger.info(
            "YoloOpenvinoPoseDetector: loaded %s | device=%s | input=%dx%d",
            xml_file.name, self._device, self._input_size[0], self._input_size[1],
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

    def detect(self, pair: FramePair) -> dict[str, list[Detection]]:
        if not pair.frames:
            return {}
        # Sequential per-camera inference — same policy as the other OpenVINO
        # plugins (a single-camera CPU deployment feeds one frame anyway).
        result: dict[str, list[Detection]] = {}
        for cam_id, frame in pair.frames.items():
            batch_tensor, lb_results = batch_letterbox([frame.image], target=self._input_size)
            raw = self._compiled([batch_tensor])[self._output]  # (1, 4+nc+K*3, A)
            if raw.ndim != 3 or raw.shape[0] != 1:
                raise RuntimeError(
                    f"YoloOpenvinoPoseDetector: unexpected output shape {raw.shape} "
                    f"(expected (1, 4+nc+K*3, A))"
                )
            result[cam_id] = decode_yolo11_pose(
                raw[0],
                camera_id=cam_id,
                capture_ts=frame.capture_ts,
                letterbox_meta=lb_results[0],
                class_names=self._class_names,
                confidence_threshold=self._confidence_threshold,
                iou_threshold=self._iou_threshold,
                kpt_conf=self._kpt_conf,
            )
        return result
