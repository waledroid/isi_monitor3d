"""YOLO prompt detector — boxes only, fed to SAM2 as box prompts.

Decode ported from ``backbone/detection/postprocess.decode_yolo11_detect`` +
``preprocess.letterbox``/``invert_letterbox_xyxy`` (boxes only). Handles:

  * **detect** ONNX (1 output) — raw anchor head ``(4+nc, A)``;
  * **seg** ONNX (2 outputs) — boxes from the detection head, the proto output
    ignored (SAM2 makes the mask);
  * **end-to-end / NMS-free** heads (``(num_det, 6[+nm])`` rows of
    ``[x1,y1,x2,y2,score,cls,...]``) — already decoded + NMS'd.

Class names come from the ONNX ``names`` metadata (ultralytics writes a
stringified ``{idx: name}`` dict); ``class_names`` in cfg overrides.
"""

from __future__ import annotations

import ast
import logging

import cv2
import numpy as np

from ...core.manifest import MaskPrompt
from .base import PromptDetector

logger = logging.getLogger(__name__)
_LETTERBOX_PAD = 114


class YoloPromptDetector(PromptDetector):
    """Auto-prompt SAM2 with YOLO boxes (detect or seg ONNX, letterboxed)."""

    def __init__(self, onnx_path, *, iou_threshold: float = 0.45, **kw):
        super().__init__(onnx_path, **kw)
        self.iou_threshold = float(iou_threshold)
        self._input_name: str | None = None
        self._imgsz: tuple[int, int] | None = None  # (h, w)

    def load(self) -> None:
        super().load()
        inp = self._session.get_inputs()[0]
        self._input_name = inp.name
        ishape = inp.shape  # [1, 3, H, W]
        if len(ishape) == 4 and isinstance(ishape[2], int) and isinstance(ishape[3], int):
            self._imgsz = (int(ishape[2]), int(ishape[3]))
        else:  # dynamic input → default square
            self._imgsz = (640, 640)
        if self._class_names is None:
            self._class_names = self._read_names()

    def _read_names(self) -> list[str] | None:
        """Pull class names from ultralytics' ONNX ``names`` metadata, if present."""
        try:
            meta = self._session.get_modelmeta().custom_metadata_map or {}
            raw = meta.get("names")
            if raw:
                d = ast.literal_eval(raw)  # "{0: 'palette', 1: 'carton'}"
                return [d[i] for i in sorted(d)]
        except Exception:  # best-effort metadata read
            logger.warning("yolo prompt detector: could not read 'names' metadata")
        return None

    def _letterbox(self, image_bgr: np.ndarray):
        src_h, src_w = image_bgr.shape[:2]
        tgt_h, tgt_w = self._imgsz
        scale = min(tgt_w / src_w, tgt_h / src_h)
        new_w, new_h = round(src_w * scale), round(src_h * scale)
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=interp)
        pad_x = (tgt_w - new_w) // 2
        pad_y = (tgt_h - new_h) // 2
        padded = cv2.copyMakeBorder(
            resized, pad_y, tgt_h - new_h - pad_y, pad_x, tgt_w - new_w - pad_x,
            cv2.BORDER_CONSTANT, value=(_LETTERBOX_PAD,) * 3)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32).transpose(2, 0, 1)[np.newaxis, ...] / 255.0
        return np.ascontiguousarray(tensor), scale, (pad_x, pad_y), (src_h, src_w)

    def detect(self, image_bgr: np.ndarray,
               project_class_names: list[str]) -> list[MaskPrompt]:
        if self._session is None or self._imgsz is None:
            self.load()
        if not self._class_names:
            logger.warning("yolo prompt detector: no class names; cannot map detections")
            return []
        tensor, scale, (pad_x, pad_y), (src_h, src_w) = self._letterbox(image_bgr)
        head = self._session.run(None, {self._input_name: tensor})[0][0]  # first output, batch 0
        boxes = self._decode(head)  # list of (x1,y1,x2,y2,score,cls_idx) in target px
        prompts: list[MaskPrompt] = []
        for x1, y1, x2, y2, _score, cls_idx in boxes:
            if cls_idx >= len(self._class_names):
                continue
            mapped = self._map_class(self._class_names[cls_idx], project_class_names)
            if mapped is None:
                continue
            # un-letterbox → source pixels, clip
            sx1 = float(np.clip((x1 - pad_x) / scale, 0, src_w - 1))
            sy1 = float(np.clip((y1 - pad_y) / scale, 0, src_h - 1))
            sx2 = float(np.clip((x2 - pad_x) / scale, 0, src_w - 1))
            sy2 = float(np.clip((y2 - pad_y) / scale, 0, src_h - 1))
            if sx2 - sx1 < 1 or sy2 - sy1 < 1:
                continue
            prompts.append(MaskPrompt(kind="box", class_name=mapped,
                                      xyxy=[sx1, sy1, sx2, sy2]))
        return prompts

    def _decode(self, head: np.ndarray) -> list[tuple]:
        """Return [(x1,y1,x2,y2,score,cls_idx), ...] in target-frame pixels."""
        nc = len(self._class_names)
        if head.ndim != 2:
            logger.warning("yolo prompt detector: unexpected head shape %s", head.shape)
            return []

        # Raw anchor head — one axis is 4+nc (the other is A, in the thousands).
        # End-to-end / NMS-free head — neither axis is 4+nc; rows are
        # [x1,y1,x2,y2,score,cls,*coeffs] (already decoded + NMS'd).
        if 4 + nc not in head.shape:
            out = []
            for row in head:
                score = float(row[4])
                if score < self.confidence_threshold:
                    continue
                out.append((float(row[0]), float(row[1]), float(row[2]), float(row[3]),
                            score, int(row[5])))
            return out

        # (4+nc, A) → (A, 4+nc)
        pred = head.transpose(1, 0) if head.shape[0] == 4 + nc else head
        bbox = pred[:, :4]
        scores_all = pred[:, 4:4 + nc]
        cls_idx = scores_all.argmax(axis=1)
        conf = scores_all.max(axis=1)
        keep = conf >= self.confidence_threshold
        if not keep.any():
            return []
        bbox, cls_idx, conf = bbox[keep], cls_idx[keep], conf[keep]
        xyxy = np.empty_like(bbox)
        xyxy[:, 0] = bbox[:, 0] - bbox[:, 2] / 2
        xyxy[:, 1] = bbox[:, 1] - bbox[:, 3] / 2
        xyxy[:, 2] = bbox[:, 0] + bbox[:, 2] / 2
        xyxy[:, 3] = bbox[:, 1] + bbox[:, 3] / 2
        xywh = np.column_stack([xyxy[:, 0], xyxy[:, 1], bbox[:, 2], bbox[:, 3]])
        idxs = cv2.dnn.NMSBoxes(xywh.tolist(), conf.astype(np.float32).tolist(),
                                self.confidence_threshold, self.iou_threshold)
        if len(idxs) == 0:
            return []
        idxs = np.asarray(idxs).reshape(-1)
        return [(float(xyxy[i, 0]), float(xyxy[i, 1]), float(xyxy[i, 2]),
                 float(xyxy[i, 3]), float(conf[i]), int(cls_idx[i])) for i in idxs]
