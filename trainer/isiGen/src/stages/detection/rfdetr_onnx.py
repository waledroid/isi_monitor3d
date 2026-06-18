"""RF-DETR prompt detector — boxes only, fed to SAM2 as box prompts.

Decode ported faithfully from ``backbone/detection/postprocess.decode_rfdetr_seg``
+ ``rfdetr_onnx_seg`` preprocess (we drop the mask branch — SAM2 makes the mask).
RF-DETR is DETR-style / NMS-free with a fixed square input and stretch-resize
preprocess (ImageNet-normalised, DINOv2 backbone). Logits are sigmoid/focal and
COCO-indexed: column 0 = background, columns ``1..nc`` = trained classes.

Default classes ``[palette, carton, polybag]`` (the trainer's order); override via
``class_names`` in cfg. The decode maps detector classes onto the project's classes
in :meth:`PromptDetector._map_class`.
"""

from __future__ import annotations

import cv2
import numpy as np

from ...core.manifest import MaskPrompt
from .base import PromptDetector

DEFAULT_CLASS_NAMES: tuple[str, ...] = ("palette", "carton", "polybag")
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


class RfdetrPromptDetector(PromptDetector):
    """Auto-prompt SAM2 with RF-DETR boxes (NMS-free, fixed square input)."""

    def __init__(self, onnx_path, *, class_names=None, num_select: int = 300, **kw):
        super().__init__(
            onnx_path,
            class_names=list(class_names) if class_names else list(DEFAULT_CLASS_NAMES),
            **kw,
        )
        self._num_select = int(num_select)
        self._input_name: str | None = None
        self._input_wh: tuple[int, int] | None = None

    def load(self) -> None:
        super().load()
        inp = self._session.get_inputs()[0]
        self._input_name = inp.name
        ishape = inp.shape  # [1, 3, H, W]
        if len(ishape) == 4 and isinstance(ishape[2], int) and isinstance(ishape[3], int):
            self._input_wh = (int(ishape[3]), int(ishape[2]))  # (w, h)
        else:  # dynamic — fall back to the model's documented 432 square
            self._input_wh = (432, 432)

    def _preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        w, h = self._input_wh
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        chw = resized.transpose(2, 0, 1).astype(np.float32) / 255.0
        chw = (chw - _IMAGENET_MEAN) / _IMAGENET_STD
        return np.ascontiguousarray(chw[np.newaxis, ...])

    def detect(self, image_bgr: np.ndarray,
               project_class_names: list[str]) -> list[MaskPrompt]:
        if self._session is None or self._input_wh is None:
            self.load()
        src_h, src_w = image_bgr.shape[:2]
        tensor = self._preprocess(image_bgr)
        outputs = self._session.run(None, {self._input_name: tensor})
        out_map = {name: outputs[i] for i, name in enumerate(self._output_names)}
        dets = out_map["dets"][0]      # (Q, 4) cxcywh-norm
        labels = out_map["labels"][0]  # (Q, head_nc) logits

        class_names = self._class_names
        head_nc = labels.shape[1]
        effective_nc = min(len(class_names), head_nc - 1)
        if effective_nc <= 0:
            return []
        logits = labels[:, 1:1 + effective_nc]

        probs = _sigmoid(logits)
        flat = probs.reshape(-1)
        k = min(self._num_select, flat.size)
        topk_idx = np.argpartition(-flat, k - 1)[:k]
        topk_idx = topk_idx[np.argsort(-flat[topk_idx])]
        scores = flat[topk_idx]
        query_idx = topk_idx // effective_nc
        class_idx = (topk_idx % effective_nc).astype(int)

        keep = scores > self.confidence_threshold
        if not np.any(keep):
            return []
        scores, query_idx, class_idx = scores[keep], query_idx[keep], class_idx[keep]

        chosen = dets[query_idx]
        cx, cy, w, h = chosen[:, 0], chosen[:, 1], chosen[:, 2], chosen[:, 3]
        xyxy = np.column_stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
        xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]] * src_w, 0, src_w)
        xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]] * src_h, 0, src_h)

        prompts: list[MaskPrompt] = []
        for i in range(xyxy.shape[0]):
            mapped = self._map_class(class_names[int(class_idx[i])], project_class_names)
            if mapped is None:
                continue
            x1, y1, x2, y2 = (float(v) for v in xyxy[i])
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            prompts.append(MaskPrompt(kind="box", class_name=mapped, xyxy=[x1, y1, x2, y2]))
        return prompts
