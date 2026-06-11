"""Person-pose overlay for the dashboard preview — runs a YOLO-pose ONNX and
draws the skeleton + the foot node (ankle midpoint) alongside the detection
boxes drawn by ``detection_overlay``.

Independent of the trainer package (onnxruntime + OpenCV); the ORT session is
built via the shared ``backbone.shared.ort_session`` helper so it gets the same
memory-safe arena options as every other session. Same raw-head decode the
trainer's PoseOnnxInferencer uses: head ``(1, 4 + nc + K*3, A)`` → person boxes +
``[K, 3]`` keypoints.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np
from backbone.shared.ort_session import build_onnx_session

logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort
except ImportError as exc:  # pragma: no cover
    raise ImportError("onnxruntime required for pose overlay") from exc

LEFT_ANKLE, RIGHT_ANKLE = 15, 16
_COCO_SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
    (1, 3), (2, 4), (3, 5), (4, 6),
]


@dataclass(slots=True)
class Pose:
    box_xyxy: np.ndarray   # (4,) x1,y1,x2,y2 in original-image px
    score: float
    keypoints: np.ndarray  # (K,3) x,y,conf in original-image px
    foot_uv: tuple[float, float]


def _foot(keypoints: np.ndarray, box: np.ndarray, kpt_conf: float) -> tuple[float, float]:
    vis = [keypoints[i] for i in (LEFT_ANKLE, RIGHT_ANKLE) if keypoints[i, 2] >= kpt_conf]
    if vis:
        return float(np.mean([k[0] for k in vis])), float(np.mean([k[1] for k in vis]))
    return float((box[0] + box[2]) / 2.0), float(box[3])


class PoseEngine:
    """Lazy YOLO-pose ONNX runner (CUDA → CPU)."""

    def __init__(self, model_path: str, conf: float = 0.3, kpt_conf: float = 0.3,
                 device: str | None = None) -> None:
        self.conf, self.kpt_conf = float(conf), float(kpt_conf)
        providers = (["CPUExecutionProvider"] if device == "cpu"
                     or "CUDAExecutionProvider" not in ort.get_available_providers()
                     else ["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.session = build_onnx_session(model_path, providers=providers)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        _, _, self.h, self.w = inp.shape
        out_c = self.session.get_outputs()[0].shape[1]
        self.k = (out_c - 4 - 1) // 3 if isinstance(out_c, int) else 17   # single person class
        self._lb = (1.0, 0, 0)
        logger.info("pose overlay: loaded %s (in=%sx%s, K=%s, %s)", model_path,
                    self.w, self.h, self.k, self.session.get_providers()[0])

    def _letterbox(self, frame: np.ndarray):
        oh, ow = frame.shape[:2]
        r = min(self.h / oh, self.w / ow)
        nw, nh = round(ow * r), round(oh * r)
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        pw, ph = self.w - nw, self.h - nh
        left, top = pw // 2, ph // 2
        padded = cv2.copyMakeBorder(resized, top, ph - top, left, pw - left,
                                    cv2.BORDER_CONSTANT, value=(114, 114, 114))
        self._lb = (r, left, top)
        return padded

    def predict(self, frame_bgr: np.ndarray) -> list[Pose]:
        rgb = cv2.cvtColor(self._letterbox(frame_bgr), cv2.COLOR_BGR2RGB)
        tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0)
        out = self.session.run(None, {self.input_name: tensor})[0]
        return self._decode(out, frame_bgr.shape[1], frame_bgr.shape[0])

    def _decode(self, raw: np.ndarray, ow: int, oh: int) -> list[Pose]:
        preds = raw[0].T                      # (A, 4 + 1 + K*3)
        scores = preds[:, 4].astype(np.float32)
        keep = scores >= self.conf
        if not np.any(keep):
            return []
        preds, scores = preds[keep], scores[keep]
        cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        boxes = np.column_stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]).astype(np.float32)
        kpts = preds[:, 5:5 + self.k * 3].reshape(-1, self.k, 3).astype(np.float32)
        xywh = np.column_stack([boxes[:, 0], boxes[:, 1],
                                boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]]).tolist()
        kept = cv2.dnn.NMSBoxes(xywh, scores.tolist(), self.conf, 0.45)
        if len(kept) == 0:
            return []
        kept = np.array(kept).flatten()
        boxes, scores, kpts = boxes[kept], scores[kept], kpts[kept]
        r, px, py = self._lb
        boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]] - px) / r, 0, ow)
        boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]] - py) / r, 0, oh)
        kpts[:, :, 0] = (kpts[:, :, 0] - px) / r
        kpts[:, :, 1] = (kpts[:, :, 1] - py) / r
        return [Pose(b, float(s), kp, _foot(kp, b, self.kpt_conf))
                for b, s, kp in zip(boxes, scores, kpts, strict=True)]

    def draw(self, image: np.ndarray, poses: list[Pose]) -> None:
        """Draw skeleton + keypoints + foot node in place (no person bounding box)."""
        for p in poses:
            for a, b in _COCO_SKELETON:
                if p.keypoints[a, 2] >= self.kpt_conf and p.keypoints[b, 2] >= self.kpt_conf:
                    cv2.line(image, tuple(p.keypoints[a, :2].astype(int)),
                             tuple(p.keypoints[b, :2].astype(int)), (255, 180, 0), 2)
            for j in range(self.k):
                if p.keypoints[j, 2] >= self.kpt_conf:
                    cv2.circle(image, tuple(p.keypoints[j, :2].astype(int)), 3, (0, 0, 255), -1)
            cv2.circle(image, (int(p.foot_uv[0]), int(p.foot_uv[1])), 6, (0, 255, 255), -1)
