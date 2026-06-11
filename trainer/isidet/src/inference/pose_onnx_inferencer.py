"""ONNX pose inferencer — decodes a YOLO-pose raw head into person boxes +
keypoints, and derives the floor **foot node** (ankle keypoints) that the
homography layer projects.

Why a dedicated module (not bolted onto ``OptimizedONNXInferencer``): that engine
returns ``supervision.Detections``, which has no slot for per-detection keypoints,
and is entangled with the detect/seg/RF-DETR paths. Pose needs a different output
(boxes + ``[K, 3]`` keypoints + a foot point), so it gets a small, self-contained,
easily-tested engine that mirrors the same letterbox + raw-head decode math.

The raw YOLO-pose ONNX head (exported with ``nms=False``) is ``(1, 4 + nc + K*3, A)``:
columns ``0:4`` = box ``cxcywh`` (decoded, letterboxed pixels), ``4:4+nc`` = class
probabilities, ``4+nc:`` = ``K`` keypoints as ``(x, y, conf)``. For COCO person:
``nc=1, K=17`` → 56 columns. Ankle keypoints 15 (left) / 16 (right) are the feet.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort
except ImportError as exc:  # pragma: no cover - environment guard
    raise ImportError(
        "onnxruntime not installed. pip install onnxruntime-gpu or onnxruntime"
    ) from exc

# COCO-17 keypoint indices for the ankles — these are the floor foot nodes.
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

# COCO-17 skeleton (pairs of keypoint indices) for drawing.
_COCO_SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
    (1, 3), (2, 4), (3, 5), (4, 6),
]


@dataclass(slots=True)
class PersonPose:
    """One detected person."""

    box_xyxy: np.ndarray        # (4,) float — x1, y1, x2, y2 in original-image pixels
    score: float                # person box confidence
    keypoints: np.ndarray       # (K, 3) float — x, y, conf in original-image pixels
    foot_uv: tuple[float, float]  # the single floor contact point (see foot_point)


def foot_point(keypoints: np.ndarray, box_xyxy: np.ndarray,
               kpt_conf: float = 0.3) -> tuple[float, float]:
    """Derive the floor foot node from the ankle keypoints.

    Midpoint of the two visible ankles; if only one is visible use it; if neither
    is visible (occluded behind a pallet, etc.) fall back to the bbox bottom-center
    — strictly more information than a box alone, never less.
    """
    la, ra = keypoints[LEFT_ANKLE], keypoints[RIGHT_ANKLE]
    visible = [k for k in (la, ra) if k[2] >= kpt_conf]
    if visible:
        xs = float(np.mean([k[0] for k in visible]))
        ys = float(np.mean([k[1] for k in visible]))
        return xs, ys
    # Fallback: bottom-center of the box.
    return float((box_xyxy[0] + box_xyxy[2]) / 2.0), float(box_xyxy[3])


class PoseOnnxInferencer:
    """Self-contained YOLO-pose ONNX engine (CUDA → CPU fallback)."""

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str | None = None,
        kpt_conf: float = 0.3,
    ) -> None:
        self.model_path = str(model_path)
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.kpt_conf = float(kpt_conf)

        providers = self._providers(device)
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(self.model_path, sess_options=opts, providers=providers)

        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        # Static export → (1, 3, H, W).
        _, _, self.model_h, self.model_w = inp.shape
        self.num_keypoints, self.nc = self._read_head_layout()
        self._last_letterbox = (1.0, 0, 0)
        logger.info(
            "PoseOnnxInferencer: %s | in=%sx%s | nc=%s | K=%s | provider=%s",
            self.model_path, self.model_w, self.model_h, self.nc, self.num_keypoints,
            self.session.get_providers()[0],
        )

    # ── setup helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _providers(device: str | None) -> list:
        available = ort.get_available_providers()
        if device == "cpu" or "CUDAExecutionProvider" not in available:
            return ["CPUExecutionProvider"]
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def _read_head_layout(self) -> tuple[int, int]:
        """Return (num_keypoints, nc). Prefer the ``kpt_shape`` embedded by
        Ultralytics in the ONNX metadata; otherwise infer from the output width
        assuming the common single-class person model."""
        out = self.session.get_outputs()[0]
        # Output shape (1, C, A); C = 4 + nc + K*3.
        c = None
        if len(out.shape) == 3 and isinstance(out.shape[1], int):
            c = out.shape[1]
        kpt_shape = None
        try:
            meta = self.session.get_modelmeta().custom_metadata_map or {}
            if "kpt_shape" in meta:
                # e.g. "[17, 3]"
                kpt_shape = [int(x) for x in meta["kpt_shape"].strip("[] ").split(",")]
        except Exception:  # pragma: no cover - metadata is best-effort
            kpt_shape = None
        if kpt_shape:
            k = kpt_shape[0]
            nc = (c - 4 - k * 3) if c else 1
            return k, max(nc, 1)
        # No metadata: assume single-class person, derive K from C.
        if c:
            return (c - 4 - 1) // 3, 1
        return 17, 1

    # ── inference ────────────────────────────────────────────────────────────

    def _letterbox(self, frame: np.ndarray, pad_color: int = 114):
        """Aspect-preserving resize + pad to (model_h, model_w). Returns the
        padded image and (ratio, pad_left, pad_top) to invert on outputs."""
        orig_h, orig_w = frame.shape[:2]
        r = min(self.model_h / orig_h, self.model_w / orig_w)
        new_w, new_h = round(orig_w * r), round(orig_h * r)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_w, pad_h = self.model_w - new_w, self.model_h - new_h
        left, top = pad_w // 2, pad_h // 2
        padded = cv2.copyMakeBorder(resized, top, pad_h - top, left, pad_w - left,
                                    cv2.BORDER_CONSTANT, value=(pad_color,) * 3)
        return padded, r, left, top

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        padded, ratio, pad_x, pad_y = self._letterbox(rgb)
        self._last_letterbox = (ratio, pad_x, pad_y)
        img = padded.transpose((2, 0, 1)).astype(np.float32) / 255.0
        return np.ascontiguousarray(img[np.newaxis, ...])

    def predict(self, frame_bgr: np.ndarray) -> list[PersonPose]:
        """Run pose inference on a BGR image → list of PersonPose."""
        tensor = self._preprocess(frame_bgr)
        outputs = self.session.run(None, {self.input_name: tensor})
        orig_h, orig_w = frame_bgr.shape[:2]
        return self._decode(outputs[0], orig_w, orig_h)

    def _decode(self, raw: np.ndarray, orig_w: int, orig_h: int) -> list[PersonPose]:
        # raw: (1, C, A) → preds (A, C); C = 4 + nc + K*3.
        preds = raw[0].T
        ncls, k = self.nc, self.num_keypoints
        scores = preds[:, 4:4 + ncls].max(axis=1).astype(np.float32)
        keep = scores >= self.conf_threshold
        if not np.any(keep):
            return []
        preds, scores = preds[keep], scores[keep]

        cxcywh = preds[:, :4]
        cx, cy, w, h = cxcywh[:, 0], cxcywh[:, 1], cxcywh[:, 2], cxcywh[:, 3]
        boxes = np.column_stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]).astype(np.float32)
        kpts = preds[:, 4 + ncls:4 + ncls + k * 3].reshape(-1, k, 3).astype(np.float32)

        # NMS (single object class: person).
        xywh = np.column_stack([boxes[:, 0], boxes[:, 1],
                                boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]]).tolist()
        kept = cv2.dnn.NMSBoxes(xywh, scores.tolist(), self.conf_threshold, self.iou_threshold)
        if len(kept) == 0:
            return []
        kept = np.array(kept).flatten()
        boxes, scores, kpts = boxes[kept], scores[kept], kpts[kept]

        # Invert the letterbox on boxes AND keypoint (x, y); keypoint conf is kept.
        ratio, pad_x, pad_y = self._last_letterbox
        boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]] - pad_x) / ratio, 0, orig_w)
        boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]] - pad_y) / ratio, 0, orig_h)
        kpts[:, :, 0] = (kpts[:, :, 0] - pad_x) / ratio
        kpts[:, :, 1] = (kpts[:, :, 1] - pad_y) / ratio

        poses: list[PersonPose] = []
        for b, s, kp in zip(boxes, scores, kpts, strict=True):
            poses.append(PersonPose(
                box_xyxy=b, score=float(s), keypoints=kp,
                foot_uv=foot_point(kp, b, self.kpt_conf),
            ))
        return poses

    # ── visualization (manual sanity check) ──────────────────────────────────

    def draw(self, frame_bgr: np.ndarray, poses: list[PersonPose]) -> np.ndarray:
        """Draw boxes, COCO skeleton, and the foot node — for run_test_pose.sh /
        manual verification that ankles land at the feet."""
        out = frame_bgr.copy()
        for p in poses:
            x1, y1, x2, y2 = p.box_xyxy.astype(int)
            cv2.rectangle(out, (x1, y1), (x2, y2), (80, 220, 80), 2)
            for (a, b) in _COCO_SKELETON:
                if p.keypoints[a, 2] >= self.kpt_conf and p.keypoints[b, 2] >= self.kpt_conf:
                    pa = tuple(p.keypoints[a, :2].astype(int))
                    pb = tuple(p.keypoints[b, :2].astype(int))
                    cv2.line(out, pa, pb, (255, 180, 0), 2)
            for j in range(self.num_keypoints):
                if p.keypoints[j, 2] >= self.kpt_conf:
                    cv2.circle(out, tuple(p.keypoints[j, :2].astype(int)), 3, (0, 0, 255), -1)
            fx, fy = int(p.foot_uv[0]), int(p.foot_uv[1])
            cv2.circle(out, (fx, fy), 6, (0, 255, 255), -1)   # foot node (yellow)
            cv2.putText(out, "foot", (fx + 6, fy), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 255), 1, cv2.LINE_AA)
        return out
