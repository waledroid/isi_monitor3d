"""Étagère (bin-rack) cell inference — the producer-side half.

For every configured zone whose camera has a fresh frame (and whose max_fps
interval elapsed): scale its cell rects from ``frame_wh`` to the actual frame,
crop rect + margin (the same margin used at training time), key each crop
``"{zone_id}:{r}:{c}"``, batch ALL due crops into ONE ``FramePair`` → one
detector call (the yolo_onnx plugin letterboxes to its input size), then per
cell take the top-confidence detection >= threshold: filled_box -> "filled",
empty_box -> "empty", none -> "unknown". Emits one raw ``EtagereStateMessage``
per zone; the Backbone stabilises and publishes.
"""

from __future__ import annotations

import logging

import cv2

from backbone.comms.schemas import EtagereCellState, EtagereStateMessage
from backbone.core.types import Detection, Frame, FramePair
from backbone.shared.etagere import EtagereConfig, EtagereZone

logger = logging.getLogger(__name__)

_SEP = ":"


def decide(dets: list[Detection], threshold: float) -> tuple[str, float]:
    """Top-confidence detection >= threshold decides the cell's state."""
    best: Detection | None = None
    for d in dets:
        if d.confidence >= threshold and (best is None or d.confidence > best.confidence):
            best = d
    if best is None:
        return "unknown", 0.0
    if best.cls == "filled_box":
        return "filled", float(best.confidence)
    if best.cls == "empty_box":
        return "empty", float(best.confidence)
    return "unknown", 0.0


class EtagereDetector:
    """Per-tick étagère cell inference for every configured zone.

    ``detector`` is any object exposing ``.detect(FramePair) ->
    dict[str, list[Detection]]`` (the ``yolo_onnx`` plugin in production, a
    fake in tests).
    """

    def __init__(self, cfg: EtagereConfig, detector, *, producer_id: str = "isistream",
                 fingerprint: str | None = None) -> None:
        assert cfg.model is not None
        self._cfg = cfg
        self._det = detector
        self._producer_id = producer_id
        self._fingerprint = fingerprint
        self._seq: dict[str, int] = {z.id: 0 for z in cfg.zones}
        self._last_run: dict[str, float] = {}
        self._warned: set[str] = set()   # zone ids already logged this run — no log spam

    def due_zones(self, frames: dict[str, Frame], now: float) -> list[EtagereZone]:
        """Zones whose camera has a fresh frame AND whose max_fps interval elapsed."""
        out = []
        for z in self._cfg.zones:
            if z.camera not in frames:
                continue
            fps = z.max_fps or self._cfg.model.max_fps
            last = self._last_run.get(z.id)
            if last is not None and (now - last) < 1.0 / fps:
                continue
            out.append(z)
        return out

    def _crop(self, frame: Frame, zone: EtagereZone, rect: tuple[float, float, float, float],
              angle_deg: float = 0.0):
        """Scale ``rect`` from the zone's declared ``frame_wh`` to the actual
        frame, pad by ``crop_margin`` (fraction of the scaled rect's side),
        and return the cropped image (``None`` if the result is degenerate).

        With ``angle_deg`` != 0 the cell is a rectangle rotated about its own
        centre (positive = clockwise on screen); the source is warped by the
        inverse rotation about that centre first, so the crop is the cell's
        content UPRIGHT — the same axis-aligned framing the model was trained on.
        """
        h, w = frame.image.shape[:2]
        sx = w / float(zone.frame_wh[0])
        sy = h / float(zone.frame_wh[1])
        x0, y0, x1, y1 = rect[0] * sx, rect[1] * sy, rect[2] * sx, rect[3] * sy
        m = self._cfg.model.crop_margin
        mx, my = (x1 - x0) * m, (y1 - y0) * m
        if abs(angle_deg) < 1e-6:
            cx0, cy0 = max(int(x0 - mx), 0), max(int(y0 - my), 0)
            cx1, cy1 = min(int(x1 + mx), w), min(int(y1 + my), h)
            if cx1 - cx0 < 4 or cy1 - cy0 < 4:
                return None
            return frame.image[cy0:cy1, cx0:cx1]
        # Rotated cell: warp the source so the cell is upright, then take the
        # padded rect. cv2's positive angle is counter-clockwise on a y-down
        # image, so a clockwise-tilted cell (angle_deg > 0) is uprighted by
        # rotating the image by +angle_deg about the cell centre.
        ccx, ccy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        cw, ch = round((x1 - x0) + 2 * mx), round((y1 - y0) + 2 * my)
        if cw < 4 or ch < 4:
            return None
        rot = cv2.getRotationMatrix2D((ccx, ccy), float(angle_deg), 1.0)
        # translate so the (upright) padded rect's top-left lands at (0, 0)
        rot[0, 2] += cw / 2.0 - ccx
        rot[1, 2] += ch / 2.0 - ccy
        return cv2.warpAffine(frame.image, rot, (cw, ch), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))

    def _warn_once(self, zone_id: str, what: str) -> None:
        if zone_id in self._warned:
            return
        self._warned.add(zone_id)
        logger.warning("isistream: étagère zone %r %s — skipping this zone", zone_id, what,
                       exc_info=True)

    def run(self, frames: dict[str, Frame], now: float) -> list[EtagereStateMessage]:
        due = self.due_zones(frames, now)
        if not due:
            return []
        crops: dict[str, Frame] = {}
        good: list[EtagereZone] = []   # zones whose crop-building didn't raise
        for z in due:
            self._last_run[z.id] = now   # rate-gate applies even if this zone fails below
            frame = frames[z.camera]
            try:
                for cell in z.cells:
                    img = self._crop(frame, z, cell.rect, cell.angle_deg)
                    if img is None:
                        continue
                    key = f"{z.id}{_SEP}{cell.r}{_SEP}{cell.c}"
                    crops[key] = Frame(camera_id=key, capture_ts=frame.capture_ts,
                                       frame_idx=frame.frame_idx, image=img)
            except Exception:
                self._warn_once(z.id, "failed while building crops")
                continue
            good.append(z)
        results: dict[str, list[Detection]] = {}
        if crops:
            first = frames[due[0].camera]
            pair = FramePair(
                capture_ts=first.capture_ts, frame_idx=first.frame_idx, frames=crops)
            results = self._det.detect(pair)
        thr = self._cfg.model.confidence_threshold
        out: list[EtagereStateMessage] = []
        for z in good:
            try:
                frame = frames[z.camera]
                cells = []
                for cell in z.cells:
                    key = f"{z.id}{_SEP}{cell.r}{_SEP}{cell.c}"
                    state, conf = decide(list(results.get(key, [])), thr)
                    cells.append(EtagereCellState(r=cell.r, c=cell.c, state=state,
                                                  confidence=conf))
                msg = EtagereStateMessage(
                    ts=frame.capture_ts, camera_id=z.camera, zone_id=z.id, name=z.name,
                    rows=z.rows, cols=z.cols, cells=tuple(cells), seq=self._seq[z.id],
                    producer_id=self._producer_id, config_fingerprint=self._fingerprint,
                )
            except Exception:
                self._warn_once(z.id, "failed while building its message")
                continue
            out.append(msg)
            self._seq[z.id] += 1
        return out


def build_etagere_detector(cfg: EtagereConfig, *, producer_id: str = "isistream",
                           fingerprint: str | None = None) -> EtagereDetector | None:
    """``None`` when the feature is off (no model / no zones)."""
    if not cfg.enabled:
        return None
    import backbone.detection  # noqa: F401  (registers yolo_onnx)
    from backbone.core.interfaces import detector_registry
    m = cfg.model
    kwargs: dict = dict(onnx_path=m.onnx_path, class_names=list(m.class_names),
                        confidence_threshold=m.confidence_threshold,
                        input_size=(m.imgsz, m.imgsz))
    if m.providers:
        kwargs["providers"] = [m.providers]
    det = detector_registry.create("yolo_onnx", **kwargs)
    logger.info("isistream: étagère detector ready (%d zone(s), imgsz %d)",
                len(cfg.zones), m.imgsz)
    return EtagereDetector(cfg, det, producer_id=producer_id, fingerprint=fingerprint)
