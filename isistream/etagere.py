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

    def _crop(self, frame: Frame, zone: EtagereZone, rect: tuple[float, float, float, float]):
        """Scale ``rect`` from the zone's declared ``frame_wh`` to the actual
        frame, pad by ``crop_margin`` (fraction of the scaled rect's side),
        and return the cropped image (``None`` if the result is degenerate)."""
        h, w = frame.image.shape[:2]
        sx = w / float(zone.frame_wh[0])
        sy = h / float(zone.frame_wh[1])
        x0, y0, x1, y1 = rect[0] * sx, rect[1] * sy, rect[2] * sx, rect[3] * sy
        m = self._cfg.model.crop_margin
        mx, my = (x1 - x0) * m, (y1 - y0) * m
        cx0, cy0 = max(int(x0 - mx), 0), max(int(y0 - my), 0)
        cx1, cy1 = min(int(x1 + mx), w), min(int(y1 + my), h)
        if cx1 - cx0 < 4 or cy1 - cy0 < 4:
            return None
        return frame.image[cy0:cy1, cx0:cx1]

    def run(self, frames: dict[str, Frame], now: float) -> list[EtagereStateMessage]:
        due = self.due_zones(frames, now)
        if not due:
            return []
        crops: dict[str, Frame] = {}
        for z in due:
            frame = frames[z.camera]
            for cell in z.cells:
                img = self._crop(frame, z, cell.rect)
                if img is None:
                    continue
                key = f"{z.id}{_SEP}{cell.r}{_SEP}{cell.c}"
                crops[key] = Frame(camera_id=key, capture_ts=frame.capture_ts,
                                   frame_idx=frame.frame_idx, image=img)
        results: dict[str, list[Detection]] = {}
        if crops:
            first = frames[due[0].camera]
            pair = FramePair(
                capture_ts=first.capture_ts, frame_idx=first.frame_idx, frames=crops)
            results = self._det.detect(pair)
        thr = self._cfg.model.confidence_threshold
        out: list[EtagereStateMessage] = []
        for z in due:
            self._last_run[z.id] = now
            frame = frames[z.camera]
            cells = []
            for cell in z.cells:
                key = f"{z.id}{_SEP}{cell.r}{_SEP}{cell.c}"
                state, conf = decide(list(results.get(key, [])), thr)
                cells.append(EtagereCellState(r=cell.r, c=cell.c, state=state, confidence=conf))
            out.append(EtagereStateMessage(
                ts=frame.capture_ts, camera_id=z.camera, zone_id=z.id, name=z.name,
                rows=z.rows, cols=z.cols, cells=tuple(cells), seq=self._seq[z.id],
                producer_id=self._producer_id, config_fingerprint=self._fingerprint,
            ))
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
