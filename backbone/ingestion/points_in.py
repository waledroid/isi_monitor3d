"""``DetectionIngest`` — the Backbone's Direction-1 input: detections, not frames.

In ``ingestion.mode: points`` the Backbone is a pure METRIC engine: an
external perception producer (the dashboard's perception loop, later a
DeepStream probe) publishes per-camera ``DetectionSetMessage``s on a dedicated
loopback UDP port, and this listener turns each one into a ``DetectionSet``
payload and submits it to the *unchanged* ``FrameSynchronizer`` — which is
duck-typed on ``.camera_id`` / ``.capture_ts``, so aligned pairing, skew
eviction, Mode-1 solo emit, and Mode-2 runtime degradation all work exactly
as they do for frames.

Contract properties this module enforces/observes (see ``schemas.py``):
  * explicit-empty heartbeat — an empty ``dets`` tuple still advances pairing
    and liveness; silence is what signals a dead producer/camera;
  * ``seq`` gaps are counted per camera (UDP loss made visible, never silent);
  * ``config_fingerprint`` mismatches are WARNED once per change (degraded
    visibility beats blindness — messages are never dropped for it);
  * transport fragments (``FragmentMessage``) are reassembled with the same
    ``FragmentBuffer`` the dashboard's bus subscriber uses.

Deliberately a concrete module, not a plugin seam: there is one sensible way
to ingest UDP/JSON detection sets (same argument as ``FrameSynchronizer``).
"""

from __future__ import annotations

import json
import logging
import socket
import threading
from dataclasses import dataclass, field

import cv2
import numpy as np

from backbone.comms.schemas import (
    DetectionSetMessage,
    FragmentBuffer,
    FragmentMessage,
    parse_envelope,
)
from backbone.core.types import Detection
from backbone.shared.timestamps import now

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DetectionSet:
    """Per-camera detections payload — the points-mode stand-in for ``Frame``.

    Quacks like a ``Frame`` for the synchronizer (``camera_id``,
    ``capture_ts``, ``frame_idx``); carries the producer's declared pixel
    space (``frame_wh``) so the orchestrator's scale boundary can map to
    calibration-frame coordinates without ever seeing an image.
    """

    camera_id: str
    capture_ts: float
    frame_idx: int
    frame_wh: tuple[int, int]
    detections: list[Detection] = field(default_factory=list)


def _rasterize_mask(poly, bbox_xyxy) -> tuple[np.ndarray | None, tuple[int, int] | None]:
    """Wire polygon → crop-local bool mask + crop origin, for occupancy.

    The metric engine's ``PalletOccupancy`` compares mask/bbox areas; a
    rasterized outline keeps that logic byte-identical to frames mode. The
    crop is the detection's bbox (the polygon never exceeds it by more than
    rounding), matching zone-scope's crop-relative mask convention.
    """
    if not poly:
        return None, None
    x0, y0 = int(np.floor(bbox_xyxy[0])), int(np.floor(bbox_xyxy[1]))
    x1, y1 = int(np.ceil(bbox_xyxy[2])), int(np.ceil(bbox_xyxy[3]))
    w, h = max(1, x1 - x0), max(1, y1 - y0)
    canvas = np.zeros((h, w), dtype=np.uint8)
    pts = np.asarray([[px - x0, py - y0] for px, py in poly], dtype=np.int32)
    cv2.fillPoly(canvas, [pts.reshape(-1, 1, 2)], 1)
    return canvas.astype(bool), (x0, y0)


def detection_set_from_message(msg: DetectionSetMessage) -> DetectionSet:
    """Wire message → in-process payload (``WireDetection`` → core ``Detection``)."""
    dets: list[Detection] = []
    for wd in msg.dets:
        mask, offset = _rasterize_mask(wd.mask_poly, wd.bbox_xyxy)
        dets.append(Detection(
            camera_id=msg.camera_id,
            capture_ts=msg.ts,
            cls=wd.cls,
            confidence=float(wd.confidence),
            bbox_xyxy=tuple(float(v) for v in wd.bbox_xyxy),
            foot_uv=(float(wd.foot_uv[0]), float(wd.foot_uv[1])),
            keypoints_uv=(np.asarray(wd.keypoints_uv, dtype=np.float64)
                          if wd.keypoints_uv is not None else None),
            mask=mask,
            mask_offset_xy=offset,
        ))
    return DetectionSet(
        camera_id=msg.camera_id,
        capture_ts=float(msg.ts),
        frame_idx=int(msg.seq),
        frame_wh=(int(msg.frame_wh[0]), int(msg.frame_wh[1])),
        detections=dets,
    )


class DetectionIngest:
    """UDP listener thread feeding ``DetectionSet``s into the synchronizer.

    ``on_set(detection_set)`` is the single delivery callback (the
    orchestrator submits to its synchronizer/bus there); parsing, fragment
    reassembly, seq accounting, and fingerprint checks all live here.
    """

    def __init__(
        self,
        camera_ids: list[str],
        *,
        host: str = "127.0.0.1",
        port: int = 9010,
        on_set,
        expected_fingerprint: str | None = None,
    ) -> None:
        self._camera_ids = set(camera_ids)
        self._host = host
        self._port = int(port)
        self._on_set = on_set
        self._expected_fingerprint = expected_fingerprint
        self._warned_fingerprint: str | None = None

        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._frag_buf = FragmentBuffer()

        self._lock = threading.Lock()
        self.sets_by_camera: dict[str, int] = {cid: 0 for cid in camera_ids}
        self.seq_gaps_by_camera: dict[str, int] = {cid: 0 for cid in camera_ids}
        self.last_seen_by_camera: dict[str, float] = {}
        self.dropped_malformed = 0
        self._last_seq: dict[str, int] = {}

    @property
    def address(self) -> tuple[str, int]:
        return (self._host, self._port)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self._host, self._port))
        self._port = self._sock.getsockname()[1]   # resolve port 0 → real port
        self._sock.settimeout(0.5)
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="points-ingest")
        self._thread.start()
        logger.info("points ingest: listening on %s:%d", self._host, self._port)

    def stop(self) -> None:
        self._stop_event.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # ---- internals ----

    def _run(self) -> None:
        assert self._sock is not None
        while not self._stop_event.is_set():
            try:
                payload, _addr = self._sock.recvfrom(65535)
            except TimeoutError:
                continue
            except OSError:
                if self._stop_event.is_set():
                    return
                continue
            self._handle_payload(payload)

    def _handle_payload(self, payload: bytes) -> None:
        try:
            msg = parse_envelope(json.loads(payload.decode("utf-8")))
        except Exception:   # malformed JSON, bad schema version, unknown type
            with self._lock:
                self.dropped_malformed += 1
            return
        if isinstance(msg, FragmentMessage):
            text = self._frag_buf.add(msg, now())
            if text is None:
                return
            try:
                msg = parse_envelope(json.loads(text))
            except Exception:
                with self._lock:
                    self.dropped_malformed += 1
                return
        if not isinstance(msg, DetectionSetMessage):
            with self._lock:
                self.dropped_malformed += 1
            return
        if msg.camera_id not in self._camera_ids:
            logger.debug("points ingest: unknown camera %r ignored", msg.camera_id)
            return

        with self._lock:
            self.sets_by_camera[msg.camera_id] += 1
            self.last_seen_by_camera[msg.camera_id] = now()
            prev = self._last_seq.get(msg.camera_id)
            if prev is not None and msg.seq > prev + 1:
                self.seq_gaps_by_camera[msg.camera_id] += msg.seq - prev - 1
            self._last_seq[msg.camera_id] = msg.seq

        if (self._expected_fingerprint is not None
                and msg.config_fingerprint is not None
                and msg.config_fingerprint != self._expected_fingerprint
                and msg.config_fingerprint != self._warned_fingerprint):
            # Warn once per distinct mismatching fingerprint; never drop —
            # degraded visibility beats blindness (fail-honestly).
            self._warned_fingerprint = msg.config_fingerprint
            logger.warning(
                "points ingest: producer config fingerprint %r != engine %r — "
                "model/zones/calibration may have drifted between processes",
                msg.config_fingerprint, self._expected_fingerprint)

        try:
            self._on_set(detection_set_from_message(msg))
        except Exception:
            logger.warning("points ingest: delivery failed", exc_info=True)
