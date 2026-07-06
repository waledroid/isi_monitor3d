"""``DiagnosticsPublisher`` — periodic node heartbeat for distributed deployments.

Runs a daemon thread that builds a ``DiagnosticsMessage`` every
``interval_sec`` seconds and publishes it through the Backbone ``Publisher``.
The ``build_message()`` method is separated from the threading so tests can
call it directly without spinning up a background thread.

Wired by the ``Orchestrator`` in ``_build()`` if ``metadata.diagnostics.enabled``
(default True).  One instance per Backbone process.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from backbone.comms.publisher import Publisher
from backbone.comms.schemas import (
    CalibrationFactCheck,
    DiagnosticsMessage,
    LatencyStats,
)
from backbone.shared.timestamps import now

logger = logging.getLogger(__name__)


class DiagnosticsPublisher:
    """Periodic heartbeat publisher for distributed node monitoring (Phase 1).

    Args:
        orchestrator: The running ``Orchestrator`` instance; read for mode,
            source_status, frame_count, rig, latency_meter, zone_count,
            and subscription_count.
        publisher:    The ``Publisher`` fan-out that routes to all sinks.
        node_id:      Unique identity string for this Backbone node (e.g.
            ``"zone_a"``).  Appears in every DiagnosticsMessage and in the
            retained ConfigMessage.
        interval_sec: Seconds between heartbeat publishes (default 5.0).
        rms_gate_px:  Maximum acceptable reprojection RMS (pixels) for
            ``CalibrationFactCheck.rms_ok`` to be True (default 2.0).
    """

    def __init__(
        self,
        orchestrator: Any,
        publisher: Publisher,
        *,
        node_id: str,
        interval_sec: float = 5.0,
        rms_gate_px: float = 2.0,
    ) -> None:
        self._orchestrator = orchestrator
        self._publisher = publisher
        self._node_id = node_id
        self._interval_sec = interval_sec
        self._rms_gate_px = rms_gate_px

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        # State for fps computation (global pipeline + per-camera ingest).
        self._last_frame_count: int | None = None
        self._last_cam_counts: dict[str, int] | None = None
        self._last_tick_ts: float | None = None

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def build_message(self) -> DiagnosticsMessage:
        """Build a ``DiagnosticsMessage`` from current orchestrator state.

        Separated from the thread so tests can call this without starting
        the background loop.

        FPS semantics:
          * Returns 0.0 on the **first** call (no prior tick to diff against).
          * Returns ``frames / dt`` on subsequent calls.
        """
        o = self._orchestrator

        # --- fps computation (pipeline pairs + per-camera ingest) ---
        current_count = o.frame_count
        cam_counts = getattr(o, "frames_by_camera", None) or {}
        current_ts = now()
        if self._last_frame_count is None or self._last_tick_ts is None:
            fps = 0.0
            fps_by_camera: dict[str, float] = dict.fromkeys(cam_counts, 0.0)
        else:
            dt = current_ts - self._last_tick_ts
            delta = current_count - self._last_frame_count
            fps = float(delta / dt) if dt > 0 else 0.0
            prev = self._last_cam_counts or {}
            fps_by_camera = {
                cam: (float(n - prev.get(cam, 0)) / dt if dt > 0 else 0.0)
                for cam, n in cam_counts.items()
            }
        self._last_frame_count = current_count
        self._last_cam_counts = dict(cam_counts)
        self._last_tick_ts = current_ts

        # --- calibration fact-check ---
        mode_str: str = o.mode
        rig = o.rig
        cam_views = rig.items()   # Mapping[str, _CameraView]
        if cam_views:
            rms_ok = all(
                v.reprojection_rms_px <= self._rms_gate_px
                for v in cam_views.values()
            )
        else:
            rms_ok = False
        cal_mode = 1 if mode_str == "single_cam_homography" else 2
        calibration = CalibrationFactCheck(loaded=True, rms_ok=rms_ok, mode=cal_mode)

        # --- latency stats ---
        latency_ms = LatencyStats(**o.latency_meter.percentiles())

        return DiagnosticsMessage(
            ts=current_ts,
            node_id=self._node_id,
            mode=mode_str,
            sources=dict(o.source_status),
            frame_count=current_count,
            fps=fps,
            fps_by_camera={c: round(v, 2) for c, v in fps_by_camera.items()},
            latency_ms=latency_ms,
            zones=o.zone_count,
            subscriptions=o.subscription_count,
            calibration=calibration,
        )

    def start(self) -> None:
        """Start the background heartbeat thread (daemon, safe to call once)."""
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="diagnostics-heartbeat",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "DiagnosticsPublisher: started (node_id=%r, interval=%.1fs)",
            self._node_id,
            self._interval_sec,
        )

    def stop(self) -> None:
        """Signal the thread to stop and wait briefly for it to exit (a daemon
        thread — a short join keeps STOP fast; process exit reaps it anyway)."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _run(self) -> None:
        """Main loop: publish, then sleep interval_sec (or until stop signal)."""
        while not self._stop.is_set():
            try:
                msg = self.build_message()
                self._publisher.publish_diagnostics(msg)
            except Exception:
                logger.warning(
                    "DiagnosticsPublisher: publish failed", exc_info=True
                )
            self._stop.wait(self._interval_sec)
