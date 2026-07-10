"""START / STOP control for the Backbone subprocess."""

from __future__ import annotations

import asyncio
import logging
import threading
import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/control")

# START polls the freshly-spawned subprocess until it either logs the build-OK
# marker (truly up) or exits (boot crash). A fixed grace is wrong: heavy imports
# (onnxruntime + CUDA) delay a config/calibration crash to ~1-3s, so a short
# grace falsely reports "running". We wait up to _BOOT_TIMEOUT_S, returning as
# soon as either outcome is decided.
_BOOT_TIMEOUT_S = 10.0
_BOOT_POLL_S = 0.2
# Logged by backbone.runtime once the orchestrator is fully built (past calibration,
# detection, and sink construction) and about to run — a reliable "it's up" signal.
_READY_MARKER = "backbone built"
# STARTs arriving this soon after a completed STOP are treated as queued/
# replayed clicks, not intent — a human confirmation click lands later.
_START_DEBOUNCE_S = 3.0
# Substrings that mark the most operator-relevant log line in a boot-crash tail.
_REASON_HINTS = ("not found", "no metadata.sinks", "failed", "error", "cannot",
                 "filenotfound", "valueerror", "no such file")


def _crash_reason(log_tail: list[str]) -> str:
    """Pick the most informative line from a boot-crash tail (a raw Python
    traceback's last line is often unhelpful, e.g. just an exception class)."""
    for line in reversed(log_tail):
        if any(h in line.lower() for h in _REASON_HINTS):
            return line.strip()
    return log_tail[-1].strip() if log_tail else "unknown — check the LOGS panel"


@router.post("/start")
async def start(request: Request) -> JSONResponse:
    supervisor = request.app.state.supervisor
    # Loud on purpose: every START click is visible in the console the moment
    # it lands — a "the backbone came back by itself" report is almost always
    # a queued/duplicated click replaying after a UI stall, and this line is
    # how the operator sees it happen.
    logger.info("control: START requested (state=%s)", supervisor.state)
    # A deliberate STOP must STICK: refuse STARTs while the teardown runs and
    # for a short window after it — that is exactly when browser-queued
    # clicks replay. A human retry a few seconds later works normally.
    if getattr(request.app.state, "stop_in_progress", False):
        logger.warning("control: START ignored — STOP still in progress")
        return JSONResponse({"action": "start", "spawned": False,
                             "state": "stopping",
                             "reason": "stop in progress — try again in a moment",
                             "log_tail": []})
    since_stop = time.monotonic() - getattr(request.app.state, "last_stop_done", -1e9)
    if since_stop < _START_DEBOUNCE_S:
        logger.warning("control: START ignored (%.1fs after STOP — queued click?)",
                       since_stop)
        return JSONResponse({"action": "start", "spawned": False,
                             "state": supervisor.state,
                             "reason": "just stopped — press START again to confirm",
                             "log_tail": []})
    spawned = supervisor.start()
    # Poll until the orchestrator declares itself built (up) or the process exits
    # (crash), whichever comes first — don't trust an early "running" that's just
    # the subprocess still importing.
    if spawned:
        for _ in range(int(_BOOT_TIMEOUT_S / _BOOT_POLL_S)):
            await asyncio.sleep(_BOOT_POLL_S)
            if supervisor.state != "running":
                break   # exited → boot crash
            if any(_READY_MARKER in line.lower() for line in supervisor.log_lines(30)):
                break   # built OK and now running
    state = supervisor.state
    # Direction 1: with ingestion.mode: points the Backbone is a pure metric
    # engine — start the in-process perception producer to feed it. Off the
    # event loop: it builds CUDA sessions (seconds).
    if state == "running":
        perception = getattr(request.app.state, "isistream", None)
        if perception is not None and perception.points_mode():
            started = await asyncio.to_thread(perception.start)
            logger.info("control: perception producer %s",
                        "running" if started else "FAILED to start")
    log_tail = supervisor.log_lines(12) if state != "running" else []
    return JSONResponse(
        {
            "action": "start",
            "spawned": spawned,
            "state": state,
            "pid": supervisor.pid,
            "last_exit_code": supervisor.last_exit_code,
            # The single most useful line + a small tail, so the frontend can show
            # *why* it didn't start without waiting for the 2s logs poll.
            "reason": _crash_reason(log_tail) if state != "running" else "",
            "log_tail": log_tail,
        }
    )


@router.post("/stop")
async def stop(request: Request) -> JSONResponse:
    """Answer INSTANTLY; tear down in the background.

    The full teardown (producer SIGTERM grace + engine SIGTERM grace + memory
    trim) takes a few seconds — blocking the HTTP response for it made the
    STOP button feel dead and let the browser queue extra clicks. The route
    now clears the LIVE caches immediately (panels blank at once), kicks the
    teardown into a worker thread, and returns; ``/state`` reports
    ``stopping`` until the thread finishes.
    """
    app_state = request.app.state
    supervisor = app_state.supervisor
    if getattr(app_state, "stop_in_progress", False):
        return JSONResponse({"action": "stop", "state": "stopping"})
    logger.info("control: STOP requested (state=%s, pid=%s)",
                supervisor.state, supervisor.pid)
    app_state.stop_in_progress = True
    # Blank the UI immediately — the system is going down by operator intent.
    bus = getattr(app_state, "bus", None)
    if bus is not None:
        try:
            bus.clear_live_state()
        except Exception:
            logger.debug("control: bus clear failed", exc_info=True)

    def _teardown() -> None:
        t0 = time.monotonic()
        try:
            perception = getattr(app_state, "isistream", None)
            if perception is not None:
                perception.stop()      # producer first — it feeds the engine
            supervisor.stop()
        finally:
            app_state.stop_in_progress = False
            # Debounce anchor: START clicks that were queued while the UI was
            # busy replay AFTER this moment and get ignored (see start()).
            app_state.last_stop_done = time.monotonic()
            logger.info("control: STOP done in %.2fs (state=%s)",
                        time.monotonic() - t0, supervisor.state)

    threading.Thread(target=_teardown, daemon=True, name="stop-teardown").start()
    return JSONResponse({"action": "stop", "stopped": True, "state": "stopping"})


@router.get("/state")
async def state(request: Request) -> JSONResponse:
    supervisor = request.app.state.supervisor
    st = ("stopping" if getattr(request.app.state, "stop_in_progress", False)
          else supervisor.state)
    return JSONResponse(
        {"state": st, "pid": supervisor.pid, "last_exit_code": supervisor.last_exit_code}
    )
