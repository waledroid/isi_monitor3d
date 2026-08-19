"""START / STOP control for the Backbone subprocess."""

from __future__ import annotations

import asyncio
import logging
import threading
import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ..system_control import start_system, stop_system

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/control")

# START/STOP mechanics (boot-marker polling, producer-then-engine teardown) live
# in monitor_web/system_control.py — shared with the SystemCycler's scheduled
# restarts so the button and the timer do exactly the same thing.
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
    result = await asyncio.to_thread(start_system, request.app.state)
    spawned, state = result["spawned"], result["state"]
    cycler = getattr(request.app.state, "cycler", None)
    if cycler is not None:
        cycler.note_started()          # an operator START resets the restart clock
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

    # Teardown off the event loop (producer first, then engine; debounce anchor
    # set at the end so queued START clicks replaying after it get ignored).
    threading.Thread(target=stop_system, args=(app_state,), daemon=True,
                     name="stop-teardown").start()
    return JSONResponse({"action": "stop", "stopped": True, "state": "stopping"})


@router.get("/state")
async def state(request: Request) -> JSONResponse:
    supervisor = request.app.state.supervisor
    st = ("stopping" if getattr(request.app.state, "stop_in_progress", False)
          else supervisor.state)
    return JSONResponse(
        {"state": st, "pid": supervisor.pid, "last_exit_code": supervisor.last_exit_code}
    )
