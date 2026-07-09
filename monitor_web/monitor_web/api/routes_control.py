"""START / STOP control for the Backbone subprocess."""

from __future__ import annotations

import asyncio
import logging
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
    supervisor = request.app.state.supervisor
    # Off the event loop: supervisor.stop() blocks for the SIGTERM grace +
    # memory trim (seconds — longer under host-RAM pressure, exactly when the
    # operator most needs the UI alive). The loop keeps serving Settings and
    # status while the teardown runs in a worker thread. (An earlier freeze
    # attributed to this pattern was actually the Settings model-list walking
    # 34k dataset files ON the loop — fixed in routes_config/detection_overlay;
    # with that gone, off-loop stop is strictly better.)
    logger.info("control: STOP requested (state=%s, pid=%s)",
                supervisor.state, supervisor.pid)
    t0 = time.monotonic()
    stopped = await asyncio.to_thread(supervisor.stop)
    # The system is DOWN — empty every live cache so the UI clears at once:
    # tracks/zone_state/observations vanish from the map, cards and panels
    # instead of aging out over their staleness windows.
    bus = getattr(request.app.state, "bus", None)
    if bus is not None:
        try:
            bus.clear_live_state()
        except Exception:
            logger.debug("control: bus clear failed", exc_info=True)
    logger.info("control: STOP done in %.2fs (state=%s)",
                time.monotonic() - t0, supervisor.state)
    return JSONResponse(
        {"action": "stop", "stopped": stopped, "state": supervisor.state}
    )


@router.get("/state")
async def state(request: Request) -> JSONResponse:
    supervisor = request.app.state.supervisor
    return JSONResponse(
        {"state": supervisor.state, "pid": supervisor.pid, "last_exit_code": supervisor.last_exit_code}
    )
