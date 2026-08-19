"""Start / stop the whole system (perception producer + metric engine) from
Python — shared by the START/STOP routes and the :class:`SystemCycler`.

Both helpers are SYNC (they block for a few seconds) and expect the FastAPI
``app.state`` (``supervisor``, ``isistream``, ``bus``, ``stop_in_progress``,
``last_stop_done``). The routes call them via ``asyncio.to_thread``.
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)

# The orchestrator logs this once its build completes; before that, a freshly
# spawned process that is still importing would report "running" falsely.
READY_MARKER = "backbone built"
BOOT_TIMEOUT_S = 10.0
BOOT_POLL_S = 0.2


def start_system(app_state) -> dict:
    """Spawn the engine, wait for its ready marker (or crash), then start the
    producer in points mode. Returns the same dict shape ``POST /api/control/start``
    answers with (``spawned``, ``state``, ``pid``, ``last_exit_code``, ``log_tail``)."""
    supervisor = app_state.supervisor
    spawned = supervisor.start()
    if spawned:
        for _ in range(int(BOOT_TIMEOUT_S / BOOT_POLL_S)):
            time.sleep(BOOT_POLL_S)
            if supervisor.state != "running":
                break   # exited → boot crash
            if any(READY_MARKER in line.lower() for line in supervisor.log_lines(30)):
                break   # built OK and now running
    state = supervisor.state
    # Direction 1: with ingestion.mode: points the Backbone is a pure metric
    # engine — start the perception producer to feed it (builds CUDA sessions).
    if state == "running":
        perception = getattr(app_state, "isistream", None)
        if perception is not None and perception.points_mode():
            started = perception.start()
            logger.info("control: perception producer %s",
                        "running" if started else "FAILED to start")
    return {
        "spawned": spawned,
        "state": state,
        "pid": supervisor.pid,
        "last_exit_code": supervisor.last_exit_code,
        "log_tail": supervisor.log_lines(12) if state != "running" else [],
    }


def stop_system(app_state) -> None:
    """Tear the system down: producer first (it feeds the engine), then the
    engine; clears the live bus caches and maintains the START debounce anchor.
    Blocks until both processes are gone."""
    supervisor = app_state.supervisor
    app_state.stop_in_progress = True
    bus = getattr(app_state, "bus", None)
    if bus is not None:
        try:
            bus.clear_live_state()
        except Exception:
            logger.debug("control: bus clear failed", exc_info=True)
    t0 = time.monotonic()
    try:
        perception = getattr(app_state, "isistream", None)
        if perception is not None:
            perception.stop()
        supervisor.stop()
    finally:
        app_state.stop_in_progress = False
        app_state.last_stop_done = time.monotonic()
        logger.info("control: STOP done in %.2fs (state=%s)",
                    time.monotonic() - t0, supervisor.state)


def system_running(app_state) -> bool:
    return getattr(app_state.supervisor, "state", "") == "running"
