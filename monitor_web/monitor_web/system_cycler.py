"""Auto-start + scheduled full restart of the system (producer + engine).

Why: long-running CUDA/ORT processes on the WSL2 rig accrete host RSS and
fragment the heap over hours ("Killed" = host-RAM OOM, see CLAUDE.md). A
periodic STOP → START of BOTH child processes returns every byte to the OS
(the supervisor's stop already runs ``malloc_trim``) with a few seconds of
blind time — acceptable for a warehouse rack/pallet monitor, and far better
than a 3 a.m. OOM. The knobs are PERSISTED UI settings (Settings ▸ Isistream ▸
Performance), readable at boot:

  auto_start:        START the system when the dashboard launches (default off)
  restart_every_min: restart the running system every N minutes (0 = never)

Rules that keep the operator in charge:
  * the cycler restarts only a system that IS running — a deliberate STOP
    stays stopped (auto_start is a boot-time action, not a watchdog);
  * it never fires while a STOP is in progress or within the START debounce;
  * a restart cycle is one stop_system() + one start_system(), logged loudly.
"""

from __future__ import annotations

import logging
import threading
import time

from .system_control import start_system, stop_system, system_running

logger = logging.getLogger(__name__)

_TICK_S = 5.0
_BOOT_SETTLE_S = 6.0        # let uvicorn/bus/hub come up before auto-START
_MIN_RESTART_MIN = 5.0      # guard against a fat-fingered 0.1-minute loop


class SystemCycler:
    def __init__(self, app_state, *, auto_start: bool = False,
                 restart_every_min: float = 0.0) -> None:
        self._app_state = app_state
        self._auto_start = bool(auto_start)
        self._restart_every_s = self._norm(restart_every_min)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._last_cycle_t: float = time.monotonic()
        self.cycles = 0
        self.last_error: str | None = None

    @staticmethod
    def _norm(minutes) -> float:
        try:
            m = float(minutes or 0.0)
        except (TypeError, ValueError):
            return 0.0
        if m <= 0:
            return 0.0
        return max(m, _MIN_RESTART_MIN) * 60.0

    # ---- settings -----------------------------------------------------------
    def configure(self, *, auto_start=None, restart_every_min=None) -> None:
        """Hot-apply new knobs (called after a ui-settings save)."""
        with self._lock:
            if auto_start is not None:
                self._auto_start = bool(auto_start)
            if restart_every_min is not None:
                new = self._norm(restart_every_min)
                if new != self._restart_every_s:
                    self._restart_every_s = new
                    self._last_cycle_t = time.monotonic()   # count from now
        logger.info("cycler: auto_start=%s restart_every=%s",
                    self._auto_start,
                    f"{self._restart_every_s / 60:.0f} min" if self._restart_every_s else "off")

    def status(self) -> dict:
        with self._lock:
            due = (self._last_cycle_t + self._restart_every_s - time.monotonic()
                   if self._restart_every_s else None)
        return {
            "auto_start": self._auto_start,
            "restart_every_min": self._restart_every_s / 60.0 if self._restart_every_s else 0,
            "next_restart_in_s": max(0.0, due) if due is not None else None,
            "cycles": self.cycles,
            "last_error": self.last_error,
        }

    def note_started(self) -> None:
        """An operator START resets the restart clock (called by the route)."""
        with self._lock:
            self._last_cycle_t = time.monotonic()

    # ---- lifecycle ----------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="system-cycler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    # ---- loop -----------------------------------------------------------------
    def _run(self) -> None:
        if self._auto_start:
            self._stop.wait(_BOOT_SETTLE_S)
            if not self._stop.is_set() and not system_running(self._app_state):
                logger.info("cycler: auto-START on launch")
                self._safe(start_system, "auto-start")
                self.note_started()
        while not self._stop.wait(_TICK_S):
            with self._lock:
                every = self._restart_every_s
                due = every > 0 and (time.monotonic() - self._last_cycle_t) >= every
            if not due:
                continue
            if getattr(self._app_state, "stop_in_progress", False):
                continue
            if not system_running(self._app_state):
                # Operator stopped it (or it crashed): not ours to revive.
                with self._lock:
                    self._last_cycle_t = time.monotonic()
                continue
            self._cycle()

    def _cycle(self) -> None:
        logger.info("cycler: scheduled restart — stopping producer + engine")
        self._safe(stop_system, "restart/stop")
        if self._stop.is_set():
            return
        logger.info("cycler: scheduled restart — starting")
        self._safe(start_system, "restart/start")
        with self._lock:
            self._last_cycle_t = time.monotonic()
        self.cycles += 1

    def _safe(self, fn, what: str) -> None:
        try:
            fn(self._app_state)
            self.last_error = None
        except Exception as exc:  # never let the cycler thread die
            self.last_error = f"{what}: {exc}"
            logger.warning("cycler: %s failed", what, exc_info=True)
