"""Spawn / terminate the Backbone orchestrator as a subprocess.

START launches ``python -m backbone.runtime --config <yaml>``.
stdout + stderr are routed into a ring buffer of the last N lines, which the
``GET /api/logs`` HTMX partial reads.

STOP sends SIGTERM, waits ``terminate_timeout_s``, then SIGKILL.

State machine:

    stopped  ─START─►  running  ─SIGTERM/clean exit─►  stopped
                   │
                   └─crash────►  crashed  ─START─►  running

The supervisor itself runs entirely in the FastAPI process — no extra
threads beyond the stdout reader. Subprocess polling is event-driven (we
poll(returncode) on each status request rather than running a watchdog
thread; cheap enough at the dashboard's read rate).
"""

from __future__ import annotations

import ctypes
import gc
import logging
import os
import shlex
import signal
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path

logger = logging.getLogger(__name__)

# Repo root: monitor_web/monitor_web/backbone_supervisor.py -> parents[2].
# The orchestrator subprocess is launched with this as its cwd so that relative
# paths inside backbone.yaml (config/backbone.yaml, calibration.json, onnx_path,
# zones_path, …) resolve the same way they do for the CLI run from the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]


class BackboneSupervisor:
    """Manages one child process running the orchestrator."""

    STATE_STOPPED = "stopped"
    STATE_RUNNING = "running"
    STATE_CRASHED = "crashed"

    def __init__(
        self,
        config_path: Path,
        *,
        terminate_timeout_s: float = 5.0,
        log_buffer_size: int = 500,
        python_exe: str | None = None,
        cwd: Path | None = None,
    ) -> None:
        self._config_path = Path(config_path)
        self._terminate_timeout = float(terminate_timeout_s)
        self._python = python_exe or sys.executable
        self._cwd = Path(cwd) if cwd is not None else _REPO_ROOT
        self._proc: subprocess.Popen | None = None
        self._reader_thread: threading.Thread | None = None
        self._log_buffer: deque[str] = deque(maxlen=int(log_buffer_size))
        self._log_lock = threading.Lock()
        self._last_exit_code: int | None = None
        # True iff the most recent termination was operator-initiated (stop()).
        # Distinguishes a clean SIGTERM-driven exit (returncode == -15) from an
        # unexpected crash that happens to have the same negative returncode.
        self._stop_requested: bool = False

    # ---- state ----

    @property
    def state(self) -> str:
        if self._proc is None:
            if self._last_exit_code in (None, 0):
                return self.STATE_STOPPED
            if self._stop_requested:
                return self.STATE_STOPPED
            return self.STATE_CRASHED
        rc = self._proc.poll()
        if rc is None:
            return self.STATE_RUNNING
        # Subprocess has exited since the last check.
        self._last_exit_code = rc
        self._proc = None
        if rc == 0 or self._stop_requested:
            return self.STATE_STOPPED
        return self.STATE_CRASHED

    @property
    def pid(self) -> int | None:
        if self._proc is None:
            return None
        return self._proc.pid

    @property
    def last_exit_code(self) -> int | None:
        return self._last_exit_code

    def log_lines(self, n: int | None = None) -> list[str]:
        with self._log_lock:
            if n is None:
                return list(self._log_buffer)
            return list(self._log_buffer)[-n:]

    # ---- control ----

    def _config_abspath(self) -> Path:
        """Absolute path the subprocess will read, given its cwd (repo root)."""
        p = self._config_path
        return p if p.is_absolute() else (self._cwd / p)

    def _free_memory(self) -> None:
        """Release the DASHBOARD's own host RAM + GPU memory:

          - drop the dashboard's ONNX Runtime CUDA sessions (cam preview / pose /
            per-zone detectors) → frees their VRAM + host RAM (they rebuild lazily on
            the next preview frame),
          - Python GC + return freed heap to the OS (glibc ``malloc_trim``) → drops RSS.

        Called before a spawn (START headroom) AND after a stop (release on idle).
        Every step is guarded so cleanup can never break START/STOP."""
        try:
            from .detection_overlay import reset_detector
            reset_detector()
        except Exception as exc:
            logger.debug("supervisor: detector reset skipped: %s", exc)
        try:
            gc.collect()
            ctypes.CDLL("libc.so.6").malloc_trim(0)   # glibc-only; no-ops elsewhere
        except Exception as exc:
            logger.debug("supervisor: malloc_trim skipped: %s", exc)

    def _find_backbone_pids(self, *, exclude: set[int] | None = None) -> list[int]:
        """Every live ``backbone.runtime`` process on the host, read straight from
        ``/proc``. This is the ground truth: it finds strays spawned under *any*
        config, and is immune to ``pgrep``'s regex-escaping and the kernel's
        command-line-length truncation (a long ``--config`` path can push the match
        token past pgrep's window). Always skips this dashboard's own PID; ``exclude``
        drops any extra PIDs the caller is handling through their own process handle."""
        skip = {os.getpid()}
        if exclude:
            skip |= set(exclude)
        found: list[int] = []
        try:
            names = os.listdir("/proc")
        except OSError as exc:           # not Linux / no procfs — nothing we can scan
            logger.debug("supervisor: /proc scan unavailable: %s", exc)
            return found
        for name in names:
            if not name.isdigit():
                continue
            pid = int(name)
            if pid in skip:
                continue
            try:
                with open(f"/proc/{pid}/cmdline", "rb") as fh:
                    cmd = fh.read()            # NUL-separated argv
            except OSError:
                continue                       # process vanished or unreadable
            if b"backbone.runtime" in cmd:
                found.append(pid)
        return found

    def _kill_backbones(self, *, why: str, exclude: set[int] | None = None) -> int:
        """SIGKILL every live ``backbone.runtime`` process (see :meth:`_find_backbone_pids`)
        and return how many were signalled. Orphans are already disconnected, so an
        immediate SIGKILL is correct and — unlike a SIGTERM/grace/SIGKILL dance — never
        blocks the event loop. Guarded per-PID so a race (already gone, or not ours)
        can't abort the sweep."""
        killed = 0
        for pid in self._find_backbone_pids(exclude=exclude):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                continue                       # won the race; already dead
            except PermissionError:
                logger.warning("supervisor: cannot kill backbone pid %s (not ours)", pid)
                continue
            killed += 1
            logger.warning("supervisor: %s backbone pid %s", why, pid)
            with self._log_lock:
                self._log_buffer.append(f"[supervisor] {why} backbone pid {pid}")
        return killed

    def _reap_orphans(self) -> None:
        """Free up host RAM + GPU before spawning the Backbone, so it gets maximum
        headroom (the dashboard runs on the same 12 GB WSL VM + shares the GPU):

          1. SIGKILL every stray ``backbone.runtime`` (orphans from a previous
             dashboard that died without STOP — e.g. OOM-killed) so they can't
             accumulate (~1.5 GB each) and exhaust RAM.
          2. Free the dashboard's own RAM + GPU sessions (see :meth:`_free_memory`).

        Called before every spawn; safe because start() holds no live process here."""
        self._kill_backbones(why="reaped orphaned")
        self._free_memory()

    def reap_orphans_on_boot(self) -> None:
        """Public hook for the app lifespan: kill orphaned Backbones for this config
        that a previous dashboard left behind (OOM-killed without a clean STOP).

        START already reaps before every spawn, but a crashed session's Backbone
        lingers (~1.5 GB) for as long as the *next* dashboard sits idle before the
        operator presses START — long enough to OOM-kill the new dashboard. Reaping
        at boot closes that window. No-op when no live process is held (boot)."""
        if self._proc is not None:
            return
        self._reap_orphans()

    def start(self) -> bool:
        """Spawn the orchestrator. Returns True if a new process was started.

        Preflights the config file first: a missing backbone.yaml is the single
        most common reason START "does nothing" (the orchestrator exits with code
        2 before printing anything useful), so we catch it here and write a crisp,
        actionable line to the log buffer instead of spawning a doomed process.
        """
        if self.state == self.STATE_RUNNING:
            return False
        cfg_abs = self._config_abspath()
        if not cfg_abs.exists():
            msg = (f"[supervisor] cannot start: config not found at {cfg_abs} "
                   f"— create it in Settings (Save) or copy config/backbone.yaml.example")
            logger.error(msg)
            with self._log_lock:
                self._log_buffer.append(msg)
            self._last_exit_code = 2
            return False
        # Reap any orphaned Backbone for this config left by a previous dashboard
        # that died without a clean STOP — otherwise they accumulate (~1.5 GB each)
        # and exhaust host RAM → the kernel OOM-kills the dashboard ("Killed").
        self._reap_orphans()
        cmd = [
            self._python, "-m", "backbone.runtime",   # package __main__ (no runpy warning)
            "--config", str(self._config_path),
        ]
        logger.info("supervisor: starting %s (cwd=%s)", shlex.join(cmd), self._cwd)
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                text=True,
                env={**os.environ},
                cwd=str(self._cwd),
            )
        except FileNotFoundError as exc:
            logger.error("supervisor: failed to spawn (%s)", exc)
            with self._log_lock:
                self._log_buffer.append(f"[supervisor] spawn failed: {exc}")
            self._last_exit_code = 127
            return False
        self._last_exit_code = None
        self._stop_requested = False
        self._reader_thread = threading.Thread(
            target=self._read_stdout, daemon=True, name="backbone-stdout",
        )
        self._reader_thread.start()
        return True

    def stop(self) -> bool:
        """SIGTERM, wait, SIGKILL if still alive. True iff a running process was stopped."""
        if self.state != self.STATE_RUNNING or self._proc is None:
            return False
        self._stop_requested = True   # mark BEFORE terminate so state() reads cleanly
        proc = self._proc
        logger.info("supervisor: SIGTERM pid=%d", proc.pid)
        try:
            proc.terminate()
            try:
                proc.wait(timeout=self._terminate_timeout)
            except subprocess.TimeoutExpired:
                logger.warning("supervisor: SIGKILL pid=%d (terminate timed out)", proc.pid)
                proc.kill()
                proc.wait(timeout=2.0)
        except ProcessLookupError:
            pass   # already dead
        finally:
            rc = proc.returncode
            self._last_exit_code = rc if rc is not None else -signal.SIGTERM
            self._proc = None
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=2.0)
            self._reader_thread = None
        # Our tracked child is gone — now sweep up ANY other backbone strays (orphans
        # from earlier crashes that STOP would otherwise leave running) so STOP leaves
        # a truly clean slate, not just the one process we launched this session.
        self._kill_backbones(why="reaped stray")
        # The Backbone subprocesses are gone (their RAM/VRAM freed by the OS). Also
        # release the DASHBOARD's own preview detector sessions + trim host RAM, so a
        # stopped system returns to a lean, camera-only footprint instead of holding on.
        self._free_memory()
        return True

    # ---- internals ----

    def _read_stdout(self) -> None:
        assert self._proc is not None
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            stripped = line.rstrip("\n")
            if not stripped:
                continue
            with self._log_lock:
                self._log_buffer.append(stripped)
