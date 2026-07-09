"""Perception producer supervisor (Direction 1).

When ``backbone.yaml`` says ``ingestion.mode: points``, the Backbone is a pure
metric engine and SOMEONE must produce detections. That someone is the
standalone producer — ``python -m perception --config backbone.yaml`` —
spawned and reaped here alongside the Backbone by the control routes.

Why a subprocess and not an in-process thread: measured on the live rig, the
same tick (zone-scoped seg + pose on 2 cameras) costs ~100 ms standalone but
~2,200 ms inside the dashboard process — uvicorn + ORT thread pools + the GIL
inflate it 20x. The producer therefore owns its own interpreter and CUDA
context, at the price of a second RTSP decode per camera (the same price the
frames-mode Backbone paid, so Direction 1 is never worse than the baseline).
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import threading
import time

import yaml

logger = logging.getLogger(__name__)

_TERM_GRACE_S = 3.0
# Auto-respawn after an UNEXPECTED producer death (segfault, OOM, camera-
# library crash): the metric engine coasts, but only a running producer
# brings detections back. Deliberate STOPs never respawn. The window guard
# stops a crash-looping producer from thrashing the GPU.
_RESPAWN_DELAY_S = 3.0
_RESPAWN_MAX_PER_WINDOW = 5
_RESPAWN_WINDOW_S = 300.0


class PerceptionHost:
    """Spawn/stop the standalone producer with the Backbone's lifecycle."""

    def __init__(self, backbone_config_path) -> None:
        self._config_path = backbone_config_path
        self._proc: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self._lock = threading.Lock()
        self._deliberate_stop = False
        self._respawns: list[float] = []

    def points_mode(self) -> bool:
        try:
            with open(self._config_path) as fh:
                cfg = yaml.safe_load(fh) or {}
            return str(cfg.get("ingestion", {}).get("mode", "frames")) == "points"
        except Exception:
            return False

    def start(self) -> bool:
        """Spawn the producer if the config says points mode. True iff a
        producer process is alive after the call."""
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return True
            if not self.points_mode():
                return False
            self._deliberate_stop = False
            self._reap_strays()
            env = dict(os.environ)
            # Same CPU caging as the Backbone spawn: ORT/BLAS pools sized for
            # a co-tenant process, not the whole machine.
            env.setdefault("OMP_NUM_THREADS", "2")
            env.setdefault("OPENBLAS_NUM_THREADS", "2")
            try:
                self._proc = subprocess.Popen(
                    [sys.executable, "-m", "perception",
                     "--config", str(self._config_path)],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, env=env)
            except Exception:
                logger.warning("perception host: spawn failed", exc_info=True)
                self._proc = None
                return False
            self._reader = threading.Thread(
                target=self._pump_logs, args=(self._proc,), daemon=True,
                name="perception-logs")
            self._reader.start()
            logger.info("perception host: producer spawned (pid=%d)", self._proc.pid)
            return True

    def stop(self) -> None:
        with self._lock:
            self._deliberate_stop = True
            proc, self._proc = self._proc, None
        if proc is None or proc.poll() is not None:
            return
        logger.info("perception host: STOP → SIGTERM pid=%d", proc.pid)
        try:
            proc.terminate()
            try:
                proc.wait(timeout=_TERM_GRACE_S)
            except subprocess.TimeoutExpired:
                logger.warning("perception host: STOP → SIGKILL pid=%d", proc.pid)
                proc.kill()
                proc.wait(timeout=2.0)
        except ProcessLookupError:
            pass
        logger.info("perception host: producer stopped (exit=%s)", proc.returncode)

    def status(self) -> dict:
        proc = self._proc
        running = proc is not None and proc.poll() is None
        return {"running": running,
                "pid": proc.pid if running else None,
                "topology": "standalone"}

    # ---- internals ----

    def _pump_logs(self, proc: subprocess.Popen) -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\n")
            if line:
                logger.info("[perception] %s", line)
        # EOF — the producer is gone. Deliberate STOP handles its own logging;
        # anything else is a crash (a segfault logs NOTHING python-side) and
        # gets respawned so a camera-library crash never silently halts
        # detection until someone notices the frozen panels.
        rc = proc.wait()
        if self._deliberate_stop or rc == 0:
            return
        now = time.time()
        self._respawns = [t for t in self._respawns if now - t < _RESPAWN_WINDOW_S]
        if len(self._respawns) >= _RESPAWN_MAX_PER_WINDOW:
            logger.error(
                "perception host: producer died (exit=%s) %d times in %.0fs — "
                "GIVING UP; press START to retry", rc,
                len(self._respawns), _RESPAWN_WINDOW_S)
            return
        self._respawns.append(now)
        logger.warning(
            "perception host: producer died unexpectedly (exit=%s) — "
            "respawning in %.0fs", rc, _RESPAWN_DELAY_S)
        timer = threading.Timer(_RESPAWN_DELAY_S, self._respawn)
        timer.daemon = True
        timer.start()

    def _respawn(self) -> None:
        if self._deliberate_stop:
            return
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return       # someone already restarted it
        self.start()

    def _reap_strays(self) -> None:
        """SIGKILL producer orphans from a previous dashboard life — exactly
        the Backbone supervisor's stray policy, for the same reason."""
        try:
            out = subprocess.run(
                ["pgrep", "-f", r"python(3)? -m perception"],
                capture_output=True, text=True, timeout=5.0).stdout
        except Exception:
            return
        for pid_s in out.split():
            pid = int(pid_s)
            if pid == os.getpid():
                continue
            try:
                os.kill(pid, signal.SIGKILL)
                logger.warning("perception host: reaped stray producer pid %d", pid)
            except OSError:
                pass
