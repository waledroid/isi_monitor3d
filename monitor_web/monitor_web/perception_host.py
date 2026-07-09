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

import yaml

logger = logging.getLogger(__name__)

_TERM_GRACE_S = 3.0


class PerceptionHost:
    """Spawn/stop the standalone producer with the Backbone's lifecycle."""

    def __init__(self, backbone_config_path) -> None:
        self._config_path = backbone_config_path
        self._proc: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self._lock = threading.Lock()

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
