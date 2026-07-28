"""Perception producer supervisor (Direction 1).

When ``backbone.yaml`` says ``ingestion.mode: points``, the Backbone is a pure
metric engine and SOMEONE must produce detections. That someone is the
standalone producer — ``python -m isistream --config backbone.yaml`` —
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
import subprocess
import sys
import threading
import time

import yaml

from . import proc_reaper

logger = logging.getLogger(__name__)

_TERM_GRACE_S = 1.5
# Auto-respawn after an UNEXPECTED producer death (segfault, OOM, camera-
# library crash): the metric engine coasts, but only a running producer
# brings detections back. Deliberate STOPs never respawn. The window guard
# stops a crash-looping producer from thrashing the GPU.
_RESPAWN_DELAY_S = 3.0
_RESPAWN_MAX_PER_WINDOW = 5
_RESPAWN_WINDOW_S = 300.0


class IsistreamHost:
    """Spawn/stop the standalone producer with the Backbone's lifecycle."""

    def __init__(self, backbone_config_path, *,
                 instance_id: str | None = None) -> None:
        self._config_path = backbone_config_path
        # Same identity contract as BackboneSupervisor: stamped onto the
        # spawned producer, matched by the reaper. Pid-qualified fallback so a
        # bare-constructed host (tests) can never match a real process.
        self._instance_id = instance_id or proc_reaper.fallback_instance_id()
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
            self._purge_stale_frame_files()
            env = dict(os.environ)
            # Same CPU caging as the Backbone spawn: ORT/BLAS pools sized for
            # a co-tenant process, not the whole machine.
            env.setdefault("OMP_NUM_THREADS", "2")
            env.setdefault("OPENBLAS_NUM_THREADS", "2")
            # Identity stamp (reaper matches it via /proc/<pid>/environ) —
            # plain assignment so OUR id wins over anything inherited.
            env[proc_reaper.MARKER_ENV] = self._instance_id
            try:
                self._proc = subprocess.Popen(
                    [sys.executable, "-m", "isistream",
                     "--config", str(self._config_path)],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, env=env,
                    # Own session ⇒ stop() can killpg the whole tree.
                    start_new_session=True)
            except Exception:
                logger.warning("isistream host: spawn failed", exc_info=True)
                self._proc = None
                return False
            self._reader = threading.Thread(
                target=self._pump_logs, args=(self._proc,), daemon=True,
                name="isistream-logs")
            self._reader.start()
            logger.info("isistream host: producer spawned (pid=%d)", self._proc.pid)
            return True

    def stop(self) -> None:
        with self._lock:
            self._deliberate_stop = True
            proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            logger.info("isistream host: STOP → SIGTERM pid=%d", proc.pid)
            proc_reaper.terminate_tree(proc, term_grace_s=_TERM_GRACE_S)
            logger.info("isistream host: producer stopped (exit=%s)", proc.returncode)
        # Parity with the Backbone supervisor's STOP: sweep any stray producer
        # of OURS (or a pre-identity orphan) so STOP leaves a clean slate.
        # Safe re respawn: _deliberate_stop is already set, _respawn checks it.
        self._reap_strays()

    def reap_orphans_on_boot(self) -> None:
        """App-lifespan hook: adopt and kill producers orphaned by a previous
        run of THIS instance (dashboard OOM-killed without a clean STOP).
        Mirrors ``BackboneSupervisor.reap_orphans_on_boot``; no-op while a
        live producer is held."""
        if self._proc is not None and self._proc.poll() is None:
            return
        self._reap_strays()
        self._purge_stale_frame_files()

    @staticmethod
    def _purge_stale_frame_files(max_age_s: float = 5.0) -> int:
        """Unlink ``isi3d_frame_*`` buses whose writer is dead. A SIGKILLed
        producer (reaped orphan) never runs its clean unlink, so its frame
        files linger in /dev/shm. Frame files are CAMERA-keyed, not instance-
        keyed — so the freshness check is mandatory: a live sibling instance
        publishing the same camera keeps its bus (``latest()`` returns a
        frame), and only truly dead/corrupt buses are removed. Never keyed on
        mtime: mmap writes don't bump it."""
        try:
            from backbone.shared.frame_shm import FrameShmReader, _default_dir
        except Exception:                  # backbone not importable — skip
            return 0
        directory = _default_dir()
        prefix = "isi3d_frame_"
        purged = 0
        try:
            names = os.listdir(directory)
        except OSError:
            return 0
        for name in names:
            if not name.startswith(prefix):
                continue
            cam_id = name[len(prefix):]
            reader = FrameShmReader(cam_id, directory, max_age_s=max_age_s)
            try:
                fresh = reader.latest() is not None
            except Exception:
                fresh = False              # unreadable/corrupt → purge
            finally:
                close = getattr(reader, "close", None)
                if close:
                    try:
                        close()
                    except Exception:
                        pass
            if fresh:
                continue
            try:
                os.unlink(os.path.join(directory, name))
                purged += 1
                logger.info("isistream host: purged stale frame bus %s", name)
            except OSError:
                pass
        return purged

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
                logger.info("[isistream] %s", line)
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
                "isistream host: producer died (exit=%s) %d times in %.0fs — "
                "GIVING UP; press START to retry", rc,
                len(self._respawns), _RESPAWN_WINDOW_S)
            return
        self._respawns.append(now)
        logger.warning(
            "isistream host: producer died unexpectedly (exit=%s) — "
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
        """SIGKILL producer strays THIS INSTANCE owns (or pre-identity
        orphans) — the Backbone supervisor's stray policy, same rule, same
        shared implementation (/proc scan; the old ``pgrep`` regex was both
        host-wide and truncation-fragile). Honors ``ISI3D_DISABLE_REAP``."""
        exclude = {self._proc.pid} if self._proc is not None else None
        proc_reaper.kill_strays(
            proc_reaper.ISISTREAM_TOKEN, self._instance_id,
            why="reaped stray producer", exclude=exclude)
