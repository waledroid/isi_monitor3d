"""JobRunner — one background worker thread for the GPU-bound phase jobs.

Modeled on monitor_web's BackboneSupervisor discipline (daemon thread + log
ring buffer + state machine) but in-process: phase jobs are plain Python
functions (the SAME `core/runners.py` functions the CLIs call), pulled from a
queue by a SINGLE worker — the structural guarantee that only one job touches
the 12 GB GPU at a time. Each job's log records are captured by a temporary
logging.Handler into a ring buffer AND teed to runs/jobs/<ts>_<p>_<phase>.log.
"""

from __future__ import annotations

import logging
import queue
import threading
import uuid
from collections import deque
from collections.abc import Callable
from datetime import datetime
from pathlib import Path


class Job:
    def __init__(self, project: str, phase: str, fn: Callable[[], dict],
                 log_buffer: int) -> None:
        self.id = uuid.uuid4().hex[:10]
        self.project = project
        self.phase = phase
        self.fn = fn
        self.state = "queued"            # queued | running | done | failed
        self.result: dict | None = None
        self.error: str | None = None
        self.created = datetime.now().isoformat(timespec="seconds")
        self.started: str | None = None
        self.finished: str | None = None
        self.log: deque[str] = deque(maxlen=log_buffer)

    def to_dict(self) -> dict:
        return {"id": self.id, "project": self.project, "phase": self.phase,
                "state": self.state, "result": self.result, "error": self.error,
                "created": self.created, "started": self.started,
                "finished": self.finished}


class _JobLogHandler(logging.Handler):
    def __init__(self, job: Job, file_path: Path) -> None:
        super().__init__()
        self.job = job
        self._fh = open(file_path, "a")
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        line = self.format(record)
        self.job.log.append(line)
        self._fh.write(line + "\n")
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        finally:
            super().close()


class JobRunner:
    def __init__(self, runs_dir: Path, log_buffer: int = 1000) -> None:
        self._runs_dir = Path(runs_dir) / "jobs"
        self._log_buffer = int(log_buffer)
        self._queue: queue.Queue[Job] = queue.Queue()
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ---- lifecycle ----

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="isigen-jobs")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._queue.put(None)  # type: ignore[arg-type]  # wake the worker
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    # ---- API ----

    def submit(self, project: str, phase: str, fn: Callable[[], dict]) -> Job:
        """Enqueue a job; refuses a duplicate queued/running (project, phase)."""
        with self._lock:
            for j in self._jobs.values():
                if (j.project, j.phase) == (project, phase) and j.state in ("queued", "running"):
                    raise ValueError(f"{phase} is already {j.state} for {project}")
            job = Job(project, phase, fn, self._log_buffer)
            self._jobs[job.id] = job
        self._queue.put(job)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[dict]:
        with self._lock:
            return [j.to_dict() for j in
                    sorted(self._jobs.values(), key=lambda j: j.created, reverse=True)]

    # ---- worker ----

    def _run(self) -> None:
        while not self._stop.is_set():
            job = self._queue.get()
            if job is None or self._stop.is_set():
                return
            job.state = "running"
            job.started = datetime.now().isoformat(timespec="seconds")
            self._runs_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            handler = _JobLogHandler(job, self._runs_dir / f"{ts}_{job.project}_{job.phase}.log")
            root = logging.getLogger()
            root.addHandler(handler)
            try:
                job.result = job.fn()
                job.state = "done"
            except Exception as exc:
                job.error = f"{type(exc).__name__}: {exc}"
                job.state = "failed"
                logging.getLogger(__name__).exception("job %s failed", job.id)
            finally:
                job.finished = datetime.now().isoformat(timespec="seconds")
                root.removeHandler(handler)
                handler.close()
