"""JobRunner — one background worker thread for phase jobs (from isiGen, CPU-only).

Phase jobs are plain Python functions (the `core/runners.py` functions) pulled from
a queue by a SINGLE worker, so only one Multical solve runs at a time. Each job's
log records are captured into a ring buffer AND teed to runs/jobs/<ts>_<p>_<phase>.log.
The live capture loop is NOT a job — it runs via routes_capture; jobs are the solves.
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

from .core import cleanup, progress


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
        self.progress: dict | None = None        # {"done", "total", "label"}
        self.log: deque[str] = deque(maxlen=log_buffer)

    def to_dict(self) -> dict:
        return {"id": self.id, "project": self.project, "phase": self.phase,
                "state": self.state, "result": self.result, "error": self.error,
                "created": self.created, "started": self.started,
                "finished": self.finished, "progress": self.progress}


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

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="isical-jobs")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._queue.put(None)  # type: ignore[arg-type]
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

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

            def _set_progress(done: int, total: int, label: str, _job: Job = job) -> None:
                _job.progress = {"done": done, "total": total, "label": label}

            progress.set_sink(_set_progress)
            cleanup.prepare_for_gpu(job.phase)
            try:
                job.result = job.fn()
                job.state = "done"
            except Exception as exc:
                # subprocess.CalledProcessError (e.g. a failing `multical` call)
                # carries the child's stderr — surface it so the operator sees the
                # real cause, not just "returned non-zero exit status 1".
                stderr = getattr(exc, "stderr", None)
                job.error = f"{type(exc).__name__}: {exc}"
                if stderr:
                    job.error += f"\n{stderr.strip() if isinstance(stderr, str) else stderr}"
                job.state = "failed"
                logging.getLogger(__name__).exception("job %s failed", job.id)
                if stderr:
                    logging.getLogger(__name__).error("job %s child stderr:\n%s", job.id, stderr)
            finally:
                progress.set_sink(None)
                job.progress = None
                cleanup.free_memory(f"after:{job.phase}")
                job.finished = datetime.now().isoformat(timespec="seconds")
                root.removeHandler(handler)
                handler.close()
