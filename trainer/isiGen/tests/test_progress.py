"""Progress hook — report() is a no-op without a sink, and the JobRunner
captures per-job progress onto the running job (shown as a Studio progress bar).
"""

import time

from src.core import progress


def test_report_is_noop_without_sink():
    progress.set_sink(None)
    progress.report(1, 10, "x")          # must not raise


def test_sink_receives_reports():
    seen = []
    progress.set_sink(lambda d, t, label: seen.append((d, t, label)))
    try:
        progress.report(3, 7, "masks")
    finally:
        progress.set_sink(None)
    assert seen == [(3, 7, "masks")]


def test_jobrunner_exposes_progress(tmp_path):
    from src.studio.jobs import JobRunner

    runner = JobRunner(tmp_path / "runs")
    runner.start()
    try:
        captured = {}

        def body():
            progress.report(2, 5, "demo")
            captured["job"] = next(iter(runner._jobs.values()))
            captured["progress"] = captured["job"].progress
            return {"ok": True}

        job = runner.submit("p", "maps", body)
        deadline = time.time() + 5
        while time.time() < deadline and runner.get(job.id).state not in ("done", "failed"):
            time.sleep(0.05)
        # progress was visible mid-run …
        assert captured["progress"] == {"done": 2, "total": 5, "label": "demo"}
        # … and is cleared once finished
        assert runner.get(job.id).state == "done"
        assert runner.get(job.id).to_dict()["progress"] is None
    finally:
        runner.stop()
