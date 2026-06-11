"""``BackboneSupervisor`` — spawn/kill subprocess + log ring buffer."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

from monitor_web.backbone_supervisor import BackboneSupervisor


def _wait_for(predicate, timeout: float = 3.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


@pytest.fixture
def echoer_script(tmp_path: Path) -> Path:
    """A child script that emits one line per second forever — a stand-in for the orchestrator."""
    script = tmp_path / "fake_orchestrator.py"
    script.write_text(
        "import sys, time\n"
        "print('fake orchestrator started', flush=True)\n"
        "i = 0\n"
        "while True:\n"
        "    print(f'tick {i}', flush=True)\n"
        "    i += 1\n"
        "    time.sleep(0.1)\n"
    )
    return script


@pytest.fixture
def crashing_script(tmp_path: Path) -> Path:
    """A child script that exits with code 13."""
    script = tmp_path / "crashing.py"
    script.write_text(
        "import sys\n"
        "print('about to crash', flush=True)\n"
        "sys.exit(13)\n"
    )
    return script


def _supervisor(script_path: Path, tmp_path: Path) -> BackboneSupervisor:
    """A supervisor wired to run ``python <script_path>`` instead of the real orchestrator."""
    sup = BackboneSupervisor(
        config_path=tmp_path / "no-such-config.yaml",
        terminate_timeout_s=1.0,
        log_buffer_size=100,
    )
    # Monkey-patch the command to point at our script rather than backbone.runtime.orchestrator.
    def patched_start():
        if sup.state == sup.STATE_RUNNING:
            return False
        import subprocess
        cmd = [sys.executable, str(script_path)]
        sup._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=1, text=True,
        )
        sup._last_exit_code = None
        import threading
        sup._reader_thread = threading.Thread(
            target=sup._read_stdout, daemon=True, name="bbn-stdout",
        )
        sup._reader_thread.start()
        return True

    sup.start = patched_start
    return sup


def test_starts_and_reaches_running_state(echoer_script, tmp_path) -> None:
    sup = _supervisor(echoer_script, tmp_path)
    assert sup.state == BackboneSupervisor.STATE_STOPPED
    assert sup.start() is True
    assert _wait_for(lambda: sup.state == BackboneSupervisor.STATE_RUNNING)
    sup.stop()


def test_double_start_is_idempotent(echoer_script, tmp_path) -> None:
    sup = _supervisor(echoer_script, tmp_path)
    assert sup.start() is True
    _wait_for(lambda: sup.state == BackboneSupervisor.STATE_RUNNING)
    assert sup.start() is False    # already running
    sup.stop()


def test_stop_sends_sigterm_and_collects_returncode(echoer_script, tmp_path) -> None:
    sup = _supervisor(echoer_script, tmp_path)
    sup.start()
    _wait_for(lambda: sup.state == BackboneSupervisor.STATE_RUNNING)
    assert sup.stop() is True
    assert sup.state == BackboneSupervisor.STATE_STOPPED
    assert sup.last_exit_code is not None


def test_stop_when_not_running_returns_false(echoer_script, tmp_path) -> None:
    sup = _supervisor(echoer_script, tmp_path)
    assert sup.stop() is False


def test_log_buffer_captures_stdout(echoer_script, tmp_path) -> None:
    sup = _supervisor(echoer_script, tmp_path)
    sup.start()
    assert _wait_for(lambda: any("tick" in line for line in sup.log_lines()))
    sup.stop()


def test_crashed_subprocess_reports_crashed_state(crashing_script, tmp_path) -> None:
    sup = _supervisor(crashing_script, tmp_path)
    sup.start()
    # Wait for the child to exit on its own.
    assert _wait_for(lambda: sup.state == BackboneSupervisor.STATE_CRASHED, timeout=3.0)
    assert sup.last_exit_code == 13


def test_pid_exposed_while_running(echoer_script, tmp_path) -> None:
    sup = _supervisor(echoer_script, tmp_path)
    sup.start()
    _wait_for(lambda: sup.state == BackboneSupervisor.STATE_RUNNING)
    assert isinstance(sup.pid, int)
    sup.stop()
    assert sup.pid is None


# ---- real start() preflight (the "START does nothing" bug) ----


def test_start_refuses_when_config_missing(tmp_path) -> None:
    """Real start() (not monkeypatched) must NOT spawn a doomed process when the
    config file is absent — it should fail fast with a clear log line."""
    sup = BackboneSupervisor(config_path=tmp_path / "absent.yaml")
    assert sup.start() is False
    assert sup.pid is None
    assert sup.last_exit_code == 2
    assert any("config not found" in line for line in sup.log_lines())


# ---- /proc-based backbone reaping (orphan + stray sweep on START/STOP) ----


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="needs Linux procfs")
def test_find_and_kill_finds_strays_via_proc(tmp_path) -> None:
    """The reaper scans /proc for EVERY backbone.runtime process — including one
    launched under a DIFFERENT config than the supervisor's — and SIGKILLs it.
    This is the robustness upgrade over the old config-scoped pgrep match."""
    import subprocess

    # A decoy whose argv contains "backbone.runtime" under an unrelated config path.
    decoy = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)",
         "-m", "backbone.runtime", "--config", "/tmp/some-other-config.yaml"],
    )
    try:
        assert _wait_for(lambda: (Path("/proc") / str(decoy.pid)).exists())
        # Supervisor is wired to a *different* config — proves the scan is host-wide,
        # not scoped to its own config path.
        sup = BackboneSupervisor(config_path=tmp_path / "unrelated.yaml")
        assert decoy.pid in sup._find_backbone_pids()
        assert os.getpid() not in sup._find_backbone_pids()   # never targets self

        killed = sup._kill_backbones(why="test-reaped")
        assert killed >= 1
        assert _wait_for(lambda: decoy.poll() is not None)    # decoy is dead
        assert any("test-reaped backbone pid" in line for line in sup.log_lines())
    finally:
        if decoy.poll() is None:
            decoy.kill()
            decoy.wait(timeout=2.0)


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="needs Linux procfs")
def test_find_backbone_pids_excludes_self(tmp_path) -> None:
    """Sanity: the scan never returns this process (no backbone.runtime in our argv),
    so a clean host yields no targets — the reaper is a no-op, not a footgun."""
    sup = BackboneSupervisor(config_path=tmp_path / "unrelated.yaml")
    pids = sup._find_backbone_pids()
    assert os.getpid() not in pids
    assert all(isinstance(p, int) for p in pids)


def test_default_cwd_is_repo_root() -> None:
    """The subprocess must run from the repo root so backbone.yaml's relative
    paths (calibration.json, onnx_path, zones_path) resolve like the CLI."""
    sup = BackboneSupervisor(config_path=Path("config/backbone.yaml"))
    # parents[2] of monitor_web/monitor_web/backbone_supervisor.py == repo root,
    # which is where backbone/ and config/ live.
    assert (sup._cwd / "backbone").is_dir()
    assert (sup._cwd / "config").exists()
