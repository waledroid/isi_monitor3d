"""Instance-identity reaper: the safety pins.

Every test that spawns a decoy stamps or strips the ``ISI3D_INSTANCE_ID``
marker to exercise one arm of the reap rule. The autouse suite guard keeps
``ISI3D_DISABLE_REAP=1`` and patches the module-level ``find_strays`` — tests
that need the real thing use the ``real_find_strays`` fixture or call the
private probes directly, and tests that kill only ever target their own
decoys.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from monitor_web import proc_reaper

pytestmark = pytest.mark.skipif(not Path("/proc").is_dir(),
                                reason="needs Linux procfs")

_TOKEN_ARGV = ["-m", "isistream"]     # puts ISISTREAM_TOKEN into the cmdline


def _decoy(marker: str | None):
    env = {k: v for k, v in os.environ.items()
           if k != proc_reaper.MARKER_ENV}
    if marker is not None:
        env[proc_reaper.MARKER_ENV] = marker
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)", *_TOKEN_ARGV],
        env=env)


def _wait_pid_visible(pid: int, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if (Path("/proc") / str(pid) / "environ").exists():
            return True
        time.sleep(0.05)
    return False


@pytest.fixture
def reaper_armed(monkeypatch, real_find_strays):
    """Re-arm the module inside one test: real finder + kill switch off."""
    monkeypatch.setattr(proc_reaper, "find_strays", real_find_strays)
    monkeypatch.delenv(proc_reaper.DISABLE_ENV, raising=False)


def test_marker_match_finds_own_stray(real_find_strays):
    p = _decoy("T1")
    try:
        assert _wait_pid_visible(p.pid)
        assert p.pid in real_find_strays(proc_reaper.ISISTREAM_TOKEN, "T1")
    finally:
        p.kill()
        p.wait(timeout=2)


def test_other_instance_marker_excluded(real_find_strays):
    """THE sibling-safety pin: another instance's process is never listed."""
    p = _decoy("T1")
    try:
        assert _wait_pid_visible(p.pid)
        assert p.pid not in real_find_strays(proc_reaper.ISISTREAM_TOKEN, "T2")
        # ...and not even a marker that PREFIXES ours may match.
        assert p.pid not in real_find_strays(proc_reaper.ISISTREAM_TOKEN, "T")
    finally:
        p.kill()
        p.wait(timeout=2)


def test_markerless_child_of_live_parent_excluded(real_find_strays):
    """A pre-identity process whose parent is alive belongs to someone."""
    p = _decoy(None)                      # markerless, ppid == pytest != 1
    try:
        assert _wait_pid_visible(p.pid)
        assert p.pid not in real_find_strays(proc_reaper.ISISTREAM_TOKEN, "T1")
    finally:
        p.kill()
        p.wait(timeout=2)


def test_markerless_orphan_reaped_legacy_arm(real_find_strays, monkeypatch):
    """A markerless TRUE orphan (ppid==1) is adopted by any instance."""
    p = _decoy(None)
    try:
        assert _wait_pid_visible(p.pid)
        monkeypatch.setattr(proc_reaper, "_ppid", lambda pid: 1)
        assert p.pid in real_find_strays(proc_reaper.ISISTREAM_TOKEN, "T1")
    finally:
        p.kill()
        p.wait(timeout=2)


def test_marked_foreign_orphan_survives_legacy_arm(real_find_strays, monkeypatch):
    """An orphan MARKED by another instance is left for its owner to adopt."""
    p = _decoy("OTHER")
    try:
        assert _wait_pid_visible(p.pid)
        monkeypatch.setattr(proc_reaper, "_ppid", lambda pid: 1)
        assert p.pid not in real_find_strays(proc_reaper.ISISTREAM_TOKEN, "T1")
    finally:
        p.kill()
        p.wait(timeout=2)


def test_unreadable_environ_never_killed(real_find_strays, monkeypatch):
    p = _decoy("T1")
    try:
        assert _wait_pid_visible(p.pid)
        monkeypatch.setattr(proc_reaper, "_read_environ", lambda pid: None)
        assert p.pid not in real_find_strays(proc_reaper.ISISTREAM_TOKEN, "T1")
    finally:
        p.kill()
        p.wait(timeout=2)


def test_disable_env_blocks_kill_but_not_find(monkeypatch, real_find_strays):
    monkeypatch.setattr(proc_reaper, "find_strays", real_find_strays)
    monkeypatch.setenv(proc_reaper.DISABLE_ENV, "1")
    p = _decoy("T1")
    try:
        assert _wait_pid_visible(p.pid)
        assert p.pid in real_find_strays(proc_reaper.ISISTREAM_TOKEN, "T1")
        assert proc_reaper.kill_strays(
            proc_reaper.ISISTREAM_TOKEN, "T1", why="test") == 0
        assert p.poll() is None            # decoy alive: kill was inert
    finally:
        p.kill()
        p.wait(timeout=2)


def test_kill_strays_kills_only_matching(reaper_armed):
    mine = _decoy("T-kill")
    other = _decoy("T-other")
    try:
        assert _wait_pid_visible(mine.pid) and _wait_pid_visible(other.pid)
        killed = proc_reaper.kill_strays(
            proc_reaper.ISISTREAM_TOKEN, "T-kill", why="test")
        assert killed == 1
        assert mine.wait(timeout=3) is not None
        assert other.poll() is None        # sibling untouched
    finally:
        for p in (mine, other):
            if p.poll() is None:
                p.kill()
                p.wait(timeout=2)


def _group_members(pgid: int) -> list[int]:
    members = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat", "rb") as fh:
                stat = fh.read()
            fields = stat[stat.rindex(b") ") + 2:].split()
            if int(fields[2]) == pgid:     # field 5 overall == pgrp
                members.append(int(name))
        except (OSError, ValueError, IndexError):
            continue
    return members


def test_terminate_tree_kills_grandchild():
    """THE ffprobe pin: a SIGTERM-ignoring parent's grandchild dies with the
    group when the child was spawned as a session leader."""
    script = (
        "import signal, subprocess, sys, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "c = subprocess.Popen(['sleep', '300'])\n"
        "print('ready', flush=True)\n"
        "time.sleep(300)\n"
    )
    proc = subprocess.Popen([sys.executable, "-u", "-c", script],
                            stdout=subprocess.PIPE, text=True,
                            start_new_session=True)
    try:
        assert proc.stdout is not None and "ready" in proc.stdout.readline()
        pgid = os.getpgid(proc.pid)
        assert pgid == proc.pid            # session leader
        assert len(_group_members(pgid)) >= 2   # parent + sleep grandchild
        method = proc_reaper.terminate_tree(proc, term_grace_s=0.5)
        assert method == "sigkill"         # parent ignored SIGTERM
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and _group_members(pgid):
            time.sleep(0.05)
        assert _group_members(pgid) == []  # grandchild died with the group
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def test_kill_stray_plain_kill_for_non_leader():
    """A legacy stray that is not a group leader gets os.kill, never killpg —
    we must not signal a group we don't own (it could be OUR group)."""
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert os.getpgid(p.pid) != p.pid   # shares pytest's group
        assert proc_reaper.kill_stray(p.pid) is True
        assert p.wait(timeout=3) is not None
        # and pytest itself is obviously still alive to assert this.
    finally:
        if p.poll() is None:
            p.kill()
        p.wait(timeout=2)


def test_fallback_instance_id_is_pid_qualified(monkeypatch):
    monkeypatch.delenv(proc_reaper.MARKER_ENV, raising=False)
    assert f"pid{os.getpid()}" in proc_reaper.fallback_instance_id()
