"""Memory-hygiene hook: gc/VRAM release + orphan reaping. Hermetic — the reaper
is fully mocked so no real process is ever signalled."""

import os

from src.core import cleanup


def test_free_memory_safe():
    v = cleanup.free_memory("test")          # must not raise with or without CUDA
    assert v is None or (isinstance(v, tuple) and len(v) == 2)


def test_reap_disabled_by_env(monkeypatch):
    monkeypatch.setenv("ISIGEN_DISABLE_REAP", "1")
    assert cleanup.reap_orphans() == []


def test_reap_only_orphaned_isigen_not_self(monkeypatch):
    monkeypatch.delenv("ISIGEN_DISABLE_REAP", raising=False)
    self_pid = os.getpid()
    orphan, live, other = 900001, 900002, 900003
    monkeypatch.setattr(cleanup, "_gpu_pids",
                        lambda: {self_pid: 500, orphan: 300, live: 300, other: 50})
    monkeypatch.setattr(cleanup, "_ppid",
                        lambda pid: 1 if pid in (orphan, other) else 4242)
    monkeypatch.setattr(cleanup, "_cmdline",
                        lambda pid: "python .../isigen/scripts/run_studio.py"
                        if pid in (orphan, live) else "python jupyter")
    killed = []
    monkeypatch.setattr(cleanup.os, "kill", lambda pid, sig: killed.append(pid))

    reaped = cleanup.reap_orphans()
    assert killed == [orphan]            # orphan + isiGen + on GPU
    assert reaped == [orphan]
    assert self_pid not in killed        # never self
    assert live not in killed            # isiGen but NOT orphaned (ppid 4242)
    assert other not in killed           # orphaned but not isiGen
