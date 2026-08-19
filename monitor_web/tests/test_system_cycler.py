"""SystemCycler — auto-START on launch + scheduled full restart (fakes only)."""
from __future__ import annotations

import time
from types import SimpleNamespace

import monitor_web.system_cycler as sc
from monitor_web.system_cycler import SystemCycler


class _Sup:
    def __init__(self, running=False):
        self.state = "running" if running else "stopped"
        self.pid = 1 if running else None
        self.last_exit_code = None
        self.calls: list[str] = []

    def start(self):
        self.calls.append("start")
        self.state, self.pid = "running", 1
        return True

    def stop(self):
        self.calls.append("stop")
        self.state, self.pid = "stopped", None
        return True

    def log_lines(self, n):
        return ["backbone built"]


class _Prod:
    def __init__(self):
        self.calls: list[str] = []

    def points_mode(self):
        return True

    def start(self):
        self.calls.append("start")
        return True

    def stop(self):
        self.calls.append("stop")


def _state(running=False):
    return SimpleNamespace(supervisor=_Sup(running), isistream=_Prod(), bus=None,
                           stop_in_progress=False, last_stop_done=-1e9)


def _fast(monkeypatch):
    monkeypatch.setattr(sc, "_TICK_S", 0.02)
    monkeypatch.setattr(sc, "_BOOT_SETTLE_S", 0.02)
    monkeypatch.setattr(sc, "_MIN_RESTART_MIN", 0.0)
    monkeypatch.setattr("monitor_web.system_control.BOOT_TIMEOUT_S", 0.0)


def _wait(pred, timeout=3.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.01)
    return False


def test_auto_start_launches_engine_then_producer(monkeypatch):
    _fast(monkeypatch)
    st = _state(running=False)
    cy = SystemCycler(st, auto_start=True, restart_every_min=0)
    cy.start()
    try:
        assert _wait(lambda: st.supervisor.calls == ["start"] and st.isistream.calls == ["start"])
        assert st.supervisor.state == "running"
    finally:
        cy.stop()


def test_no_auto_start_when_disabled(monkeypatch):
    _fast(monkeypatch)
    st = _state(running=False)
    cy = SystemCycler(st, auto_start=False, restart_every_min=0)
    cy.start()
    try:
        time.sleep(0.15)
        assert st.supervisor.calls == []
    finally:
        cy.stop()


def test_scheduled_restart_cycles_running_system(monkeypatch):
    _fast(monkeypatch)
    st = _state(running=True)
    cy = SystemCycler(st, auto_start=False, restart_every_min=0.0005)   # 30 ms
    cy.start()
    try:
        assert _wait(lambda: cy.cycles >= 1)
        # producer stopped before engine, then engine started before producer
        assert st.isistream.calls[:1] == ["stop"]
        assert st.supervisor.calls[:2] == ["stop", "start"]
        assert st.isistream.calls[:2] == ["stop", "start"]
        assert cy.status()["restart_every_min"] > 0
    finally:
        cy.stop()


def test_restart_respects_operator_stop(monkeypatch):
    _fast(monkeypatch)
    st = _state(running=False)        # operator stopped it
    cy = SystemCycler(st, auto_start=False, restart_every_min=0.0005)
    cy.start()
    try:
        time.sleep(0.2)
        assert st.supervisor.calls == [] and cy.cycles == 0
    finally:
        cy.stop()


def test_restart_skipped_while_stop_in_progress(monkeypatch):
    _fast(monkeypatch)
    st = _state(running=True)
    st.stop_in_progress = True
    cy = SystemCycler(st, auto_start=False, restart_every_min=0.0005)
    cy.start()
    try:
        time.sleep(0.2)
        assert cy.cycles == 0 and st.supervisor.calls == []
    finally:
        cy.stop()


def test_configure_hot_applies_and_norm(monkeypatch):
    _fast(monkeypatch)
    st = _state(running=True)
    cy = SystemCycler(st, auto_start=False, restart_every_min=0)
    assert cy.status()["restart_every_min"] == 0 and cy.status()["next_restart_in_s"] is None
    cy.configure(restart_every_min=60)
    s = cy.status()
    assert s["restart_every_min"] == 60 and 0 < s["next_restart_in_s"] <= 3600
    cy.configure(restart_every_min="garbage")
    assert cy.status()["restart_every_min"] == 0
    monkeypatch.setattr(sc, "_MIN_RESTART_MIN", 5.0)
    cy.configure(restart_every_min=1)          # below the floor → clamped to 5
    assert cy.status()["restart_every_min"] == 5
