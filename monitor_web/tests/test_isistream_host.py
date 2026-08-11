"""IsistreamHost — Direction-1 producer supervisor: mode detection + lifecycle."""

from __future__ import annotations

import time

import yaml

from monitor_web.isistream_host import IsistreamHost


def _write_cfg(tmp_path, mode):
    p = tmp_path / "backbone.yaml"
    p.write_text(yaml.safe_dump({
        "ingestion": {"mode": mode},
        "cameras": {"cam_a": {"source": {"name": "rtsp", "url": "rtsp://x"}}},
    }))
    return p


def test_points_mode_detection(tmp_path):
    assert IsistreamHost(_write_cfg(tmp_path, "points")).points_mode() is True
    assert IsistreamHost(_write_cfg(tmp_path, "frames")).points_mode() is False
    assert IsistreamHost(tmp_path / "missing.yaml").points_mode() is False


def test_start_is_a_noop_in_frames_mode(tmp_path):
    host = IsistreamHost(_write_cfg(tmp_path, "frames"))
    assert host.start() is False
    assert host.status()["running"] is False
    host.stop()          # idempotent, never raises


def test_points_mode_spawns_and_stop_kills(tmp_path):
    # The spawned producer exits quickly (bad config: no calibration_path),
    # but the host must spawn it, report it, and stop() must be clean.
    host = IsistreamHost(_write_cfg(tmp_path, "points"))
    assert host.start() is True
    host.stop()
    deadline = time.time() + 5.0
    while host.status()["running"] and time.time() < deadline:
        time.sleep(0.1)
    assert host.status()["running"] is False


def test_reap_goes_through_shared_finder(tmp_path, monkeypatch):
    """The stray sweep must use the shared identity-scoped finder with the
    isistream token and THIS host's instance id (the pgrep sweep is gone —
    it was host-wide and could kill a live production producer)."""
    from monitor_web import proc_reaper

    calls: list[tuple] = []
    monkeypatch.setattr(
        proc_reaper, "find_strays",
        lambda token, instance_id, exclude=None: calls.append(
            (token, instance_id, exclude)) or [])
    monkeypatch.delenv(proc_reaper.DISABLE_ENV, raising=False)
    host = IsistreamHost(_write_cfg(tmp_path, "points"), instance_id="test-is-1")
    host._reap_strays()
    assert calls == [(proc_reaper.ISISTREAM_TOKEN, "test-is-1", None)]


def test_spawn_sets_session_and_marker(tmp_path, monkeypatch):
    """The producer must be a session leader stamped with the instance id."""
    import subprocess as sp

    captured: dict = {}

    class _FakeProc:
        pid = 4243
        stdout = iter(())

        def poll(self):
            return None

    monkeypatch.setattr(sp, "Popen",
                        lambda cmd, **kw: captured.update(kw) or _FakeProc())
    host = IsistreamHost(_write_cfg(tmp_path, "points"), instance_id="test-is-2")
    monkeypatch.setattr(host, "_reap_strays", lambda: None)
    assert host.start() is True
    assert captured["start_new_session"] is True
    assert captured["env"]["ISI3D_INSTANCE_ID"] == "test-is-2"


def test_purge_stale_frame_files_keeps_fresh_bus(tmp_path, monkeypatch):
    """A dead writer's bus is unlinked; a live (fresh) bus survives — frame
    files are camera-keyed, so a sibling instance's live bus must be kept."""
    import time as _time

    import numpy as np
    from backbone.shared.frame_shm import FrameShmWriter, shm_path

    shm_dir = tmp_path / "shm"
    shm_dir.mkdir()
    monkeypatch.setenv("ISI3D_SHM_DIR", str(shm_dir))
    img = np.zeros((4, 4, 3), dtype=np.uint8)

    stale = FrameShmWriter("cam_stale", str(shm_dir))
    stale.write(img, capture_ts=_time.time() - 100.0)
    stale.close()
    fresh = FrameShmWriter("cam_fresh", str(shm_dir))
    fresh.write(img, capture_ts=_time.time())

    purged = IsistreamHost._purge_stale_frame_files()
    assert purged == 1
    import os as _os
    assert not _os.path.exists(shm_path("cam_stale", str(shm_dir)))
    assert _os.path.exists(shm_path("cam_fresh", str(shm_dir)))
    fresh.close()
