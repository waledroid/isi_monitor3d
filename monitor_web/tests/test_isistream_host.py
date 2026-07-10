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
