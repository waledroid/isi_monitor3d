"""PerceptionHost — Direction-1 glue: mode detection + safe lifecycle."""

from __future__ import annotations

import yaml

from monitor_web.perception_host import PerceptionHost


def _write_cfg(tmp_path, mode):
    p = tmp_path / "backbone.yaml"
    p.write_text(yaml.safe_dump({
        "ingestion": {"mode": mode},
        "cameras": {"cam_a": {"source": {"name": "rtsp", "url": "rtsp://x"}}},
    }))
    return p


def test_points_mode_detection(tmp_path):
    assert PerceptionHost(_write_cfg(tmp_path, "points")).points_mode() is True
    assert PerceptionHost(_write_cfg(tmp_path, "frames")).points_mode() is False
    assert PerceptionHost(tmp_path / "missing.yaml").points_mode() is False


def test_start_is_a_noop_in_frames_mode(tmp_path):
    host = PerceptionHost(_write_cfg(tmp_path, "frames"))
    assert host.start() is False
    assert host.status() == {"running": False}
    host.stop()          # idempotent, never raises


def test_failed_start_releases_and_reports(tmp_path):
    # points mode but the config lacks calibration_path → build fails → False,
    # streams released, status stays not-running.
    host = PerceptionHost(_write_cfg(tmp_path, "points"))
    assert host.start() is False
    assert host.status() == {"running": False}
    host.stop()
