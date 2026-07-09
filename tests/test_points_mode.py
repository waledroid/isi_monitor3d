"""Direction 1 — ``ingestion.mode: points``: the Backbone as a metric engine.

Covers the ingest listener (``backbone/ingestion/points_in.py``), the
points-mode orchestrator end-to-end (DetectionSetMessages over real loopback
UDP → Track2D out), and the DIFFERENTIAL test: the same detections through
frames mode and points mode must produce identical Track2D streams — the
proof that Direction 1 changed the plumbing, not the physics.
"""

from __future__ import annotations

import json
import socket
import time
from pathlib import Path

import yaml

from backbone.comms.schemas import DetectionSetMessage, WireDetection
from backbone.comms.udp_sink import send_json_datagram
from backbone.ingestion.points_in import (
    DetectionIngest,
    DetectionSet,
    detection_set_from_message,
)
from backbone.runtime import Orchestrator

from .test_orchestrator import (
    _bind_receiver,
    _make_frame_pair,
    _write_calibration,
    _write_config,
    _write_stub_onnx,
)


def _write_points_config(tmp_path: Path, *, calibration_path: Path,
                         udp_port: int, listen_port: int = 0) -> Path:
    config = {
        "calibration_path": str(calibration_path),
        "cameras": {
            "cam_a": {"source": {"name": "replay", "frames": []}},
            "cam_b": {"source": {"name": "replay", "frames": []}},
        },
        "ingestion": {
            "mode": "points",
            "points": {"listen_host": "127.0.0.1", "listen_port": listen_port,
                       "max_skew_ms": 33.0},
        },
        "homography": {
            "tracker": {"plugin": "bytetrack"},
            "track_config": {"min_hits_to_confirm": 1, "max_lost_frames": 30},
        },
        "metadata": {
            "sinks": [{"plugin": "udp", "host": "127.0.0.1", "port": udp_port}],
        },
    }
    path = tmp_path / "backbone_points.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _det_msg(camera_id: str, ts: float, seq: int, *, cls: str = "person",
             foot=(500.0, 500.0), mask_poly=None) -> DetectionSetMessage:
    return DetectionSetMessage(
        ts=ts, camera_id=camera_id, frame_wh=(1000, 1000), seq=seq,
        dets=(WireDetection(cls=cls, confidence=0.9,
                            bbox_xyxy=(foot[0] - 40, foot[1] - 160, foot[0] + 40, foot[1]),
                            foot_uv=foot, mask_poly=mask_poly),))


# ---- DetectionIngest unit tests ----


def test_ingest_delivers_sets_and_counts_seq_gaps() -> None:
    got: list[DetectionSet] = []
    ing = DetectionIngest(["cam_a", "cam_b"], port=0, on_set=got.append)
    ing.start()
    try:
        send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        for seq in (0, 1, 4):   # gap of 2 (2, 3 lost)
            send_json_datagram(
                send, ing.address,
                _det_msg("cam_a", 1.0 + seq * 0.04, seq).model_dump_json().encode())
        deadline = time.time() + 2.0
        while len(got) < 3 and time.time() < deadline:
            time.sleep(0.02)
        assert len(got) == 3
        assert got[0].camera_id == "cam_a" and got[0].detections[0].cls == "person"
        assert ing.seq_gaps_by_camera["cam_a"] == 2
        assert ing.sets_by_camera["cam_a"] == 3
        send.close()
    finally:
        ing.stop()


def test_ingest_explicit_empty_heartbeat_delivers() -> None:
    got: list[DetectionSet] = []
    ing = DetectionIngest(["cam_a"], port=0, on_set=got.append)
    ing.start()
    try:
        send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        msg = DetectionSetMessage(ts=1.0, camera_id="cam_a",
                                  frame_wh=(1280, 720), seq=0, dets=())
        send_json_datagram(send, ing.address, msg.model_dump_json().encode())
        deadline = time.time() + 2.0
        while not got and time.time() < deadline:
            time.sleep(0.02)
        assert got and got[0].detections == []
        send.close()
    finally:
        ing.stop()


def test_ingest_reassembles_fragmented_sets() -> None:
    got: list[DetectionSet] = []
    ing = DetectionIngest(["cam_a"], port=0, on_set=got.append)
    ing.start()
    try:
        send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        poly = tuple((float(400 + i % 200), float(400 + (i * 7) % 200))
                     for i in range(300))
        msg = _det_msg("cam_a", 2.0, 0, cls="palette",
                       mask_poly=poly)
        payload = msg.model_dump_json().encode()
        assert len(payload) > 1300
        send_json_datagram(send, ing.address, payload)
        deadline = time.time() + 2.0
        while not got and time.time() < deadline:
            time.sleep(0.02)
        assert got and got[0].detections[0].mask is not None
        send.close()
    finally:
        ing.stop()


def test_ingest_ignores_unknown_camera_and_counts_malformed() -> None:
    got: list[DetectionSet] = []
    ing = DetectionIngest(["cam_a"], port=0, on_set=got.append)
    ing.start()
    try:
        send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        send_json_datagram(send, ing.address,
                           _det_msg("cam_zz", 1.0, 0).model_dump_json().encode())
        send.sendto(b"not json", ing.address)
        time.sleep(0.3)
        assert got == []
        assert ing.dropped_malformed == 1
        send.close()
    finally:
        ing.stop()


def test_mask_poly_rasterizes_crop_local() -> None:
    msg = _det_msg("cam_a", 1.0, 0, cls="palette", foot=(100.0, 200.0),
                   mask_poly=((60.0, 40.0), (140.0, 40.0), (140.0, 200.0), (60.0, 200.0)))
    ds = detection_set_from_message(msg)
    d = ds.detections[0]
    assert d.mask is not None and d.mask_offset_xy is not None
    ox, oy = d.mask_offset_xy
    # The mask lives in bbox-crop coordinates; area ≈ the polygon's area.
    assert (ox, oy) == (60, 40)
    assert 0.9 < d.mask.sum() / (80 * 160) < 1.1


# ---- points-mode orchestrator e2e (real loopback UDP in AND out) ----


def test_points_mode_e2e_udp_in_tracks_out(tmp_path: Path) -> None:
    cal_path = _write_calibration(tmp_path)
    sock, port = _bind_receiver()
    try:
        cfg_path = _write_points_config(tmp_path, calibration_path=cal_path,
                                        udp_port=port)
        orch = Orchestrator(cfg_path)
        assert orch._detector is None and orch._person_detector is None
        orch._points_ingest.start()
        send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # 3 aligned ticks: both cameras see a person at the frame centre.
            for i in range(3):
                ts = 100.0 + i * 0.04
                for cam in ("cam_a", "cam_b"):
                    send_json_datagram(
                        send, orch._points_ingest.address,
                        _det_msg(cam, ts, i).model_dump_json().encode())
                # The ingest thread pairs them; step the pipeline like run() would.
                deadline = time.time() + 2.0
                pair = None
                while pair is None and time.time() < deadline:
                    try:
                        pair = orch._bus.get_latest(timeout=0.1)
                    except Exception:
                        pair = None
                assert pair is not None, "ingest never produced a pair"
                orch.step(pair)

            # Track2D must arrive on the out socket (skim observations etc.).
            while True:
                payload, _ = sock.recvfrom(8192)
                msg = json.loads(payload.decode("utf-8"))
                if msg["type"] == "track_2d":
                    break
            assert msg["cls"] == "person"
            assert orch.source_status["cam_a"] == "alive"
            assert orch.frames_by_camera["cam_a"] == 3
        finally:
            send.close()
            orch._points_ingest.stop()
            orch.publisher.close()
    finally:
        sock.close()


def test_points_mode_refuses_image_snapshots(tmp_path: Path) -> None:
    cal_path = _write_calibration(tmp_path)
    cfg_path = _write_points_config(tmp_path, calibration_path=cal_path,
                                    udp_port=50000)
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["metadata"]["images"] = {"enabled": True, "out_dir": str(tmp_path)}
    cfg_path.write_text(yaml.safe_dump(cfg))
    try:
        Orchestrator(cfg_path)
        raise AssertionError("points mode must refuse metadata.images")
    except ValueError as exc:
        assert "images" in str(exc)


# ---- THE differential test: frames mode vs points mode, same detections ----


def test_differential_frames_vs_points_identical_tracks(tmp_path: Path) -> None:
    """Same detections through both ingest modes ⇒ identical Track2D streams.

    Frames mode runs the stub-ONNX detector; every raw detection is recorded
    at the detector boundary, replayed into a points-mode orchestrator as
    ``DetectionSet``s with the same capture timestamps, and the two published
    track streams must match in class, position, and identity."""
    cal_path = _write_calibration(tmp_path)
    onnx_path = _write_stub_onnx(tmp_path)

    # --- frames mode, recording the detector boundary ---
    sock_f, port_f = _bind_receiver()
    cfg_f = _write_config(tmp_path, calibration_path=cal_path,
                          onnx_path=onnx_path, udp_port=port_f)
    orch_f = Orchestrator(cfg_f)
    recorded: list[dict] = []
    real_detect = orch_f._detector.detect

    def recording_detect(pair):
        out = real_detect(pair)
        recorded.append({
            cid: [(d.cls, float(d.confidence), tuple(d.bbox_xyxy),
                   tuple(d.foot_uv)) for d in dets]
            for cid, dets in out.items()})
        return out

    orch_f._detector.detect = recording_detect

    tracks_frames: list[tuple] = []
    ticks = [100.0 + i * 0.04 for i in range(5)]
    for ts in ticks:
        pair = _make_frame_pair(orch_f.rig, capture_ts=ts)
        t2, _ = orch_f.step(pair)
        tracks_frames.extend((round(ts, 4), t.track_id, t.cls,
                              round(t.xy_m[0], 6), round(t.xy_m[1], 6))
                             for t in t2)
    orch_f.publisher.close()
    sock_f.close()
    assert tracks_frames, "frames mode produced no tracks — stub broken?"

    # --- points mode, replaying the recorded detections ---
    sock_p, port_p = _bind_receiver()
    cfg_p = _write_points_config(tmp_path, calibration_path=cal_path,
                                 udp_port=port_p)
    orch_p = Orchestrator(cfg_p)
    tracks_points: list[tuple] = []
    for i, ts in enumerate(ticks):
        pair = None
        for cam_id, dets in recorded[i].items():
            msg = DetectionSetMessage(
                ts=ts, camera_id=cam_id, frame_wh=(1000, 1000), seq=i,
                dets=tuple(WireDetection(cls=c, confidence=conf,
                                         bbox_xyxy=bbox, foot_uv=foot)
                           for c, conf, bbox, foot in dets))
            candidate = orch_p._sync.submit(detection_set_from_message(msg))
            if candidate is not None:
                pair = candidate
        assert pair is not None, f"tick {i}: synchronizer emitted no pair"
        t2, _ = orch_p.step(pair)
        tracks_points.extend((round(ts, 4), t.track_id, t.cls,
                              round(t.xy_m[0], 6), round(t.xy_m[1], 6))
                             for t in t2)
    orch_p.publisher.close()
    sock_p.close()

    assert tracks_points == tracks_frames
