"""``Orchestrator`` — wires every sub-module from YAML.

The end-to-end test uses:
  * a synthetic 2-camera ``calibration.json`` (same look-down rig from earlier tests),
  * a stub YOLO ONNX (one strong "person" anchor — same pattern as ``test_yolo_onnx.py``),
  * a ``ReplayFrameSource`` in-memory frames (no real RTSP, no real cameras),
  * a UDP sink to a loopback socket so we can assert published envelopes.

This proves the orchestrator builds the entire S2-S6 pipeline correctly from
``backbone.yaml`` and that ``step()`` produces both ``Track2D`` and (with a
matching subscription) ``Track3D`` on a real OS socket.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path

import numpy as np
import onnx
import yaml
from onnx import TensorProto, helper, numpy_helper

from backbone.core.types import Frame, FramePair
from backbone.runtime import Orchestrator
from backbone.shared.geometry import (
    floor_homography_from_K_R_t,
    projection_from_K_R_t,
)

K = np.array([[1000.0, 0.0, 500.0], [0.0, 1000.0, 500.0], [0.0, 0.0, 1.0]])
R_LOOK_DOWN = np.diag([1.0, -1.0, -1.0])
CLASS_NAMES = ["person", "forklift", "pallet"]
NC = len(CLASS_NAMES)


def _camera_cal_dict(camera_id: str, xy: tuple[float, float]) -> dict:
    t = np.array([xy[0], xy[1], 3.0])
    return {
        "camera_id": camera_id,
        "image_size_wh": [1000, 1000],
        "K": K.tolist(),
        "D": [0.0, 0.0, 0.0, 0.0, 0.0],
        "R": R_LOOK_DOWN.tolist(),
        "t": t.tolist(),
        "H": floor_homography_from_K_R_t(K, R_LOOK_DOWN, t).tolist(),
        "P": projection_from_K_R_t(K, R_LOOK_DOWN, t).tolist(),
        "reprojection_rms_px": 0.1,
    }


def _write_calibration(tmp_path: Path) -> Path:
    payload = {
        "version": 1,
        "created_at": "2026-05-18T00:00:00Z",
        "floor_anchor_method": "synthetic",
        "floor_origin_note": "test",
        "cameras": {
            "cam_a": _camera_cal_dict("cam_a", (0.0, 0.0)),
            "cam_b": _camera_cal_dict("cam_b", (2.0, 0.0)),
        },
    }
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(payload))
    return path


def _write_stub_onnx(tmp_path: Path) -> Path:
    """Batch=2 stub: cam_a + cam_b each see a person whose foot back-projects to world (1, 0, 0).

    For a top-down rig with cam_a at (0,0,3) and cam_b at (2,0,3), focal 1000,
    principal (500, 500), the world point (1, 0, 0) projects to:
        cam_a (batch 0): u = 833.33, v = 500
        cam_b (batch 1): u = 166.67, v = 500
    Both back-project to world (1, 0), so cross-cam fusion succeeds and the
    triangulation subscription path can run.

    Note: the model ignores its input and returns a fixed (2, 7, 64) tensor.
    Because the orchestrator's letterbox preserves aspect ratio for a
    1920x1080 source, the pixel coordinates here are in the 640x640 model
    frame — we want them at the model-frame coordinates that letterbox-inverse
    back to source pixels matching the 1080p projection. We instead just
    place the anchors at the 640x640 frame's known back-projection of world
    (1, 0, 0): the letterbox scale=640/1920=1/3 and pad_y=140 for 1080p.
    """
    num_anchors = 64

    # Source-frame pixels (1920x1080):
    #   cam_a: (833.33 * 1920/1000, 500 * 1920/1000) but cameras output 1000x1000
    # The orchestrator test feeds 1080p frames; the FootProjector reads the
    # ORIGINAL source pixel from Detection.foot_uv. We need the model's
    # 640-frame anchor to letterbox-invert to a source-frame foot pixel that
    # projects to world (1, 0) via the calibration's H matrix (which was
    # built for a 1000x1000 image_size, but H is image-size-invariant in
    # this synthetic case because K is the same).
    #
    # Simpler path: pick anchor positions in the 640 model frame that, after
    # letterbox-inverse to 1920x1080, give a foot pixel of (833.33, 500) for
    # cam_a and (166.67, 500) for cam_b in the 1000x1000 calibration frame.
    # Since we use a 1080p source (1920x1080) but the calibration assumes
    # 1000x1000, we'd need a calibration that matches the source size.
    # Cleaner fix: use 1000x1000 source images in the orchestrator test
    # (matches the calibration). The orchestrator test's _make_frame_pair
    # already uses zero images; we just change their shape.
    out_value = np.zeros((2, 4 + NC, num_anchors), dtype=np.float32)
    # Each camera's anchor at the model-frame pixel that letterbox-inverts
    # to its 1000x1000 source-frame projection of world (1, 0, 0).
    # 1000x1000 source → letterbox scale = 640/1000 = 0.64, no padding.
    # cam_a source pixel (833.33, 500) → model frame (533.33, 320).
    # cam_b source pixel (166.67, 500) → model frame (106.67, 320).
    out_value[0, 0, 0] = 833.33 * 0.64   # cam_a cx in model frame
    out_value[0, 1, 0] = 500.0 * 0.64    # cam_a cy in model frame
    out_value[0, 2, 0] = 50.0            # w
    out_value[0, 3, 0] = 100.0           # h
    out_value[0, 4, 0] = 0.95            # person confidence
    out_value[1, 0, 0] = 166.67 * 0.64
    out_value[1, 1, 0] = 500.0 * 0.64
    out_value[1, 2, 0] = 50.0
    out_value[1, 3, 0] = 100.0
    out_value[1, 4, 0] = 0.95

    input_tv = helper.make_tensor_value_info("images", TensorProto.FLOAT, ["N", 3, 640, 640])
    output_tv = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [2, 4 + NC, num_anchors]
    )
    const = numpy_helper.from_array(out_value, name="const_out")
    node = helper.make_node("Constant", inputs=[], outputs=["output"], value=const)
    graph = helper.make_graph([node], "stub", [input_tv], [output_tv])
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=10,
    )
    path = tmp_path / "stub.onnx"
    onnx.save(model, str(path))
    return path


def _write_config(
    tmp_path: Path,
    *,
    calibration_path: Path,
    onnx_path: Path,
    udp_port: int,
    subscriptions_path: Path | None = None,
) -> Path:
    config = {
        "calibration_path": str(calibration_path),
        "cameras": {
            "cam_a": {"source": {"name": "replay", "frames": []}},
            "cam_b": {"source": {"name": "replay", "frames": []}},
        },
        "ingestion": {
            "frame_sync": {"max_skew_ms": 33.0, "max_age_ms": 1000.0, "buffer_size": 8},
            "frame_bus": {"default_maxsize": 8},
        },
        "detection": {
            "plugin": "yolo_onnx",
            "onnx_path": str(onnx_path),
            "class_names": CLASS_NAMES,
            "providers": ["CPUExecutionProvider"],
            "confidence_threshold": 0.25,
        },
        "homography": {
            "tracker": {"plugin": "bytetrack"},
            "track_config": {"min_hits_to_confirm": 1, "max_lost_frames": 30},
        },
        "metadata": {
            "sinks": [{"plugin": "udp", "host": "127.0.0.1", "port": udp_port}],
        },
    }
    if subscriptions_path is not None:
        config["subscriptions_path"] = str(subscriptions_path)
    path = tmp_path / "backbone.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _bind_receiver() -> tuple[socket.socket, int]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    sock.settimeout(2.0)
    return sock, sock.getsockname()[1]


def _make_frame_pair(orch_rig, capture_ts: float) -> FramePair:
    """Build a FramePair of 1000x1000 BGR frames matching the calibration image_size.

    The stub ONNX ignores image content; we just need shape compatibility.
    """
    img = np.zeros((1000, 1000, 3), dtype=np.uint8)
    frames = {
        cam_id: Frame(camera_id=cam_id, capture_ts=capture_ts, frame_idx=0, image=img)
        for cam_id in orch_rig.camera_ids
    }
    return FramePair(capture_ts=capture_ts, frame_idx=0, frames=frames)


# ---- tests ----


def test_orchestrator_builds_from_yaml(tmp_path: Path) -> None:
    cal_path = _write_calibration(tmp_path)
    onnx_path = _write_stub_onnx(tmp_path)
    sock, port = _bind_receiver()
    try:
        cfg_path = _write_config(tmp_path, calibration_path=cal_path,
                                 onnx_path=onnx_path, udp_port=port)
        orch = Orchestrator(cfg_path)
        # Components present and wired:
        assert set(orch.rig.camera_ids) == {"cam_a", "cam_b"}
        assert len(orch.publisher.sinks) == 1
        assert orch.latency_meter.name == "capture_to_publish"
        orch.publisher.close()
    finally:
        sock.close()


def test_orchestrator_step_emits_track_2d_over_udp(tmp_path: Path) -> None:
    cal_path = _write_calibration(tmp_path)
    onnx_path = _write_stub_onnx(tmp_path)
    sock, port = _bind_receiver()
    try:
        cfg_path = _write_config(tmp_path, calibration_path=cal_path,
                                 onnx_path=onnx_path, udp_port=port)
        orch = Orchestrator(cfg_path)
        # Drive a few synthetic frames through step() directly.
        tracks_2d_emitted: list = []
        for i in range(3):
            pair = _make_frame_pair(orch.rig, capture_ts=i * 0.033)
            t2, _ = orch.step(pair)
            tracks_2d_emitted.extend(t2)

        # At least one Track2D should have made it to the UDP sink.
        payload, _ = sock.recvfrom(8192)
        msg = json.loads(payload.decode("utf-8"))
        assert msg["type"] == "track_2d"
        assert msg["cls"] == "person"
        # The 2-camera fusion produced an averaged position; verify it's plausible.
        assert -5.0 <= msg["xy_m"][0] <= 5.0
        assert orch.frame_count == 3
        orch.publisher.close()
    finally:
        sock.close()


def test_orchestrator_emits_track_3d_when_subscription_matches(tmp_path: Path) -> None:
    cal_path = _write_calibration(tmp_path)
    onnx_path = _write_stub_onnx(tmp_path)
    sock, port = _bind_receiver()
    try:
        # Subscription with no zone gate, matching all 2-cam persons.
        subs_path = tmp_path / "subs.yaml"
        subs_path.write_text(yaml.safe_dump([{
            "name": "all_persons",
            "module": "test",
            "match": {"cls": "person", "cameras_seeing_min": 2},
            "request": "xyz",
        }]))
        cfg_path = _write_config(tmp_path, calibration_path=cal_path,
                                 onnx_path=onnx_path, udp_port=port,
                                 subscriptions_path=subs_path)
        orch = Orchestrator(cfg_path)
        for i in range(3):
            pair = _make_frame_pair(orch.rig, capture_ts=i * 0.033)
            orch.step(pair)

        # Drain whatever the sink emitted, look for at least one track_3d.
        saw_2d = False
        saw_3d = False
        for _ in range(64):
            try:
                payload, _ = sock.recvfrom(8192)
            except TimeoutError:
                break
            msg = json.loads(payload.decode("utf-8"))
            if msg["type"] == "track_2d":
                saw_2d = True
            elif msg["type"] == "track_3d":
                saw_3d = True
                # 3D track inherits the 2D track_id.
                assert msg["track_id"] >= 1
                assert "max_reprojection_error_px" in msg
        assert saw_2d
        assert saw_3d
        orch.publisher.close()
    finally:
        sock.close()


def test_orchestrator_rejects_config_without_sinks(tmp_path: Path) -> None:
    """A backbone.yaml without metadata.sinks must refuse to start."""
    cal_path = _write_calibration(tmp_path)
    onnx_path = _write_stub_onnx(tmp_path)
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "calibration_path": str(cal_path),
        "cameras": {
            "cam_a": {"source": {"name": "replay", "frames": []}},
            "cam_b": {"source": {"name": "replay", "frames": []}},
        },
        "detection": {"plugin": "yolo_onnx", "onnx_path": str(onnx_path),
                      "class_names": CLASS_NAMES, "providers": ["CPUExecutionProvider"]},
        "metadata": {"sinks": []},
    }))
    import pytest

    with pytest.raises(ValueError, match="at least one sink"):
        Orchestrator(cfg_path)


# ---- Mode 1 / Mode 2 / runtime degradation ----


def _write_single_cam_calibration(tmp_path: Path) -> Path:
    """Mode 1 calibration with a fitted 4-point homography for cam_a only."""
    from calibration.calibrate_single_cam import PointPair, build_single_camera_calibration

    pairs = [
        PointPair(pixel_uv=(100.0, 100.0), world_xy_m=(-4.0, -4.0)),
        PointPair(pixel_uv=(900.0, 100.0), world_xy_m=(4.0, -4.0)),
        PointPair(pixel_uv=(900.0, 700.0), world_xy_m=(4.0, 2.0)),
        PointPair(pixel_uv=(100.0, 700.0), world_xy_m=(-4.0, 2.0)),
    ]
    cal = build_single_camera_calibration(
        camera_id="cam_a", image_size_wh=(1000, 1000), pairs=pairs,
    )
    path = tmp_path / "calibration_mode1.json"
    path.write_text(cal.to_json())
    return path


def _write_stub_onnx_for_single_cam(tmp_path: Path) -> Path:
    """Batch=1 stub ONNX: one strong 'person' anchor at the model-frame centre."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    num_anchors = 64
    out_value = np.zeros((1, 4 + NC, num_anchors), dtype=np.float32)
    out_value[0, 0, 0] = 500.0 * 0.64    # cx in 640 frame for source 1000x1000
    out_value[0, 1, 0] = 500.0 * 0.64
    out_value[0, 2, 0] = 50.0
    out_value[0, 3, 0] = 100.0
    out_value[0, 4, 0] = 0.95

    input_tv = helper.make_tensor_value_info("images", TensorProto.FLOAT, ["N", 3, 640, 640])
    output_tv = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 4 + NC, num_anchors]
    )
    const = numpy_helper.from_array(out_value, name="const_out")
    node = helper.make_node("Constant", inputs=[], outputs=["output"], value=const)
    graph = helper.make_graph([node], "stub1", [input_tv], [output_tv])
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10,
    )
    path = tmp_path / "stub_single.onnx"
    onnx.save(model, str(path))
    return path


def test_mode1_build_skips_triangulator(tmp_path: Path) -> None:
    """Single-camera config ⇒ orchestrator does NOT instantiate the triangulation stack."""
    cal_path = _write_single_cam_calibration(tmp_path)
    onnx_path = _write_stub_onnx_for_single_cam(tmp_path)
    sock, port = _bind_receiver()
    try:
        cfg_path = tmp_path / "mode1.yaml"
        cfg_path.write_text(yaml.safe_dump({
            "calibration_path": str(cal_path),
            "cameras": {
                "cam_a": {"source": {"name": "replay", "frames": []}},
            },
            "detection": {
                "plugin": "yolo_onnx", "onnx_path": str(onnx_path),
                "class_names": CLASS_NAMES, "providers": ["CPUExecutionProvider"],
            },
            "homography": {
                "tracker": {"plugin": "bytetrack"},
                "track_config": {"min_hits_to_confirm": 1, "max_lost_frames": 30},
            },
            "metadata": {"sinks": [{"plugin": "udp", "host": "127.0.0.1", "port": port}]},
        }))
        orch = Orchestrator(cfg_path)
        assert orch.mode == "single_cam_homography"
        # Triangulation stack stays None.
        assert orch._triangulator is None
        assert orch._associator is None
        assert orch._reproj_gate is None
        assert orch._tracker_3d is None
        # source_status starts alive.
        assert orch.source_status == {"cam_a": "alive"}
        orch.publisher.close()
    finally:
        sock.close()


def test_mode1_emits_track2d_only_no_track3d(tmp_path: Path) -> None:
    """Mode 1 + single-cam FramePair → Track2D on UDP; no Track3D ever."""
    cal_path = _write_single_cam_calibration(tmp_path)
    onnx_path = _write_stub_onnx_for_single_cam(tmp_path)
    sock, port = _bind_receiver()
    try:
        cfg_path = tmp_path / "mode1.yaml"
        cfg_path.write_text(yaml.safe_dump({
            "calibration_path": str(cal_path),
            "cameras": {
                "cam_a": {"source": {"name": "replay", "frames": []}},
            },
            "detection": {
                "plugin": "yolo_onnx", "onnx_path": str(onnx_path),
                "class_names": CLASS_NAMES, "providers": ["CPUExecutionProvider"],
            },
            "homography": {
                "tracker": {"plugin": "bytetrack"},
                "track_config": {"min_hits_to_confirm": 1, "max_lost_frames": 30},
            },
            "metadata": {"sinks": [{"plugin": "udp", "host": "127.0.0.1", "port": port}]},
        }))
        orch = Orchestrator(cfg_path)
        # Build a solo FramePair (one camera, one frame).
        img = np.zeros((1000, 1000, 3), dtype=np.uint8)
        for i in range(3):
            ts = i * 0.033
            frame = Frame(camera_id="cam_a", capture_ts=ts, frame_idx=i, image=img)
            pair = FramePair(capture_ts=ts, frame_idx=i, frames={"cam_a": frame})
            _t2, t3 = orch.step(pair)
            assert t3 == []   # never any Track3D in Mode 1
        # At least one Track2D should have made it to UDP.
        payload, _ = sock.recvfrom(8192)
        msg = json.loads(payload.decode("utf-8"))
        assert msg["type"] == "track_2d"
        assert msg["cameras_seeing"] == ["cam_a"]
        orch.publisher.close()
    finally:
        sock.close()


def test_mode2_ingestion_loop_failure_does_not_set_stop_event(tmp_path: Path) -> None:
    """A per-source crash must NOT propagate to the global stop_event."""
    cal_path = _write_calibration(tmp_path)
    onnx_path = _write_stub_onnx(tmp_path)
    sock, port = _bind_receiver()
    try:
        cfg_path = _write_config(tmp_path, calibration_path=cal_path,
                                 onnx_path=onnx_path, udp_port=port)
        orch = Orchestrator(cfg_path)

        class _CrashingSource:
            camera_id = "cam_b"
            def frames(self):
                raise RuntimeError("simulated network drop")
            def start(self):
                pass
            def stop(self):
                pass

        orch._ingestion_loop(_CrashingSource())   # runs synchronously in this test
        assert orch.source_status["cam_b"] == "crashed"
        assert not orch.stop_event.is_set()
        orch.publisher.close()
    finally:
        sock.close()


def _write_stub_object_onnx_single(tmp_path: Path, cls_idx: int) -> Path:
    """Batch=1 detect stub: one strong anchor of class `cls_idx` at frame centre."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    num_anchors = 64
    out = np.zeros((1, 4 + NC, num_anchors), dtype=np.float32)
    out[0, 0, 0] = 500.0 * 0.64
    out[0, 1, 0] = 500.0 * 0.64
    out[0, 2, 0] = 50.0
    out[0, 3, 0] = 100.0
    out[0, 4 + cls_idx, 0] = 0.95
    input_tv = helper.make_tensor_value_info("images", TensorProto.FLOAT, ["N", 3, 640, 640])
    output_tv = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4 + NC, num_anchors])
    const = numpy_helper.from_array(out, name="const_obj")
    node = helper.make_node("Constant", inputs=[], outputs=["output"], value=const)
    graph = helper.make_graph([node], "stubobj", [input_tv], [output_tv])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
    path = tmp_path / f"stub_obj_{cls_idx}.onnx"
    onnx.save(model, str(path))
    return path


def _write_stub_pose_onnx_single(tmp_path: Path) -> Path:
    """Batch=1 pose stub: one 'person' anchor (4 + 1 + 17*3 channels) with ankles."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    nkp = 17
    chan = 4 + 1 + nkp * 3
    num_anchors = 64
    out = np.zeros((1, chan, num_anchors), dtype=np.float32)
    cx = cy = 500.0 * 0.64
    out[0, 0, 0], out[0, 1, 0], out[0, 2, 0], out[0, 3, 0] = cx, cy, 40.0, 120.0
    out[0, 4, 0] = 0.95                       # person score
    for kp in (15, 16):                       # ankles near the foot, high conf
        base = 5 + kp * 3
        out[0, base, 0], out[0, base + 1, 0], out[0, base + 2, 0] = cx, cy + 50.0, 0.9
    input_tv = helper.make_tensor_value_info("images", TensorProto.FLOAT, ["N", 3, 640, 640])
    output_tv = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, chan, num_anchors])
    const = numpy_helper.from_array(out, name="const_pose")
    node = helper.make_node("Constant", inputs=[], outputs=["output"], value=const)
    graph = helper.make_graph([node], "stubpose", [input_tv], [output_tv])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
    path = tmp_path / "stub_pose.onnx"
    onnx.save(model, str(path))
    return path


def test_mode1_pose_detector_emits_person_and_pallet(tmp_path: Path) -> None:
    """A configured `detection.pose_onnx_path` runs the pose detector alongside the
    object detector → both person AND pallet Track2D are emitted (Phase 2)."""
    cal_path = _write_single_cam_calibration(tmp_path)
    obj_onnx = _write_stub_object_onnx_single(tmp_path, cls_idx=2)   # pallet
    pose_onnx = _write_stub_pose_onnx_single(tmp_path)               # person
    sock, port = _bind_receiver()
    try:
        cfg_path = tmp_path / "mode1_pose.yaml"
        cfg_path.write_text(yaml.safe_dump({
            "calibration_path": str(cal_path),
            "cameras": {"cam_a": {"source": {"name": "replay", "frames": []}}},
            "detection": {
                "plugin": "yolo_onnx", "onnx_path": str(obj_onnx),
                "class_names": CLASS_NAMES, "providers": ["CPUExecutionProvider"],
                "pose_onnx_path": str(pose_onnx), "pose_confidence_threshold": 0.25,
            },
            "homography": {
                "tracker": {"plugin": "bytetrack"},
                "track_config": {"min_hits_to_confirm": 1, "max_lost_frames": 30},
            },
            "metadata": {"sinks": [{"plugin": "udp", "host": "127.0.0.1", "port": port}]},
        }))
        orch = Orchestrator(cfg_path)
        assert orch._person_detector is not None            # pose detector wired
        img = np.zeros((1000, 1000, 3), dtype=np.uint8)
        classes = set()
        for i in range(5):
            ts = i * 0.033
            frame = Frame(camera_id="cam_a", capture_ts=ts, frame_idx=i, image=img)
            pair = FramePair(capture_ts=ts, frame_idx=i, frames={"cam_a": frame})
            t2, _ = orch.step(pair)
            classes.update(t.cls for t in t2)
        assert "person" in classes                          # from the pose detector
        assert "pallet" in classes                          # from the object detector
        orch.publisher.close()
    finally:
        sock.close()


_OCC_CLASSES = ["pallet", "carton", "polybag"]


def _write_stub_pallet_carton_onnx(tmp_path: Path) -> Path:
    """Batch=1 detect stub with TWO anchors in source coords: a pallet at
    x[400,600] y[600,720] and a carton resting on it at x[430,570] y[500,600]
    (carton base = pallet top, horizontally aligned → A associates them).
    Values are in the 640 target frame (source x0.64, since 1000->640 letterbox)."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    s = 0.64
    out = np.zeros((1, 4 + 3, 64), dtype=np.float32)
    # anchor 0 = pallet (cls 0): source bbox (400,600,600,720)
    out[0, 0, 0], out[0, 1, 0] = 500 * s, 660 * s          # cx, cy
    out[0, 2, 0], out[0, 3, 0] = 200 * s, 120 * s          # w, h
    out[0, 4, 0] = 0.95
    # anchor 1 = carton (cls 1): source bbox (430,500,570,600)
    out[0, 0, 1], out[0, 1, 1] = 500 * s, 550 * s
    out[0, 2, 1], out[0, 3, 1] = 140 * s, 100 * s
    out[0, 5, 1] = 0.90
    input_tv = helper.make_tensor_value_info("images", TensorProto.FLOAT, ["N", 3, 640, 640])
    output_tv = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 7, 64])
    const = numpy_helper.from_array(out, name="const_pc")
    node = helper.make_node("Constant", inputs=[], outputs=["output"], value=const)
    graph = helper.make_graph([node], "stubpc", [input_tv], [output_tv])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
    path = tmp_path / "stub_pc.onnx"
    onnx.save(model, str(path))
    return path


def test_mode1_pallet_occupancy_full_carton(tmp_path: Path) -> None:
    """A carton resting on a pallet → that pallet's Track2D publishes
    occupancy_state='full', content='carton' (the empty/full KPI)."""
    cal_path = _write_single_cam_calibration(tmp_path)
    onnx_path = _write_stub_pallet_carton_onnx(tmp_path)
    sock, port = _bind_receiver()
    try:
        cfg_path = tmp_path / "mode1_occ.yaml"
        cfg_path.write_text(yaml.safe_dump({
            "calibration_path": str(cal_path),
            "cameras": {"cam_a": {"source": {"name": "replay", "frames": []}}},
            "detection": {
                "plugin": "yolo_onnx", "onnx_path": str(onnx_path),
                "class_names": _OCC_CLASSES, "providers": ["CPUExecutionProvider"],
                "confidence_threshold": 0.25,
            },
            "homography": {
                "tracker": {"plugin": "bytetrack"},
                "track_config": {"min_hits_to_confirm": 1, "max_lost_frames": 30},
            },
            "metadata": {"sinks": [{"plugin": "udp", "host": "127.0.0.1", "port": port}]},
        }))
        orch = Orchestrator(cfg_path)
        img = np.zeros((1000, 1000, 3), dtype=np.uint8)
        pallet_state = None
        for i in range(4):
            ts = i * 0.033
            frame = Frame(camera_id="cam_a", capture_ts=ts, frame_idx=i, image=img)
            pair = FramePair(capture_ts=ts, frame_idx=i, frames={"cam_a": frame})
            t2, _ = orch.step(pair)
            for t in t2:
                if t.cls == "pallet":
                    pallet_state = (t.occupancy_state, t.occupancy_content)
        assert pallet_state == ("full", "carton"), pallet_state
        orch.publisher.close()
    finally:
        sock.close()
