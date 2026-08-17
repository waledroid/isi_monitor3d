"""``Track2DMessage`` / ``Track3DMessage`` — pydantic round-trip + version gate."""

from __future__ import annotations

import json

import numpy as np
import pytest
from pydantic import ValidationError

from backbone.comms.schemas import (
    SCHEMA_VERSION,
    CalibrationFactCheck,
    ConfigMessage,
    DiagnosticsMessage,
    EtagereCellState,
    EtagereStateMessage,
    LatencyStats,
    MessageType,
    PassingEventMessage,
    SchemaVersionError,
    Track2DMessage,
    Track3DMessage,
    ZoneSpec,
    ZoneStateMessage,
    parse_envelope,
)
from backbone.core.types import Track2D, Track3D


def _track_2d() -> Track2D:
    return Track2D(
        track_id=47,
        cls="person",
        capture_ts=1731423891.234,
        xy_m=(3.21, 1.84),
        vxy_m=(0.45, -0.10),
        confidence=0.93,
        cameras_seeing=("cam_a", "cam_b"),
    )


def _track_3d() -> Track3D:
    return Track3D(
        track_id=47,
        cls="person",
        capture_ts=1731423891.234,
        xyz_m=(3.21, 1.84, 0.0),
        vxyz_m=(0.45, -0.10, 0.0),
        contributing_cameras=("cam_a", "cam_b"),
        max_reprojection_error_px=2.7,
        keypoints_xyz=None,
    )


def test_track2d_from_internal() -> None:
    msg = Track2DMessage.from_track(_track_2d())
    assert msg.type == MessageType.TRACK_2D
    assert msg.schema_version == SCHEMA_VERSION
    assert msg.track_id == 47
    assert msg.cls == "person"
    assert msg.xy_m == (3.21, 1.84)
    assert msg.vxy_m == (0.45, -0.10)
    assert msg.confidence == 0.93
    assert msg.cameras_seeing == ("cam_a", "cam_b")


def test_track3d_from_internal_no_keypoints() -> None:
    msg = Track3DMessage.from_track(_track_3d())
    assert msg.type == MessageType.TRACK_3D
    assert msg.track_id == 47
    assert msg.xyz_m == (3.21, 1.84, 0.0)
    assert msg.max_reprojection_error_px == pytest.approx(2.7)
    assert msg.keypoints_xyz is None


def test_track3d_from_internal_with_keypoints() -> None:
    track = _track_3d()
    track_with_kp = Track3D(
        track_id=track.track_id,
        cls=track.cls,
        capture_ts=track.capture_ts,
        xyz_m=track.xyz_m,
        vxyz_m=track.vxyz_m,
        contributing_cameras=track.contributing_cameras,
        max_reprojection_error_px=track.max_reprojection_error_px,
        keypoints_xyz=np.array([[1.0, 2.0, 0.0], [1.1, 2.1, 0.5]]),
    )
    msg = Track3DMessage.from_track(track_with_kp)
    assert msg.keypoints_xyz is not None
    assert len(msg.keypoints_xyz) == 2
    assert msg.keypoints_xyz[0] == (1.0, 2.0, 0.0)
    assert msg.keypoints_xyz[1] == (1.1, 2.1, 0.5)


def test_track2d_json_roundtrip() -> None:
    msg = Track2DMessage.from_track(_track_2d())
    payload = msg.model_dump_json()
    data = json.loads(payload)
    parsed = Track2DMessage.model_validate(data)
    assert parsed == msg


def test_track3d_json_roundtrip() -> None:
    msg = Track3DMessage.from_track(_track_3d())
    parsed = Track3DMessage.model_validate_json(msg.model_dump_json())
    assert parsed == msg


def test_parse_envelope_discriminates_by_type() -> None:
    t2 = Track2DMessage.from_track(_track_2d())
    t3 = Track3DMessage.from_track(_track_3d())
    parsed_2 = parse_envelope(json.loads(t2.model_dump_json()))
    parsed_3 = parse_envelope(json.loads(t3.model_dump_json()))
    assert isinstance(parsed_2, Track2DMessage)
    assert isinstance(parsed_3, Track3DMessage)


def test_parse_envelope_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="unknown message type"):
        parse_envelope({"schema_version": SCHEMA_VERSION, "type": "nope"})


def test_parse_envelope_rejects_wrong_version() -> None:
    payload = json.loads(Track2DMessage.from_track(_track_2d()).model_dump_json())
    payload["schema_version"] = 0
    with pytest.raises(SchemaVersionError):
        parse_envelope(payload)


def test_parse_envelope_accepts_version_3() -> None:
    """Version 3 messages (pre-Phase-B) are still accepted by parse_envelope."""
    payload = json.loads(Track2DMessage.from_track(_track_2d()).model_dump_json())
    payload["schema_version"] = 3
    # Should not raise — parse_envelope accepts both v3 and v4.
    msg = parse_envelope(payload)
    assert isinstance(msg, Track2DMessage)


def test_parse_envelope_accepts_current_version() -> None:
    """Messages at the current SCHEMA_VERSION are accepted."""
    payload = json.loads(Track2DMessage.from_track(_track_2d()).model_dump_json())
    assert payload["schema_version"] == SCHEMA_VERSION == 6
    msg = parse_envelope(payload)
    assert isinstance(msg, Track2DMessage)


def test_passing_event_message_fields() -> None:
    """PassingEventMessage has correct type and required fields."""
    msg = PassingEventMessage(
        ts=99.0, track_id=5, cls="palette", zone="B3D", direction="enter",
    )
    assert msg.type == MessageType.PASSING
    assert msg.schema_version == SCHEMA_VERSION
    assert msg.track_id == 5
    assert msg.cls == "palette"
    assert msg.zone == "B3D"
    assert msg.direction == "enter"
    assert msg.ts == pytest.approx(99.0)


def test_passing_event_message_json_roundtrip() -> None:
    """PassingEventMessage serialises and deserialises cleanly."""
    msg = PassingEventMessage(
        ts=100.5, track_id=7, cls="person", zone="RACK_A", direction="leave",
    )
    data = json.loads(msg.model_dump_json())
    back = PassingEventMessage.model_validate(data)
    assert back == msg


def test_passing_event_message_from_event() -> None:
    """``PassingEventMessage.from_event`` accepts any duck-typed event object."""
    from backbone.shared.zone_transitions import PassingEvent

    ev = PassingEvent(track_id=3, cls="forklift", zone="DANGER", direction="enter", ts=10.0)
    msg = PassingEventMessage.from_event(ev)
    assert msg.track_id == 3
    assert msg.cls == "forklift"
    assert msg.zone == "DANGER"
    assert msg.direction == "enter"
    assert msg.ts == pytest.approx(10.0)


def test_parse_envelope_dispatches_passing_type() -> None:
    """parse_envelope returns a PassingEventMessage for 'passing' type."""
    msg = PassingEventMessage(
        ts=5.0, track_id=1, cls="person", zone="B3D", direction="leave",
    )
    data = json.loads(msg.model_dump_json())
    parsed = parse_envelope(data)
    assert isinstance(parsed, PassingEventMessage)
    assert parsed.direction == "leave"


def test_passing_direction_rejects_invalid_value() -> None:
    """'direction' must be 'enter' or 'leave' — pydantic rejects anything else."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        PassingEventMessage(
            ts=1.0, track_id=1, cls="person", zone="B3D", direction="sideways",
        )


def test_schema_version_is_6() -> None:
    """Pin the current schema version so a bump is explicit and visible.
    v5 added ProximityMessage; v6 added ObservationsMessage (per-camera raw
    detections for display consumers)."""
    assert SCHEMA_VERSION == 6


def test_topic_version_is_v1() -> None:
    """Pin the current MQTT topic-contract version (shared by node + gateway)."""
    from backbone.comms.schemas import TOPIC_VERSION

    assert TOPIC_VERSION == "v1"


def test_extra_fields_rejected() -> None:
    payload = json.loads(Track2DMessage.from_track(_track_2d()).model_dump_json())
    payload["unknown_field"] = "surprise"
    with pytest.raises(ValidationError):
        Track2DMessage.model_validate(payload)


def test_confidence_out_of_range_rejected() -> None:
    with pytest.raises(ValidationError):
        Track2DMessage(
            ts=0.0, track_id=1, cls="person",
            xy_m=(0.0, 0.0), vxy_m=(0.0, 0.0),
            confidence=2.0,   # > 1.0
            cameras_seeing=("cam_a",),
        )


def test_negative_track_id_rejected() -> None:
    with pytest.raises(ValidationError):
        Track2DMessage(
            ts=0.0, track_id=-1, cls="person",
            xy_m=(0.0, 0.0), vxy_m=(0.0, 0.0),
            confidence=0.5, cameras_seeing=("cam_a",),
        )


def test_messages_are_frozen() -> None:
    msg = Track2DMessage.from_track(_track_2d())
    with pytest.raises(ValidationError):
        msg.track_id = 999  # frozen → mutation rejected


def test_track3d_single_view_roundtrip() -> None:
    t = Track3D(
        track_id=8, cls="person", capture_ts=1.0,
        xyz_m=(1.0, 2.0, 0.0), vxyz_m=(0.0, 0.0, 0.0),
        contributing_cameras=("cam_a",), max_reprojection_error_px=0.0,
        single_view=True, confidence=0.5,
    )
    msg = Track3DMessage.from_track(t)
    assert msg.single_view is True
    assert msg.confidence == pytest.approx(0.5)
    # JSON wire round-trip preserves the flags.
    back = Track3DMessage.model_validate_json(msg.model_dump_json())
    assert back.single_view is True and back.confidence == pytest.approx(0.5)


def test_track3d_defaults_are_two_view() -> None:
    msg = Track3DMessage.from_track(_track_3d())   # no single_view set
    assert msg.single_view is False
    assert msg.confidence == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# DiagnosticsMessage
# ---------------------------------------------------------------------------

def _make_diagnostics() -> DiagnosticsMessage:
    return DiagnosticsMessage(
        ts=1_700_000_000.0,
        node_id="zone_a",
        mode="single_cam_homography",
        sources={"cam_a": "alive"},
        frame_count=42,
        fps=12.3,
        latency_ms=LatencyStats(p50=10.0, p95=18.0, p99=22.0, n=100),
        zones=3,
        subscriptions=0,
        calibration=CalibrationFactCheck(loaded=True, rms_ok=True, mode=1),
    )


def test_diagnostics_message_type_and_version() -> None:
    msg = _make_diagnostics()
    assert msg.type == MessageType.DIAGNOSTICS
    assert msg.schema_version == SCHEMA_VERSION
    assert msg.node_id == "zone_a"
    assert msg.mode == "single_cam_homography"
    assert msg.frame_count == 42
    assert msg.fps == pytest.approx(12.3)
    assert msg.zones == 3
    assert msg.subscriptions == 0
    assert msg.calibration.loaded is True
    assert msg.calibration.rms_ok is True
    assert msg.calibration.mode == 1
    assert msg.latency_ms.p95 == pytest.approx(18.0)


def test_diagnostics_message_json_roundtrip() -> None:
    msg = _make_diagnostics()
    data = json.loads(msg.model_dump_json())
    back = DiagnosticsMessage.model_validate(data)
    assert back == msg


def test_parse_envelope_dispatches_diagnostics() -> None:
    msg = _make_diagnostics()
    data = json.loads(msg.model_dump_json())
    parsed = parse_envelope(data)
    assert isinstance(parsed, DiagnosticsMessage)
    assert parsed.node_id == "zone_a"
    assert parsed.fps == pytest.approx(12.3)


def test_diagnostics_bad_version_raises() -> None:
    data = json.loads(_make_diagnostics().model_dump_json())
    data["schema_version"] = 0
    with pytest.raises(SchemaVersionError):
        parse_envelope(data)


def test_diagnostics_extra_fields_rejected() -> None:
    data = json.loads(_make_diagnostics().model_dump_json())
    data["unexpected_field"] = "nope"
    with pytest.raises(ValidationError):
        DiagnosticsMessage.model_validate(data)


# ---------------------------------------------------------------------------
# ConfigMessage
# ---------------------------------------------------------------------------

def _make_config() -> ConfigMessage:
    return ConfigMessage(
        ts=1_700_000_000.0,
        node_id="zone_a",
        area="Zone A",
        mode="single_cam_homography",
        cameras=["cam_a"],
        zones=[
            ZoneSpec(
                name="rack_b3",
                kind="etagere",
                type="storage",
                severity="info",
                polygon=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            )
        ],
        calibration=CalibrationFactCheck(loaded=True, rms_ok=True, mode=1),
    )


def test_config_message_type_and_version() -> None:
    msg = _make_config()
    assert msg.type == MessageType.CONFIG
    assert msg.schema_version == SCHEMA_VERSION
    assert msg.node_id == "zone_a"
    assert msg.area == "Zone A"
    assert msg.mode == "single_cam_homography"
    assert msg.cameras == ["cam_a"]
    assert len(msg.zones) == 1
    assert msg.zones[0].name == "rack_b3"
    assert msg.zones[0].polygon == [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    assert msg.calibration.mode == 1


def test_config_message_json_roundtrip() -> None:
    msg = _make_config()
    data = json.loads(msg.model_dump_json())
    back = ConfigMessage.model_validate(data)
    assert back == msg


def test_zonespec_z_base_m_defaults_zero() -> None:
    spec = ZoneSpec(
        name="rack_b3", kind="etagere", type="storage", severity="info",
        polygon=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    )
    assert spec.z_base_m == 0.0


def test_zonespec_parses_without_z_base_m_old_advert() -> None:
    """A retained ConfigMessage advert from a pre-z_base_m Backbone (no key
    in the wire payload) must still parse — additive, default-valued."""
    old_advert = {
        "name": "rack_b3", "kind": "etagere", "type": "storage", "severity": "info",
        "polygon": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    }
    spec = ZoneSpec.model_validate(old_advert)
    assert spec.z_base_m == 0.0


def test_zonespec_z_base_m_roundtrips() -> None:
    spec = ZoneSpec(
        name="sortie_machine_1", kind="palette", type="storage", severity="info",
        polygon=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], z_base_m=0.304,
    )
    back = ZoneSpec.model_validate(json.loads(spec.model_dump_json()))
    assert back.z_base_m == pytest.approx(0.304)


def test_parse_envelope_dispatches_config() -> None:
    msg = _make_config()
    data = json.loads(msg.model_dump_json())
    parsed = parse_envelope(data)
    assert isinstance(parsed, ConfigMessage)
    assert parsed.area == "Zone A"
    assert len(parsed.zones) == 1


def test_config_bad_version_raises() -> None:
    data = json.loads(_make_config().model_dump_json())
    data["schema_version"] = 99
    with pytest.raises(SchemaVersionError):
        parse_envelope(data)


def test_config_extra_fields_rejected() -> None:
    data = json.loads(_make_config().model_dump_json())
    data["rogue_key"] = "surprise"
    with pytest.raises(ValidationError):
        ConfigMessage.model_validate(data)


# ---------------------------------------------------------------------------
# ZoneStateMessage — the per-zone object list for FMS/WMS (zone folder topic)
# ---------------------------------------------------------------------------

def _make_zone_state():
    from backbone.comms.schemas import ZoneObject, ZoneStateMessage
    return ZoneStateMessage(
        ts=1_700_000_000.0,
        zone="B3D",
        objects=(
            ZoneObject(
                track_id=7, cls="palette", confidence=0.91, xy_m=(3.0, 4.0),
                occupancy_state="full", occupancy_content="carton",
                occupancy_confidence=0.88,
            ),
            ZoneObject(track_id=9, cls="person", confidence=0.75, xy_m=(3.5, 4.2)),
        ),
        count=2,
    )


def test_zone_state_message_type_and_version() -> None:
    from backbone.comms.schemas import MessageType
    msg = _make_zone_state()
    assert msg.type == MessageType.ZONE_STATE
    assert msg.schema_version == SCHEMA_VERSION
    assert msg.count == 2
    assert msg.objects[0].cls == "palette"


def test_zone_state_json_roundtrip() -> None:
    from backbone.comms.schemas import ZoneStateMessage
    msg = _make_zone_state()
    data = json.loads(msg.model_dump_json())
    back = ZoneStateMessage.model_validate(data)
    assert back == msg


def test_parse_envelope_dispatches_zone_state() -> None:
    from backbone.comms.schemas import ZoneStateMessage
    data = json.loads(_make_zone_state().model_dump_json())
    parsed = parse_envelope(data)
    assert isinstance(parsed, ZoneStateMessage)
    assert parsed.zone == "B3D"
    assert parsed.objects[1].track_id == 9


def test_zone_state_empty_zone_is_explicit() -> None:
    """An empty zone is an explicit empty objects tuple, never an absent message."""
    from backbone.comms.schemas import ZoneStateMessage
    msg = ZoneStateMessage(ts=1.0, zone="B3D", objects=(), count=0)
    data = json.loads(msg.model_dump_json())
    assert data["objects"] == []
    assert data["count"] == 0


def test_zone_state_extra_fields_rejected() -> None:
    from backbone.comms.schemas import ZoneStateMessage
    data = json.loads(_make_zone_state().model_dump_json())
    data["rogue_key"] = "surprise"
    with pytest.raises(ValidationError):
        ZoneStateMessage.model_validate(data)


def test_zone_object_confidence_out_of_range_rejected() -> None:
    from backbone.comms.schemas import ZoneObject
    with pytest.raises(ValidationError):
        ZoneObject(track_id=1, cls="palette", confidence=1.5, xy_m=(0.0, 0.0))


def test_zone_state_bad_version_raises() -> None:
    data = json.loads(_make_zone_state().model_dump_json())
    data["schema_version"] = 99
    with pytest.raises(SchemaVersionError):
        parse_envelope(data)


# ---- ZoneDecisionModel — the PalletStateManager verdict (additive in v6) ----


def test_zone_state_with_decision_round_trip() -> None:
    """New payload: the optional ``decision`` object survives the wire."""
    from backbone.comms.schemas import ZoneDecisionModel, ZoneStateMessage
    msg = ZoneStateMessage(
        **{**_make_zone_state().model_dump(),
           "decision": ZoneDecisionModel(
               palette_state="palette_loaded",
               content=("carton",),
               counts={"palette": 1, "carton": 2},
           )})
    data = json.loads(msg.model_dump_json())
    back = parse_envelope(data)
    assert isinstance(back, ZoneStateMessage)
    assert back.decision is not None
    assert back.decision.palette_state == "palette_loaded"
    assert back.decision.content == ("carton",)
    assert back.decision.counts == {"palette": 1, "carton": 2}


def test_zone_state_without_decision_still_parses() -> None:
    """Old payload (pre-decision Backbone) has no ``decision`` key — the
    mixed-version rollout requires it to parse, defaulting to None."""
    data = json.loads(_make_zone_state().model_dump_json())
    data.pop("decision", None)
    back = parse_envelope(data)
    assert back.decision is None


def test_zone_decision_defaults() -> None:
    from backbone.comms.schemas import ZoneDecisionModel
    d = ZoneDecisionModel(palette_state="no_data")
    assert d.content == ()
    assert d.counts == {}


def test_zone_decision_extra_fields_rejected() -> None:
    from backbone.comms.schemas import ZoneDecisionModel
    with pytest.raises(ValidationError):
        ZoneDecisionModel.model_validate(
            {"palette_state": "no_palette", "rogue_key": 1})
    # And an unknown key nested under decision inside a full message too.
    from backbone.comms.schemas import ZoneStateMessage
    data = json.loads(_make_zone_state().model_dump_json())
    data["decision"] = {"palette_state": "no_palette", "rogue_key": 1}
    with pytest.raises(ValidationError):
        ZoneStateMessage.model_validate(data)


def test_zone_decision_from_backbone_dataclass() -> None:
    """``from_decision`` bridges the homography-side ``ZoneDecision`` without
    the schema module importing it (process-boundary style, like from_state)."""
    from backbone.comms.schemas import ZoneDecisionModel
    from backbone.homography.pallet_state_manager import ZoneDecision
    d = ZoneDecision(zone_id="zp_a", zone_name="rack_a",
                     palette_state="palette_empty", content=(),
                     counts={"palette": 1})
    m = ZoneDecisionModel.from_decision(d)
    assert m.palette_state == "palette_empty"
    assert m.content == ()
    assert m.counts == {"palette": 1}


def test_proximity_message_round_trip() -> None:
    """v5 ProximityMessage — the person↔object distance signal — survives the
    wire and parse_envelope discriminates it."""
    import json

    from backbone.comms.schemas import ProximityMessage, ProximityPair

    msg = ProximityMessage(
        ts=123.4,
        max_distance_m=6.0,
        pairs=(ProximityPair(
            person_track_id=7, object_track_id=3, object_cls="palette",
            distance_m=1.25, person_xy_m=(1.0, 2.0), object_xy_m=(2.0, 2.75)),),
    )
    parsed = parse_envelope(json.loads(msg.model_dump_json()))
    assert isinstance(parsed, ProximityMessage)
    assert parsed.pairs[0].distance_m == 1.25
    assert parsed.schema_version == SCHEMA_VERSION


def test_proximity_empty_pairs_is_explicit() -> None:
    """An empty proximity message (the clear signal) is valid — never silence."""
    import json

    from backbone.comms.schemas import ProximityMessage

    msg = ProximityMessage(ts=1.0, max_distance_m=6.0, pairs=())
    parsed = parse_envelope(json.loads(msg.model_dump_json()))
    assert parsed.pairs == ()


def test_observations_message_round_trip() -> None:
    """v6 ObservationsMessage — per-camera raw detections for display consumers."""
    import json

    from backbone.comms.schemas import ObservationDet, ObservationsMessage

    msg = ObservationsMessage(
        ts=12.5, camera_id="cam_a", frame_wh=(1920, 1080),
        dets=(ObservationDet(
            cls="palette", confidence=0.91, bbox_xyxy=(10.0, 20.0, 110.0, 90.0),
            foot_uv=(60.0, 90.0), occupancy_state="full",
            occupancy_content="carton", occupancy_confidence=0.8,
            mask_poly=((10.0, 20.0), (110.0, 20.0), (110.0, 90.0))),),
    )
    parsed = parse_envelope(json.loads(msg.model_dump_json()))
    assert isinstance(parsed, ObservationsMessage)
    assert parsed.dets[0].mask_poly is not None and len(parsed.dets[0].mask_poly) == 3
    assert parsed.dets[0].occupancy_state == "full"
    # Boxes-only det (masks not decoded) is equally valid.
    lean = ObservationsMessage(ts=1.0, camera_id="cam_b", frame_wh=(1920, 1080),
                               dets=(ObservationDet(
                                   cls="carton", confidence=0.5,
                                   bbox_xyxy=(0, 0, 1, 1), foot_uv=(0.5, 1.0)),))
    parsed = parse_envelope(json.loads(lean.model_dump_json()))
    assert parsed.dets[0].mask_poly is None


# ---- DetectionSetMessage (Direction 1: perception → metric engine) ----


def test_detection_set_round_trip_through_parse_envelope() -> None:
    from backbone.comms.schemas import DetectionSetMessage, WireDetection

    msg = DetectionSetMessage(
        ts=123.456, camera_id="cam_a", frame_wh=(1280, 720), seq=42,
        config_fingerprint="abc123",
        dets=(
            WireDetection(cls="palette", confidence=0.15,
                          bbox_xyxy=(10.0, 20.0, 110.0, 90.0),
                          foot_uv=(60.0, 90.0),
                          mask_poly=((10.0, 20.0), (110.0, 20.0), (60.0, 90.0))),
            WireDetection(cls="person", confidence=0.8,
                          bbox_xyxy=(0.0, 0.0, 50.0, 150.0), foot_uv=(25.0, 150.0),
                          keypoints_uv=((10.0, 10.0, 0.9),) * 17),
        ))
    parsed = parse_envelope(json.loads(msg.model_dump_json()))
    assert isinstance(parsed, DetectionSetMessage)
    assert parsed == msg


def test_detection_set_explicit_empty_is_valid() -> None:
    from backbone.comms.schemas import DetectionSetMessage

    msg = DetectionSetMessage(ts=1.0, camera_id="cam_b", frame_wh=(1920, 1080),
                              seq=0, dets=())
    parsed = parse_envelope(json.loads(msg.model_dump_json()))
    assert parsed.dets == ()


def test_send_json_datagram_shared_fragmentation() -> None:
    import socket as _socket

    from backbone.comms.schemas import (
        DetectionSetMessage,
        FragmentBuffer,
        FragmentMessage,
        WireDetection,
    )
    from backbone.comms.udp_sink import send_json_datagram

    recv = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
    recv.bind(("127.0.0.1", 0))
    recv.settimeout(2.0)
    send = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)

    poly = tuple((float(i), float(i)) for i in range(300))
    msg = DetectionSetMessage(
        ts=2.0, camera_id="cam_a", frame_wh=(1280, 720), seq=7,
        dets=tuple(WireDetection(cls="palette", confidence=0.5,
                                 bbox_xyxy=(0.0, 0.0, 9.0, 9.0),
                                 foot_uv=(4.0, 9.0), mask_poly=poly)
                   for _ in range(3)))
    payload = msg.model_dump_json().encode()
    assert len(payload) > 1300
    send_json_datagram(send, recv.getsockname(), payload)

    first = json.loads(recv.recvfrom(65535)[0])
    assert first["type"] == "fragment"
    frags = [first] + [json.loads(recv.recvfrom(65535)[0])
                       for _ in range(first["n"] - 1)]
    buf = FragmentBuffer()
    text = None
    for f in frags:
        text = buf.add(FragmentMessage.model_validate(f), now=0.0) or text
    assert parse_envelope(json.loads(text)) == msg
    recv.close()
    send.close()


# ---------------------------------------------------------------------------
# zone_id — STABLE zone identity on the wire (additive within v6)
# ---------------------------------------------------------------------------

def test_zone_id_round_trips_on_passing_and_zone_state_and_spec() -> None:
    """zone_id survives serialization on all three carriers and parse_envelope."""
    from backbone.comms.schemas import ZoneObject

    ev = PassingEventMessage(
        ts=1.0, track_id=5, cls="palette", zone="Loading Bay",
        zone_id="z1", direction="enter",
    )
    assert parse_envelope(json.loads(ev.model_dump_json())).zone_id == "z1"

    zs = ZoneStateMessage(
        ts=1.0, zone="Loading Bay", zone_id="z1",
        objects=(ZoneObject(track_id=7, cls="palette", confidence=0.9, xy_m=(1.0, 1.0)),),
        count=1,
    )
    assert parse_envelope(json.loads(zs.model_dump_json())).zone_id == "z1"

    spec = ZoneSpec(
        name="Loading Bay", zone_id="z1", kind="palette", type="storage",
        severity="info", polygon=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
    )
    back = ZoneSpec.model_validate(json.loads(spec.model_dump_json()))
    assert back.zone_id == "z1"


def test_legacy_v6_payload_without_zone_id_still_parses() -> None:
    """A payload from a pre-id Backbone (no zone_id key) parses with default "".

    This is why the change is additive (default-valued) and needs NO version
    bump — old v6 packets remain valid.
    """
    legacy_passing = {
        "schema_version": 6, "type": "passing", "ts": 1.0, "track_id": 5,
        "cls": "palette", "zone": "B3D", "direction": "enter",
    }
    parsed = parse_envelope(legacy_passing)
    assert isinstance(parsed, PassingEventMessage)
    assert parsed.zone_id == ""

    legacy_zone_state = {
        "schema_version": 6, "type": "zone_state", "ts": 1.0, "zone": "B3D",
        "objects": [], "count": 0,
    }
    assert parse_envelope(legacy_zone_state).zone_id == ""


# ---------------------------------------------------------------------------
# EtagereStateMessage (v6 additive: etagere_state — occupancy matrix)
# ---------------------------------------------------------------------------


def test_etagere_state_round_trip() -> None:
    cells = tuple(
        EtagereCellState(r=r, c=c, state="filled" if (r + c) % 2 else "empty",
                         confidence=0.9)
        for r in (1, 2, 3) for c in (1, 2, 3)
    )
    msg = EtagereStateMessage(ts=1.5, camera_id="cam_a", zone_id="et_1",
                              name="Étagère A", cells=cells, seq=4,
                              producer_id="isistream")
    data = json.loads(msg.model_dump_json())
    assert data["type"] == "etagere_state"
    assert data["rows"] == 3 and data["cols"] == 3
    assert len(data["cells"]) == 9
    back = parse_envelope(data)
    assert isinstance(back, EtagereStateMessage)
    assert back == msg
    assert MessageType.ETAGERE_STATE.value == "etagere_state"


def test_etagere_state_rejects_bad_state_and_extra() -> None:
    with pytest.raises(ValidationError):
        EtagereCellState(r=1, c=1, state="half")
    with pytest.raises(ValidationError):
        EtagereStateMessage(ts=0.0, camera_id="cam_a", zone_id="z", cells=(),
                            bogus=1)


def test_etagere_state_defaults() -> None:
    msg = EtagereStateMessage(ts=0.0, camera_id="cam_a", zone_id="z", cells=())
    assert msg.stabilized is False and msg.seq == 0 and msg.name == ""
    assert msg.schema_version == SCHEMA_VERSION
