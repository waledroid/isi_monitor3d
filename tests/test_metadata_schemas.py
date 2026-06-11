"""``Track2DMessage`` / ``Track3DMessage`` — pydantic round-trip + version gate."""

from __future__ import annotations

import json

import numpy as np
import pytest
from pydantic import ValidationError

from backbone.core.types import Track2D, Track3D
from backbone.metadata.schemas import (
    SCHEMA_VERSION,
    MessageType,
    SchemaVersionError,
    Track2DMessage,
    Track3DMessage,
    parse_envelope,
)


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
