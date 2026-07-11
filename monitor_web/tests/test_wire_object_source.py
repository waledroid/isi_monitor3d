"""``WireObjectSource`` — cam-view object boxes straight from the bus observations.

Points mode (Direction 1): isistream detects only inside the floor zones and
echoes per-camera ``ObservationsMessage``s. The cam view draws those object dets
DIRECTLY, with NO dependency on the pixel-space ``zone_patches`` that gate the
ZONE PANELS — so boxes appear wherever isistream detects even when
``zone_patches.yaml`` is empty (the regression this source fixes: workers idle
with no patches ⇒ no cam-view boxes despite good detection).
"""

import time
from types import SimpleNamespace

import numpy as np
from backbone.comms.schemas import ObservationDet, ObservationsMessage

from monitor_web.pose_overlay import WireObjectSource


class _FakeBus:
    def __init__(self, msg):
        self._msg = msg

    def snapshot(self):
        return SimpleNamespace(observations_by_camera=(
            {self._msg.camera_id: self._msg} if self._msg else {}))


def _obs_msg(cam="cam_a", dets=None, ts=None, frame_wh=(640, 480)):
    return ObservationsMessage(
        ts=ts if ts is not None else time.time(), camera_id=cam,
        frame_wh=frame_wh, dets=tuple(dets or []))


def _obs_det(cls="palette", conf=0.9, bbox=(40.0, 40.0, 120.0, 120.0), **extra):
    return ObservationDet(cls=cls, confidence=conf, bbox_xyxy=bbox,
                          foot_uv=((bbox[0] + bbox[2]) / 2.0, bbox[3]), **extra)


def _frame(w=640, h=480):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_object_dets_come_through_with_no_patches():
    """The whole point: a palette on the wire reaches the cam view even though
    there is NO zone patch anywhere — the source takes no patches at all."""
    bus = _FakeBus(_obs_msg(dets=[_obs_det(cls="palette", conf=0.87)]))
    src = WireObjectSource(lambda: bus, "cam_a")
    out = src.objects(_frame())
    assert len(out) == 1
    assert out[0].cls == "palette"
    assert out[0].confidence == 0.87


def test_persons_are_filtered_out():
    """Persons ride WirePoseSource (skeletons) — the object source drops them so
    they are never boxed as objects."""
    bus = _FakeBus(_obs_msg(dets=[_obs_det(cls="person"), _obs_det(cls="carton")]))
    out = WireObjectSource(lambda: bus, "cam_a").objects(_frame())
    assert [d.cls for d in out] == ["carton"]


def test_bboxes_scale_to_the_display_frame():
    """Observations are in the producer's frame_wh; the source rescales to the
    cam frame so boxes land on the right pixels."""
    # obs frame 320x240, display 640x480 → exactly 2x on both axes.
    bus = _FakeBus(_obs_msg(frame_wh=(320, 240),
                            dets=[_obs_det(bbox=(10.0, 20.0, 50.0, 60.0))]))
    out = WireObjectSource(lambda: bus, "cam_a").objects(_frame(640, 480))
    assert out[0].bbox_xyxy == (20.0, 40.0, 100.0, 120.0)


def test_mask_polygon_scales_too():
    bus = _FakeBus(_obs_msg(frame_wh=(320, 240),
                            dets=[_obs_det(mask_poly=((10.0, 10.0), (20.0, 30.0)))]))
    out = WireObjectSource(lambda: bus, "cam_a").objects(_frame(640, 480))
    assert out[0].mask_poly == [[20.0, 20.0], [40.0, 60.0]]


def test_stale_observations_yield_nothing():
    """Observations older than the max age produce no boxes (no stale ghosts)."""
    old = _obs_msg(dets=[_obs_det()], ts=time.time() - 10.0)
    assert WireObjectSource(lambda: old and _FakeBus(old), "cam_a")  # sanity
    out = WireObjectSource(lambda: _FakeBus(old), "cam_a").objects(_frame())
    assert out == []


def test_absent_bus_or_camera_yields_nothing():
    assert WireObjectSource(lambda: None, "cam_a").objects(_frame()) == []
    bus = _FakeBus(_obs_msg(cam="cam_b", dets=[_obs_det()]))  # different camera
    assert WireObjectSource(lambda: bus, "cam_a").objects(_frame()) == []
