from __future__ import annotations

import socket
import time

from backbone.comms.schemas import EtagereCellState, EtagereStateMessage
from backbone.ingestion.points_in import DetectionIngest


def _msg(cam="cam_a"):
    return EtagereStateMessage(ts=1.0, camera_id=cam, zone_id="et_1",
                               cells=(EtagereCellState(r=1, c=1, state="filled"),),
                               rows=1, cols=1)


def _send(port: int, msg) -> None:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.sendto(msg.model_dump_json().encode("utf-8"), ("127.0.0.1", port))
    s.close()


def _wait(pred, timeout=2.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.02)
    return False


def test_etagere_routed_to_on_etagere_not_on_set() -> None:
    sets, shelves = [], []
    ing = DetectionIngest(["cam_a"], port=0, on_set=sets.append, on_etagere=shelves.append)
    ing.start()
    try:
        _send(ing.port, _msg())
        assert _wait(lambda: len(shelves) == 1)
        assert isinstance(shelves[0], EtagereStateMessage) and shelves[0].zone_id == "et_1"
        assert sets == []
        assert ing.etagere_by_zone.get("et_1") == 1
    finally:
        ing.stop()


def test_etagere_unknown_camera_ignored_and_no_callback_counts_dropped() -> None:
    shelves = []
    ing = DetectionIngest(["cam_a"], port=0, on_set=lambda s: None, on_etagere=shelves.append)
    ing.start()
    try:
        _send(ing.port, _msg(cam="cam_zzz"))
        time.sleep(0.2)
        assert shelves == []
    finally:
        ing.stop()
    ing2 = DetectionIngest(["cam_a"], port=0, on_set=lambda s: None)   # no on_etagere
    ing2.start()
    try:
        _send(ing2.port, _msg())
        assert _wait(lambda: ing2.dropped_malformed >= 1)
    finally:
        ing2.stop()
