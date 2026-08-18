"""Hermetic end-to-end: étagère producer → points ingest → stabilizer → UDP.

No ONNX, no CUDA — ``_AllFilled`` stands in for the yolo_onnx detector.
Exercises the real wire path: ``EtagereDetector.run`` emits raw
``EtagereStateMessage``s, a real ``DetectionIngest`` UDP thread receives them
and hands them to a real ``EtagereStateTracker``, whose stabilised output is
fanned out through a real ``Publisher``/``UdpSink`` to a loopback socket —
the same chain the Backbone + isistream run in production (Task 1-9), minus
the model.
"""

from __future__ import annotations

import json
import socket

import numpy as np

from backbone.comms.publisher import Publisher
from backbone.comms.schemas import parse_envelope
from backbone.comms.udp_sink import UdpSink
from backbone.core.types import Detection, Frame
from backbone.ingestion.points_in import DetectionIngest
from backbone.shared.etagere import EtagereCell, EtagereConfig, EtagereModel, EtagereZone
from backbone.shared.etagere_state import EtagereStateTracker
from isistream.etagere import EtagereDetector


class _AllFilled:
    def detect(self, pair):
        return {k: [Detection(camera_id=k, capture_ts=0.0, cls="filled_box", confidence=0.95,
                              bbox_xyxy=(0, 0, 10, 10), foot_uv=(5, 10), keypoints_uv=None)]
                for k in pair.frames}


def test_producer_to_bus_roundtrip() -> None:
    out_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    out_sock.bind(("127.0.0.1", 0))
    out_sock.settimeout(3.0)
    out_port = out_sock.getsockname()[1]
    sink = UdpSink(host="127.0.0.1", port=out_port)     # match UdpSink's ctor kwargs
    pub = Publisher([sink])
    tracker = EtagereStateTracker()
    ingest = DetectionIngest(["cam_a"], port=0, on_set=lambda s: None,
                             on_etagere=lambda m: (lambda o: pub.publish_etagere_state(o) if o else None)(tracker.update(m)))
    ingest.start()
    try:
        cells = [EtagereCell(r=r, c=c, rect=(c * 50, r * 50, c * 50 + 40, r * 50 + 40))
                 for r in (1, 2, 3) for c in (1, 2, 3)]
        cfg = EtagereConfig(model=EtagereModel(onnx_path="x"),
                            zones=(EtagereZone(id="et_1", name="A", camera="cam_a",
                                               frame_wh=(320, 240), cells=tuple(cells)),))
        det = EtagereDetector(cfg, _AllFilled(), producer_id="test")
        frame = Frame(camera_id="cam_a", capture_ts=1.0, frame_idx=0,
                      image=np.zeros((240, 320, 3), np.uint8))
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            for msg in det.run({"cam_a": frame}, now=0.0):
                s.sendto(msg.model_dump_json().encode(), ("127.0.0.1", ingest.port))
            data, _ = out_sock.recvfrom(65535)
            got = parse_envelope(json.loads(data.decode()))
            assert got.type.value == "etagere_state" and got.stabilized is True
            assert [c.state for c in got.cells] == ["filled"] * 9
            assert got.zone_id == "et_1" and got.producer_id == "test"
        finally:
            s.close()
    finally:
        ingest.stop()
        pub.close()
        out_sock.close()
