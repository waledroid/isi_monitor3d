from __future__ import annotations

import numpy as np

from backbone.core.types import Detection, Frame, FramePair
from backbone.shared.etagere import EtagereCell, EtagereConfig, EtagereModel, EtagereZone
from isistream.etagere import EtagereDetector


class _FakeDet:
    """Returns, per crop key, the detections queued for it; records the batch."""
    def __init__(self, by_key):
        self.by_key = by_key
        self.pairs: list[FramePair] = []

    def detect(self, pair: FramePair):
        self.pairs.append(pair)
        return {k: self.by_key.get(k, []) for k in pair.frames}


def _det(cls: str, conf: float, key: str) -> Detection:
    return Detection(camera_id=key, capture_ts=0.0, cls=cls, confidence=conf,
                     bbox_xyxy=(10, 10, 100, 100), foot_uv=(55, 100), keypoints_uv=None)


def _cfg(max_fps=2.0, margin=0.08) -> EtagereConfig:
    cells = [EtagereCell(r=r, c=c, rect=(c * 100, r * 100, c * 100 + 80, r * 100 + 80))
             for r in (1, 2, 3) for c in (1, 2, 3)]
    return EtagereConfig(
        model=EtagereModel(onnx_path="x.onnx", crop_margin=margin, max_fps=max_fps),
        zones=(EtagereZone(id="et_1", name="A", camera="cam_a", frame_wh=(640, 480),
                           cells=tuple(cells)),),
    )


def _frame(w=1280, h=960) -> Frame:
    return Frame(camera_id="cam_a", capture_ts=1.0, frame_idx=0,
                 image=np.zeros((h, w, 3), dtype=np.uint8))


def _cfg_two_zones(max_fps=2.0, margin=0.08) -> EtagereConfig:
    """Two zones on the same camera — 9 cells each, 18 crops total."""
    cells = [EtagereCell(r=r, c=c, rect=(c * 100, r * 100, c * 100 + 80, r * 100 + 80))
             for r in (1, 2, 3) for c in (1, 2, 3)]
    return EtagereConfig(
        model=EtagereModel(onnx_path="x.onnx", crop_margin=margin, max_fps=max_fps),
        zones=(
            EtagereZone(id="et_1", name="A", camera="cam_a", frame_wh=(640, 480),
                       cells=tuple(cells)),
            EtagereZone(id="et_2", name="B", camera="cam_a", frame_wh=(640, 480),
                       cells=tuple(cells)),
        ),
    )


def test_run_batches_nine_crops_and_maps_scale_and_margin() -> None:
    fake = _FakeDet({})
    ed = EtagereDetector(_cfg(), fake, producer_id="p", fingerprint="fp")
    msgs = ed.run({"cam_a": _frame()}, now=10.0)
    assert len(fake.pairs) == 1 and len(fake.pairs[0].frames) == 9
    crop = fake.pairs[0].frames["et_1:1:1"]
    # rect (100,100,180,180) in 640x480 → x2 in 1280x960 → (200,200,360,360), +8% margin (12.8 px each side)
    assert crop.image.shape[0] == crop.image.shape[1]
    assert 176 <= crop.image.shape[0] <= 190
    assert len(msgs) == 1 and msgs[0].zone_id == "et_1" and msgs[0].camera_id == "cam_a"
    assert msgs[0].producer_id == "p" and msgs[0].config_fingerprint == "fp"
    assert [c.state for c in msgs[0].cells] == ["unknown"] * 9
    assert msgs[0].ts == 1.0 and msgs[0].stabilized is False


def test_decision_per_cell() -> None:
    fake = _FakeDet({
        "et_1:1:1": [_det("filled_box", 0.9, "k"), _det("empty_box", 0.4, "k")],
        "et_1:1:2": [_det("empty_box", 0.8, "k")],
        "et_1:1:3": [_det("filled_box", 0.2, "k")],       # below 0.3 → unknown
    })
    ed = EtagereDetector(_cfg(), fake)
    msg = ed.run({"cam_a": _frame()}, now=0.0)[0]
    st = {(c.r, c.c): (c.state, c.confidence) for c in msg.cells}
    assert st[(1, 1)] == ("filled", 0.9)
    assert st[(1, 2)] == ("empty", 0.8)
    assert st[(1, 3)][0] == "unknown"
    assert st[(3, 3)][0] == "unknown"


def test_max_fps_gate_and_seq() -> None:
    fake = _FakeDet({})
    ed = EtagereDetector(_cfg(max_fps=2.0), fake)
    assert len(ed.run({"cam_a": _frame()}, now=0.0)) == 1
    assert ed.run({"cam_a": _frame()}, now=0.1) == []          # 0.5 s interval not elapsed
    m = ed.run({"cam_a": _frame()}, now=0.6)
    assert len(m) == 1 and m[0].seq == 1


def test_zone_without_fresh_frame_skipped() -> None:
    fake = _FakeDet({})
    ed = EtagereDetector(_cfg(), fake)
    assert ed.run({"cam_b": _frame()}, now=0.0) == []
    assert fake.pairs == []


def test_tick_isolates_per_message_send_failures() -> None:
    """One zone's failed UDP send must not drop another zone's message this
    tick — mirrors the DetectionSetMessage emit loop's per-camera isolation
    (a failure in the outer ``EtagereDetector.run()`` call is a different,
    already-guarded failure mode; this covers a failure in the per-message
    ``send_json_datagram`` call inside the loop over its results)."""
    from isistream import core as isicore
    from isistream.core import IsistreamCore

    fake = _FakeDet({})
    ed = EtagereDetector(_cfg_two_zones(), fake)

    calls = {"n": 0}

    def flaky_send(sock, addr, data):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("boom")

    orig_send = isicore.send_json_datagram
    isicore.send_json_datagram = flaky_send
    try:
        core = IsistreamCore(
            camera_ids=["cam_a"],
            frame_provider=lambda cam_id: (np.zeros((960, 1280, 3), dtype=np.uint8), 1.0),
            object_detector=None, pose_detector=None,
            ingest_addr=("127.0.0.1", 0), etagere_detector=ed)
        core.tick()
    finally:
        isicore.send_json_datagram = orig_send

    # ONE detect() call batching both zones' crops (9 cells x 2 zones).
    assert len(fake.pairs) == 1 and len(fake.pairs[0].frames) == 18
    # Both zone messages were attempted, plus cam_a's DetectionSetMessage
    # heartbeat (object_detector=None ⇒ explicit-empty) — 3 sends total.
    assert calls["n"] == 3
    # ...but only the surviving étagère send is counted — the failed first
    # send did not abort the loop and drop the second zone's message.
    assert sum(core.etagere_sent.values()) == 1
    # The DetectionSetMessage stage's own send (3rd call) succeeded and is
    # unaffected by the étagère stage's earlier failure.
    assert core.sets_sent["cam_a"] == 1
    # A per-message send failure is not the detector-stage failure mode.
    assert core.last_error is None
