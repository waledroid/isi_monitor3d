from __future__ import annotations

import logging

import numpy as np
import yaml

from backbone.core.types import Detection, Frame, FramePair
from backbone.shared.etagere import EtagereCell, EtagereConfig, EtagereModel, EtagereZone
from isistream.core import _build_etagere_stage
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


def test_run_isolates_a_zone_whose_crop_building_raises() -> None:
    """One zone's crop-building blowing up (degenerate rects, off-frame
    scaling, any exception) must not abort run() and drop every other zone's
    message (I3) — the other zone still batches its crops and gets a
    message."""
    fake = _FakeDet({})
    ed = EtagereDetector(_cfg_two_zones(), fake)
    orig_crop = ed._crop

    def flaky_crop(frame, zone, rect, angle_deg=0.0):
        if zone.id == "et_1":
            raise ValueError("boom")
        return orig_crop(frame, zone, rect, angle_deg)

    ed._crop = flaky_crop
    msgs = ed.run({"cam_a": _frame()}, now=0.0)
    assert [m.zone_id for m in msgs] == ["et_2"]
    # Only zone et_2's 9 crops made it into the (single) detect() batch.
    assert len(fake.pairs) == 1 and len(fake.pairs[0].frames) == 9
    assert all(k.startswith("et_2:") for k in fake.pairs[0].frames)


def test_run_isolates_a_zone_whose_message_building_raises() -> None:
    """A failure while assembling a zone's EtagereStateMessage (the second
    per-zone try/except) must likewise skip only that zone."""
    fake = _FakeDet({})
    ed = EtagereDetector(_cfg_two_zones(), fake)

    import isistream.etagere as et_mod

    # Patch EtagereStateMessage so the FIRST call (zone et_1, since due_zones
    # preserves cfg.zones order) raises, the second succeeds.
    calls = {"n": 0}
    orig_msg_cls = et_mod.EtagereStateMessage

    def flaky_message(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return orig_msg_cls(*a, **kw)

    et_mod.EtagereStateMessage = flaky_message
    try:
        msgs = ed.run({"cam_a": _frame()}, now=0.0)
    finally:
        et_mod.EtagereStateMessage = orig_msg_cls
    assert [m.zone_id for m in msgs] == ["et_2"]
    # Both zones' crops WERE batched into the one detect() call — the
    # message-build failure happens after detection, not before.
    assert len(fake.pairs) == 1 and len(fake.pairs[0].frames) == 18


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


# ---- C1(c): _build_etagere_stage distinguishes "disabled" (silent) from
# "configured but broken" (ERROR + traceback) ----

def test_build_etagere_stage_none_config_path_is_silent(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        assert _build_etagere_stage({}, None, producer_id="p") is None
    assert caplog.records == []


def test_build_etagere_stage_missing_file_is_silent(tmp_path, caplog) -> None:
    """No etagere.yaml beside backbone.yaml — the documented "missing file ⇒
    feature off, no error" contract. Most deployments never touch étagère."""
    backbone_yaml = tmp_path / "backbone.yaml"
    backbone_yaml.write_text("cameras: {}\n")
    with caplog.at_level(logging.WARNING):
        result = _build_etagere_stage({}, str(backbone_yaml), producer_id="p")
    assert result is None
    assert caplog.records == []


def test_build_etagere_stage_invalid_yaml_logs_error(tmp_path, caplog) -> None:
    """A PRESENT but broken config (fails pydantic validation) is an operator
    mistake, not a disabled feature — must be loud."""
    backbone_yaml = tmp_path / "backbone.yaml"
    backbone_yaml.write_text("cameras: {}\n")
    (tmp_path / "etagere.yaml").write_text(yaml.safe_dump({
        "model": {"onnx_path": "x.onnx"},
        "zones": [{"id": "z", "camera": "cam_a", "frame_wh": [10, 10],
                   "cells": [{"r": 1, "c": 1, "rect": [0, 0, 5, 5]}]}],  # only 1 of 9 cells
    }))
    with caplog.at_level(logging.ERROR):
        result = _build_etagere_stage({}, str(backbone_yaml), producer_id="p")
    assert result is None
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.ERROR
    assert "étagère config" in caplog.records[0].message


def test_build_etagere_stage_detector_build_failure_logs_error(tmp_path, caplog, monkeypatch) -> None:
    """A valid, enabled config whose detector fails to build (unreachable
    onnx_path, bad providers, ...) must not silently leave the feature off —
    ERROR + traceback, never a bare WARNING (and never onnxruntime/CUDA
    actually touched here — the detector builder itself is stubbed out)."""
    backbone_yaml = tmp_path / "backbone.yaml"
    backbone_yaml.write_text("cameras: {}\n")
    (tmp_path / "etagere.yaml").write_text(yaml.safe_dump({
        "model": {"onnx_path": "/nonexistent/model.onnx"},
        "zones": [{"id": "et_1", "camera": "cam_a", "frame_wh": [1920, 1080],
                   "cells": [{"r": r, "c": c, "rect": [c * 10, r * 10, c * 10 + 8, r * 10 + 8]}
                             for r in (1, 2, 3) for c in (1, 2, 3)]}],
    }))

    def _raise(*a, **kw):
        raise RuntimeError("onnx session build failed")

    monkeypatch.setattr("isistream.etagere.build_etagere_detector", _raise)
    with caplog.at_level(logging.ERROR):
        result = _build_etagere_stage({}, str(backbone_yaml), producer_id="p")
    assert result is None
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.ERROR
    assert "detector failed to build" in caplog.records[0].message


def test_rotated_cell_is_cropped_upright() -> None:
    """A cell with angle_deg is warped upright before cropping: a bright bar
    painted along the cell's tilted axis comes out horizontal in the crop."""
    import cv2

    from backbone.shared.etagere import EtagereCell
    fake = _FakeDet({})
    # one 1x1 "grid": a 200x60 cell centred at (320,240), tilted 20° clockwise
    cell = EtagereCell(r=1, c=1, rect=(220, 210, 420, 270), angle_deg=20.0)
    cfg = EtagereConfig(
        model=EtagereModel(onnx_path="x.onnx", crop_margin=0.0),
        zones=(EtagereZone(id="z", camera="cam_a", frame_wh=(640, 480),
                           rows=1, cols=1, cells=(cell,)),),
    )
    img = np.zeros((480, 640, 3), np.uint8)
    # bright bar along the tilted axis: draw axis-aligned then rotate the IMAGE
    # by 20° clockwise about the cell centre (cv2 positive angle = CCW → use -20)
    cv2.rectangle(img, (240, 232), (400, 248), (255, 255, 255), -1)
    m = cv2.getRotationMatrix2D((320, 240), -20.0, 1.0)
    img = cv2.warpAffine(img, m, (640, 480))
    ed = EtagereDetector(cfg, fake)
    ed.run({"cam_a": Frame(camera_id="cam_a", capture_ts=0.0, frame_idx=0, image=img)}, now=0.0)
    crop = fake.pairs[0].frames["z:1:1"].image
    h, w = crop.shape[:2]
    assert (w, h) == (200, 60)
    mid = crop[h // 2 - 2:h // 2 + 3, 20:-20].mean()     # along the horizontal middle
    top = crop[2:7, 20:-20].mean()                         # top band should be dark
    assert mid > 200 and top < 40, (mid, top)
