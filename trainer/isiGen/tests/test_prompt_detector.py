"""Auto-prompt detector for the masks phase (detect → box → SAM2).

Hermetic: no GPU / no real ONNX. The decoders are exercised on synthetic ORT
output arrays via an injected fake session; ``run_masks`` + the studio routes use
stub maskers/detectors. Pins: RF-DETR + YOLO box decode → class-mapped
``MaskPrompt``s, ``build_prompt_detector`` output-name selection, class mapping,
``list_detector_onnx`` discovery, and the run_masks/Studio integration.
"""

from __future__ import annotations

import numpy as np
import pytest
from src.core.manifest import MaskPrompt
from src.stages.detection import base as det_base
from src.stages.detection.rfdetr_onnx import RfdetrPromptDetector
from src.stages.detection.yolo_onnx import YoloPromptDetector

CLASSES = ["palette", "carton", "polybag"]


class _Inp:
    def __init__(self, name, shape):
        self.name, self.shape = name, shape


class _Meta:
    def __init__(self, m):
        self.custom_metadata_map = m


class _FakeSession:
    """Minimal stand-in for an ORT InferenceSession."""

    def __init__(self, input_shape, outputs, meta=None):
        self._inputs = [_Inp("images", input_shape)]
        self._outputs = outputs       # list, in output_names order
        self._meta = meta or {}

    def get_inputs(self):
        return self._inputs

    def run(self, _names, _feed):
        return self._outputs

    def get_modelmeta(self):
        return _Meta(self._meta)


# ── RF-DETR decode ──────────────────────────────────────────────────────────
def test_rfdetr_decode_emits_classmapped_box():
    # 2 queries: q0 = polybag (col 3) high logit; q1 = all low → dropped.
    dets = np.array([[[0.5, 0.5, 0.2, 0.2], [0.1, 0.1, 0.05, 0.05]]], dtype=np.float32)
    labels = np.full((1, 2, 4), -10.0, dtype=np.float32)   # cols: bg, palette, carton, polybag
    labels[0, 0, 3] = 10.0                                  # q0 strongly polybag
    sess = _FakeSession([1, 3, 432, 432], [dets, labels])
    det = RfdetrPromptDetector("x.onnx", session=sess, output_names=["dets", "labels"],
                               class_names=CLASSES, confidence_threshold=0.3)
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    prompts = det.detect(img, CLASSES)
    assert len(prompts) == 1
    p = prompts[0]
    assert p.kind == "box" and p.class_name == "polybag"
    x1, y1, x2, y2 = p.xyxy
    assert x1 == pytest.approx(320, abs=2) and x2 == pytest.approx(480, abs=2)
    assert y1 == pytest.approx(240, abs=2) and y2 == pytest.approx(360, abs=2)


def test_rfdetr_threshold_drops_low_scores():
    dets = np.array([[[0.5, 0.5, 0.2, 0.2]]], dtype=np.float32)
    labels = np.full((1, 1, 4), -10.0, dtype=np.float32)
    labels[0, 0, 3] = 0.0                                   # sigmoid(0)=0.5
    sess = _FakeSession([1, 3, 432, 432], [dets, labels])
    det = RfdetrPromptDetector("x.onnx", session=sess, output_names=["dets", "labels"],
                               class_names=CLASSES, confidence_threshold=0.9)
    assert det.detect(np.zeros((600, 800, 3), np.uint8), CLASSES) == []


# ── YOLO decode ─────────────────────────────────────────────────────────────
def test_yolo_raw_head_decode_reads_names_metadata():
    # (4+nc, A) raw head, A=2; anchor0 = polybag at target px (square img → no pad).
    head = np.zeros((1, 7, 2), dtype=np.float32)
    head[0, :, 0] = [320, 320, 100, 100, 0.05, 0.05, 0.9]   # cx,cy,w,h, palette,carton,polybag
    head[0, :, 1] = [10, 10, 5, 5, 0.01, 0.01, 0.02]        # below thresh
    sess = _FakeSession([1, 3, 640, 640], [head],
                        meta={"names": "{0: 'palette', 1: 'carton', 2: 'polybag'}"})
    det = YoloPromptDetector("x.onnx", session=sess, output_names=["output0"])
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    prompts = det.detect(img, CLASSES)
    assert len(prompts) == 1
    assert prompts[0].class_name == "polybag"
    x1, y1, x2, y2 = prompts[0].xyxy
    assert (x1, y1, x2, y2) == pytest.approx((270, 270, 370, 370), abs=2)


def test_yolo_end2end_head_decode():
    # NMS-free rows: [x1,y1,x2,y2,score,cls]; neither axis == 4+nc(=7).
    head = np.array([[[270, 270, 370, 370, 0.8, 2],
                      [0, 0, 10, 10, 0.1, 0]]], dtype=np.float32)
    sess = _FakeSession([1, 3, 640, 640], [head],
                        meta={"names": "{0: 'palette', 1: 'carton', 2: 'polybag'}"})
    det = YoloPromptDetector("x.onnx", session=sess, output_names=["output0"])
    prompts = det.detect(np.zeros((640, 640, 3), np.uint8), CLASSES)
    assert [p.class_name for p in prompts] == ["polybag"]


# ── class mapping ───────────────────────────────────────────────────────────
def test_map_class_single_class_accepts_anything():
    det = RfdetrPromptDetector("x.onnx", session=_FakeSession([1, 3, 4, 4], []),
                               output_names=["dets", "labels"])
    assert det._map_class("whatever", ["polybag"]) == "polybag"


def test_map_class_case_insensitive_and_unknown_dropped():
    det = RfdetrPromptDetector("x.onnx", session=_FakeSession([1, 3, 4, 4], []),
                               output_names=["dets", "labels"])
    assert det._map_class("Polybag", CLASSES) == "polybag"
    assert det._map_class("forklift", CLASSES) is None


def test_map_class_override():
    det = RfdetrPromptDetector("x.onnx", session=_FakeSession([1, 3, 4, 4], []),
                               output_names=["dets", "labels"],
                               class_map={"bag": "polybag"})
    assert det._map_class("bag", CLASSES) == "polybag"


# ── factory output-name selection ───────────────────────────────────────────
@pytest.mark.parametrize("names,cls", [
    (["dets", "labels", "masks"], RfdetrPromptDetector),
    (["dets", "labels"], RfdetrPromptDetector),
    (["output0"], YoloPromptDetector),
    (["output0", "output1"], YoloPromptDetector),   # seg head + protos
])
def test_build_prompt_detector_selects_by_output_names(monkeypatch, names, cls):
    monkeypatch.setattr(det_base, "_build_session",
                        lambda p: (_FakeSession([1, 3, 4, 4], []), names))
    det = det_base.build_prompt_detector("x.onnx", class_names=CLASSES)
    assert isinstance(det, cls)


# ── list_detector_onnx ──────────────────────────────────────────────────────
def test_list_detector_onnx_scans_roots(tmp_path, monkeypatch):
    from src.core import models
    root = tmp_path / "isidet"
    (root / "models" / "rfdetr").mkdir(parents=True)
    (root / "runs" / "yolo").mkdir(parents=True)
    (root / "models" / "rfdetr" / "a.onnx").write_bytes(b"x")
    (root / "runs" / "yolo" / "b.onnx").write_bytes(b"x")
    (root / "models" / "notes.txt").write_text("ignore")
    monkeypatch.setattr(models, "_ISIDET_ROOT", root)
    monkeypatch.setattr(models, "_MODEL_ROOTS", (root / "models", root / "runs"))
    out = models.list_detector_onnx()
    labels = [m["label"] for m in out]
    assert labels == ["models/rfdetr/a.onnx", "runs/yolo/b.onnx"]
    assert all(m["path"].endswith(".onnx") for m in out)


# ── run_masks integration ───────────────────────────────────────────────────
class _StubMasker:
    def load(self):
        pass

    def segment_prompted(self, img, prompts):
        h, w = img.shape[:2]
        out = {}
        for p in prompts:
            m = np.zeros((h, w), bool)
            m[10:50, 10:50] = True
            out[p.class_name] = m
        return out

    def segment_auto(self, img):
        h, w = img.shape[:2]
        m = np.zeros((h, w), bool)
        m[10:50, 10:50] = True
        return [m]

    def close(self):
        pass


class _StubDetector:
    loaded = closed = False

    def load(self):
        type(self).loaded = True

    def detect(self, img, project_class_names):
        return [MaskPrompt(kind="box", class_name="polybag", xyxy=[10.0, 10.0, 60.0, 60.0])]

    def close(self):
        type(self).closed = True


def test_run_masks_auto_prompt(tiny_project, monkeypatch):
    from src.core import runners
    from src.core.manifest import Manifest
    from src.core.project import load_project
    from src.stages import detection as detection_pkg
    pdir, _ = tiny_project

    monkeypatch.setattr(runners.MASKERS, "create", lambda *a, **k: _StubMasker())
    monkeypatch.setattr(detection_pkg, "build_prompt_detector",
                        lambda path, **cfg: _StubDetector())

    # Give one record hand-drawn prompts — those must WIN over the detector.
    m = Manifest.load(pdir)
    recs = m.active()
    hand = recs[0]
    hand.mask_prompts = [MaskPrompt(kind="box", class_name=hand.class_name,
                                    xyxy=[5.0, 5.0, 40.0, 40.0])]
    m.upsert(hand)
    m.save()

    res = runners.run_masks(pdir, prompt_detector="/fake/model.onnx")
    assert res["masked"] == len(recs)
    assert _StubDetector.loaded and _StubDetector.closed       # loaded once + cleaned up

    # onnx_path persisted to config.
    cfg = load_project(pdir).phase("masking")
    assert cfg["prompt_detector"]["onnx_path"] == "/fake/model.onnx"

    # Sources: hand-drawn → prompted; the rest → auto_detect.
    after = {r.id: r for r in Manifest.load(pdir).active()}
    assert after[hand.id].mask_source == "prompted"
    others = [r for rid, r in after.items() if rid != hand.id]
    assert others and all(r.mask_source == "auto_detect" for r in others)


def test_run_masks_clear_detector(tiny_project, monkeypatch):
    """Passing "none" clears the persisted detector and falls back to SAM2."""
    from src.core import runners
    from src.core.project import load_project
    pdir, _ = tiny_project
    monkeypatch.setattr(runners.MASKERS, "create", lambda *a, **k: _StubMasker())

    runners.run_masks(pdir, prompt_detector="none")
    assert load_project(pdir).phase("masking")["prompt_detector"]["onnx_path"] is None


# ── studio routes ───────────────────────────────────────────────────────────
def test_route_detector_models(tiny_project, monkeypatch):
    from fastapi.testclient import TestClient
    from src.core import models
    from src.studio.app import create_app
    from src.studio.config import Settings
    pdir, _ = tiny_project
    root = pdir.parent.parent / "isidet"
    (root / "models").mkdir(parents=True)
    (root / "models" / "m.onnx").write_bytes(b"x")
    monkeypatch.setattr(models, "_ISIDET_ROOT", root)
    monkeypatch.setattr(models, "_MODEL_ROOTS", (root / "models", root / "runs"))
    with TestClient(create_app(Settings())) as c:
        r = c.get("/api/p/tiny/detector-models").json()
    assert r["current"] is None
    assert [m["label"] for m in r["models"]] == ["models/m.onnx"]


def test_route_detect_prompts(tiny_project, monkeypatch):
    from fastapi.testclient import TestClient
    from src.stages import detection as detection_pkg
    from src.studio.app import create_app
    from src.studio.config import Settings
    _pdir, _ = tiny_project
    monkeypatch.setattr(detection_pkg, "build_prompt_detector",
                        lambda path, **cfg: _StubDetector())
    with TestClient(create_app(Settings())) as c:
        rid = c.get("/api/p/tiny/records").json()["records"][0]["id"]
        r = c.post(f"/api/p/tiny/records/{rid}/detect-prompts",
                   json={"onnx_path": "/fake/model.onnx"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] and body["prompts"][0]["class_name"] == "polybag"
    assert body["prompts"][0]["kind"] == "box"
