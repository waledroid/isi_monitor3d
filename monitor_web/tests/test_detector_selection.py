"""Plugin auto-selection + model discovery for the detection overlay.

Hermetic: no real ONNX is loaded — ``select_plugin`` is a pure function over
output names, and the model-discovery test uses tiny stub ``.onnx`` files on disk.
"""

from __future__ import annotations

import importlib

from monitor_web.detection_overlay import select_plugin

# --- select_plugin: the pure plugin-selection rule ---------------------------

def test_rfdetr_selected_from_named_outputs():
    # The three RF-DETR output names → rfdetr_onnx_seg, regardless of base backend.
    assert select_plugin("yolo_onnx", ["dets", "labels", "masks"]) == "rfdetr_onnx_seg"


def test_rfdetr_selected_when_three_named_outputs_partial_match():
    # 3 outputs including the RF-DETR names (any order / extra) still → rfdetr.
    assert select_plugin("yolo_onnx", ["labels", "dets", "masks"]) == "rfdetr_onnx_seg"


def test_rfdetr_not_selected_for_two_yolo_outputs():
    # YOLO-seg = 2 unnamed/anonymous outputs → yolo_onnx_seg, not rfdetr.
    assert select_plugin("yolo_onnx", ["output0", "output1"]) == "yolo_onnx_seg"


def test_yolo_seg_from_two_outputs():
    assert select_plugin("yolo_onnx", ["a", "b"]) == "yolo_onnx_seg"
    assert select_plugin("yolo_openvino", ["a", "b"]) == "yolo_openvino_seg"


def test_yolo_detect_from_one_output():
    assert select_plugin("yolo_onnx", ["output0"]) == "yolo_onnx"
    assert select_plugin("yolo_openvino", ["output0"]) == "yolo_openvino"


def test_base_plugin_unchanged_for_empty_outputs():
    assert select_plugin("yolo_onnx", []) == "yolo_onnx"


def test_rfdetr_priority_over_arity():
    # Even if there were only 2 outputs, an RF-DETR-named set wins. (Defensive:
    # exporters could drop masks; dets+labels alone is still RF-DETR.)
    assert select_plugin("yolo_onnx", ["dets", "labels"]) == "rfdetr_onnx_seg"


# --- list_trained_onnx: model discovery now scans models/rfdetr/ too ---------

def test_list_trained_onnx_finds_rfdetr_under_models(tmp_path, monkeypatch):
    import monitor_web.model_store as ov  # canonical home of model discovery

    repo = tmp_path
    runs = repo / "trainer/isidet/runs/segment/models/yolo/r1/weights"
    rfdetr = repo / "trainer/isidet/models/rfdetr/07-06-2026_0909"
    runs.mkdir(parents=True)
    rfdetr.mkdir(parents=True)
    yolo_onnx = runs / "best.onnx"
    rfdetr_onnx = rfdetr / "inference_model.sim.onnx"
    yolo_onnx.write_bytes(b"\x00")
    rfdetr_onnx.write_bytes(b"\x00")

    monkeypatch.setattr(ov, "_REPO_ROOT", repo)
    monkeypatch.setattr(ov, "_RUNS_ROOT", repo / "trainer/isidet/runs")
    monkeypatch.setattr(
        ov, "_MODEL_ROOTS",
        (repo / "trainer/isidet/runs", repo / "trainer/isidet/models/rfdetr"),
    )

    files = ov.list_trained_onnx()
    paths = {f["path"] for f in files}
    labels = {f["label"] for f in files}
    assert str(yolo_onnx) in paths
    assert str(rfdetr_onnx) in paths
    # Labels are relative to trainer/isidet/, so the RF-DETR export reads cleanly.
    assert "models/rfdetr/07-06-2026_0909/inference_model.sim.onnx" in labels
    assert "runs/segment/models/yolo/r1/weights/best.onnx" in labels


def test_module_imports_clean():
    # Guard: the overlay module still imports without error after the refactor.
    importlib.reload(importlib.import_module("monitor_web.detection_overlay"))
