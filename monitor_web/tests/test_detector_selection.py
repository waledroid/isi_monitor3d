"""Plugin auto-selection + model discovery — the OpenVINO-only surface.

Hermetic: no real IR is loaded — ``select_plugin`` is a pure function over
output names, and the model-discovery test uses tiny stub ``.xml`` files on disk.
"""

from __future__ import annotations

import importlib

from monitor_web.detection_overlay import select_plugin

# --- select_plugin: the pure plugin-selection rule ---------------------------


def test_yolo_seg_from_two_outputs():
    # YOLO-seg IR = 2 outputs (head + mask protos) → the seg plugin.
    assert select_plugin("yolo_openvino", ["a", "b"]) == "yolo_openvino_seg"


def test_yolo_detect_from_one_output():
    assert select_plugin("yolo_openvino", ["output0"]) == "yolo_openvino"


def test_empty_outputs_fall_back_to_detect():
    # Unreadable IR → no output names → the safe default is the detect plugin.
    assert select_plugin("yolo_openvino", []) == "yolo_openvino"


# --- model discovery: *.xml IRs under the ONE models/ root -------------------


def test_list_trained_onnx_finds_xml_under_models(tmp_path, monkeypatch):
    """list_trained_onnx() surfaces object-detection IRs under <repo>/models/;
    IRs whose path mentions "pose" appear ONLY in list_pose_onnx()."""
    import monitor_web.model_store as ms

    models = tmp_path / "m"                      # avoid "pose" in tmp dir names
    (models / "pallet_nano").mkdir(parents=True)
    (models / "yolo11n-pose").mkdir(parents=True)
    det_xml = models / "pallet_nano" / "model.xml"
    pose_xml = models / "yolo11n-pose" / "model.xml"
    det_xml.write_text("<net/>")
    pose_xml.write_text("<net/>")

    monkeypatch.setattr(ms, "_MODELS_ROOT", models)

    det_files = ms.list_trained_onnx()
    pose_files = ms.list_pose_onnx()
    assert {f["path"] for f in det_files} == {str(det_xml)}
    assert {f["path"] for f in pose_files} == {str(pose_xml)}
    # Labels are relative to models/, so the dropdown reads cleanly.
    assert det_files[0]["label"] == "pallet_nano/model.xml"
    assert pose_files[0]["label"] == "yolo11n-pose/model.xml"


def test_module_imports_clean():
    # Guard: the overlay module still imports without error after the refactor.
    importlib.reload(importlib.import_module("monitor_web.detection_overlay"))
