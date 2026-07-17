# Modularity — plugins, contracts, copy-portable exports

**WHY** — swap what varies; keep the rest concrete; lift any app out of the repo.

**WHAT** — 5 plugin seams, wire contracts instead of shared code, `scripts/export_module.sh`.

## The five plugin seams — and only these

`backbone/core/interfaces.py`; count pinned by `tests/test_registry.py::test_five_seams_present`.

| Seam | Implementations | Why a plugin |
|---|---|---|
| `FrameSource` | `rtsp`, `replay` | RTSP, MP4 replay, future USB/ROS |
| `Detector` | `yolo_onnx`, `yolo_onnx_seg`, `yolo_onnx_pose`, `yolo_openvino`, `yolo_openvino_seg`, `rfdetr_onnx_seg` | NVIDIA vs Intel vs NMS-free RF-DETR |
| `Tracker` | `bytetrack` | ByteTrack-in-**meters**; SORT swappable |
| `Triangulator` | `opencv_dlt` | 2-cam DLT; aniposelib for ≥3 cams |
| `MetadataSink` | `udp`, `mqtt` | future ROS / S7 PLC |

**HOW** — decorator registers at import time; each package `__init__.py` imports its modules:

```python
# backbone/detection/yolo_onnx.py
@detector_registry.register("yolo_onnx")
class YoloOnnxDetector(Detector):
    ...
```

```python
# backbone/core/registry.py
registry.register(name)          # decorator: name -> class
registry.create(name, **kwargs)  # ONLY the orchestrator calls this
registry.names()                 # ['replay', 'rtsp']
```

!!! warning "No ABCs elsewhere"
    `FootProjector`, `CrossCamFusion`, `DisagreementGate`, `SubscriptionManager`, `ReprojectionGate`, `KeypointAssociator`, `TemporalStabilizer`, `ZoneScopedDetector` — concrete, single-implementation. `registry.create()` runs only in `backbone/runtime/orchestrator.py`.

## Process boundaries are contracts

Zero module imports; schemas, never code.

| Contract | Surface | Carries |
|---|---|---|
| **UDP/JSON wire** | `backbone/comms/schemas.py` (`SCHEMA_VERSION = 6`) | `DetectionSetMessage` in (:9010); tracks, zone state, passings, observations out |
| **isicomms MQTT→REST** | topics `isiMonitor3D/v1/<node>/...`; REST `/nodes /zones /tracks /passings` + `/ui` | versioned MQTT JSON in; Bearer-token polling out. Ports + `ISI_GATEWAY_*` env = frozen |

## Copy-portable module exports

Shared core travels as a **wheel** built at export time — nothing vendored:

```bash
scripts/export_module.sh <isical|isistream|isigen|isidet|isicomms> <dest-dir> [onprem|cloud]
```

### How to launch each module

| Module | In this repo (dev) | From an exported copy |
|---|---|---|
| **isical** (calibration studio, :8300) | `conda activate monitor3d && python -m isical` | `cd isical-portable && ./launch.sh` |
| **isistream** (perception producer) | dashboard **START** (or `python -m isistream --config config/backbone.yaml`) | `cd isistream-portable && cp config.example.yaml config.yaml` → fill cameras/model → `./launch.sh --config config.yaml` — needs system GStreamer |
| **isiGen** (synthetic-data studio, :8200) | `cd trainer/isiGen && ./launch.sh` | same — `./launch.sh` (plain copy) |
| **isidet** (trainer) | `conda activate isi-train && cd trainer/isidet && python scripts/run_train.py --config configs/train_pallet.yaml` | same, after `conda env create -f isi-train.yml` |
| **isicomms** (broker + gateway, :1883/:8080 + `/ui`) | `docker compose -p on-prem -f isicomms/deploy/onprem/docker-compose.yml up -d` (or bare: `python -m isicomms`) | `cd isicomms-portable && docker compose up -d` (cloud: `./gen-certs.sh` → `.env` first) |

!!! note
    isistream is only reusable as wheel+launcher. Wheels never committed; re-run the exporter to refresh.
