# Modularity — plugins, contracts, copy-portable exports

**WHY** — swap what genuinely varies (cameras, models, trackers, transports) without touching the pipeline; keep everything else concrete so the code stays readable; and let every app be lifted out of the repo and dropped into another project.

**WHAT** — three mechanisms: **5 plugin seams** (ABCs + registries), **wire contracts instead of shared code** across processes, and **`scripts/export_module.sh`** for self-contained module folders.

## The five plugin seams — and only these

Defined in `backbone/core/interfaces.py`; the count is pinned by `tests/test_registry.py::test_five_seams_present`.

| Seam | v1 implementations | Why a plugin |
|---|---|---|
| `FrameSource` | `rtsp`, `replay` (`backbone/ingestion/`) | RTSP today, MP4 replay for dev/tests, future USB/ROS |
| `Detector` | `yolo_onnx`, `yolo_onnx_seg`, `yolo_onnx_pose`, `yolo_openvino`, `yolo_openvino_seg`, `rfdetr_onnx_seg` (`backbone/detection/`) | NVIDIA (ORT CUDA/TensorRT) vs Intel (OpenVINO) vs NMS-free RF-DETR |
| `Tracker` | `bytetrack` (`backbone/homography/`) | ByteTrack-in-**meters**; SORT/OC-SORT swappable |
| `Triangulator` | `opencv_dlt` (`backbone/triangulation/`) | 2-cam DLT now; aniposelib for ≥3 cams (S5.5) |
| `MetadataSink` | `udp`, `mqtt` (`backbone/comms/`) | UDP/JSON local, MQTT fabric, future ROS/S7 PLC |

**HOW** — a decorator registers the class at import time; each package `__init__.py` imports its implementation modules so importing the package fires registration:

```python
# backbone/detection/yolo_onnx.py — real code
@detector_registry.register("yolo_onnx")
class YoloOnnxDetector(Detector):
    ...
```

```python
# backbone/core/registry.py — the whole mechanism
registry.register(name)   # decorator: name -> class, refuses duplicates
registry.create(name, **kwargs)   # ONLY the orchestrator calls this
registry.names()          # ['replay', 'rtsp'] after `import backbone.ingestion`
```

!!! warning "Where ABCs do NOT belong"
    `FootProjector`, `CrossCamFusion`, `DisagreementGate`, `SubscriptionManager`, `ReprojectionGate`, `KeypointAssociator`, `TemporalStabilizer`, `ZoneScopedDetector` — all **concrete, single-implementation**. Each has one sensible implementation; wrapping them in ABCs is ceremony, not modularity. Similarly, the orchestrator (`backbone/runtime/orchestrator.py`) is the *only* place `registry.create()` runs — plugins never instantiate each other.

## Process boundaries are contracts

The Backbone has **zero imports** from modules; modules share **schemas, never code**. Evolution = expand `schemas.py` additively (principle #5).

| Contract | File / surface | Carries |
|---|---|---|
| **Backbone wire (UDP/JSON)** | `backbone/comms/schemas.py` (`SCHEMA_VERSION = 6`) | isistream → engine: `DetectionSetMessage` (:9010); engine → consumers: tracks, zone state, passings, observations |
| **isicomms (MQTT in, REST out)** | topics `isiMonitor3D/v1/<node>/...`; REST `GET /nodes /zones /tracks /passings` + `/ui` probe | Any producer publishing versioned MQTT JSON feeds the gateway; consumers poll with a Bearer token. Ports, paths, `ISI_GATEWAY_*` env prefix = **frozen interface** |

## Copy-portable module exports

One command produces a self-contained folder you copy anywhere; the shared core travels as a **wheel** built fresh at export time (nothing vendored, nothing to drift):

```bash
scripts/export_module.sh <isical|isistream|isigen|isidet|isicomms> <dest-dir> [onprem|cloud]
```

### Detachability map (from `docs/REUSE.md`)

| Module | In-repo coupling | Export contents | Launch at destination |
|---|---|---|---|
| **isiGen** `trainer/isiGen/` | none — fully standalone | plain folder copy (`requirements.txt` + `launch.sh`) | `./launch.sh` → Studio :8200 (`ISIGEN_*`) |
| **isidet** `trainer/isidet/` | none — fully standalone | plain folder copy (`isi-train.yml`) | `conda env create -f isi-train.yml` → train scripts |
| **isical** `isical/` + `calibration/` | `calibration/` + `backbone.shared.geometry` + lazy `backbone.ingestion` — all inside the wheel | source + backbone wheel + `setup_multical.sh` + launcher | `./launch.sh` → Studio :8300 (`ISICAL_*`) |
| **isistream** `isistream/` | a backbone application by design — ships **inside** the wheel | launcher + `config.example.yaml` + wheel (no source copy — it would shadow the package) | system GStreamer → `./launch.sh --config config.yaml` |
| **isicomms** `isicomms/` | imports only the light backbone surface (schemas + zones) | compose stack (onprem/cloud) + gateway & backbone wheels + `Dockerfile.portable` (offline build) | `docker compose up -d` (cloud: `./gen-certs.sh` → `.env` → `up -d`) |

!!! note "Honest units"
    isistream **cannot** be reused as a bare folder copy — its imports live in the backbone package; the exported wheel+launcher is the honest unit. Wheels are never committed; re-run the exporter to refresh a deployment.
