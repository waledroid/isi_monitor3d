# System tree — who lives where

**WHY** — boundary lines are the architecture.
**WHAT** — annotated 2-level tree, verified on disk.
**HOW** — `backbone/` is the shared core; the rest feeds it, consumes it, or trains for it.

```text
isi_monitor3d/
├── backbone/                 # SHARED CORE — exported as a wheel
│   ├── core/                 #   registry + types + the 5 ABC seams
│   ├── shared/               #   camera_rig, geometry, zones, frame_shm (/dev/shm bus)
│   ├── ingestion/            #   rtsp/replay sources, FrameSynchronizer, points_in (:9010)
│   ├── detection/            #   Detector plugins + zone_scope.py + tiling.py
│   ├── homography/           #   foot proj → fusion → gate → ByteTrack-in-m → stabilizer
│   ├── triangulation/        #   DLT + reprojection gate + 3D Kalman
│   ├── comms/                #   schemas.py (wire contract v6) + udp/mqtt sinks
│   └── runtime/              #   Orchestrator — python -m backbone.runtime
├── isistream/                # perception producer: capture → detect → pose → UDP :9010
├── monitor_web/              # dashboard (FastAPI :8000): Pixi map, /ws/video, START/STOP
│   └── monitor_web/          #   camera_hub, zone_worker, zone_projection, pose_overlay
├── isical/                   # calibration Studio (FastAPI :8300)
├── calibration/              # Multical wrappers (.venv-multical), Mode-1 single-cam, schema
├── isicomms/                 # MQTT broker + gateway
│   ├── isicomms/             #   mqtt_subscriber → state → REST api/ (+ /ui probe)
│   └── deploy/               #   docker-compose: onprem/ and cloud/ (TLS)
├── trainer/
│   ├── isiGen/               # synthetic data Studio :8200 — SDXL + ControlNet → YOLO-seg
│   └── isidet/               # trainer (isi-train env): configs, data/, runs/, exports
├── config/                   # backbone.yaml (credentials — NEVER commit), zones.yaml, ...
├── models/                   # .onnx + .trt_cache (TensorRT engines)
├── scripts/                  # export_module.sh, dataset prep
├── tools/                    # rtsp_smoke, detection_smoke, latency_probe, onnx_inspect
├── tests/                    # 290+ hermetic tests
├── docs/                     # REUSE.md, specs/, mqtt-architecture.md
└── environment.yml           # monitor3d conda env (py3.10, GStreamer, ORT-GPU + TRT)
```

## The two wire contracts

| Contract | Direction | Transport | Notes |
|---|---|---|---|
| `DetectionSetMessage` | isistream → engine | UDP :9010 | per camera per tick, explicit-empty heartbeat, `seq` |
| tracks / zone_state / passings / observations | engine → consumers | UDP :50001 + MQTT | v6; observations UDP-only |
| MQTT `isiMonitor3D/v1/<node_id>/<suffix>` | node → broker → gateway | :1883 | gateway re-serves as REST :8080, Bearer token |

!!! note "Three Python environments, never mixed"
    `monitor3d` conda (runtime), `calibration/.venv-multical` (pins opencv-contrib ≤4.7), `isi-train` conda (ultralytics pulls `opencv-python`).
