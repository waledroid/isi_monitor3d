# System tree — who lives where

**WHY** — boundary lines are the architecture.
**WHAT** — annotated 2-level tree, verified on disk.
**HOW** — `backbone/` is the shared core; the rest feeds it, consumes it, or trains for it.

```text
isi_monitor3d/
├── backbone/                 # SHARED CORE
│   ├── core/                 #   registry + types + the 5 ABC seams
│   ├── shared/               #   camera_rig, geometry, zones, frame_shm (/dev/shm bus)
│   ├── ingestion/            #   rtsp/replay sources, FrameSynchronizer, points_in
│   ├── detection/            #   Detector plugins + zone_scope.py + tiling.py
│   ├── homography/           #   foot proj → fusion → gate → ByteTrack-in-m → stabilizer
│   ├── triangulation/        #   DLT + reprojection gate + 3D Kalman
│   ├── comms/                #   schemas.py + udp/mqtt sinks
│   └── runtime/              #   Orchestrator — python -m backbone.runtime
├── isistream/                # perception producer: capture → detect → pose
├── monitor_web/              # operator dashboard — python -m monitor_web (alias 3d)
│   └── monitor_web/          #   camera_hub, zone_worker, zone_projection, pose_overlay
├── isical/                   # calibration Studio
├── calibration/              # Multical wrappers (.venv-multical), Mode-1 single-cam, schema
├── isicomms/                 # MQTT broker + gateway
│   ├── isicomms/             #   mqtt_subscriber → state → REST api/
│   └── deploy/               #   docker-compose: onprem/ and cloud/ (TLS)
├── trainer/
│   ├── isiGen/               # synthetic-data Studio
│   └── isidet/               # trainer (isi-train env): configs, data/, runs/, exports
├── config/                   # backbone.yaml (credentials — NEVER commit), zones.yaml, ...
├── models/                   # .onnx + .trt_cache (TensorRT engines)
├── scripts/                  # export_module.sh, dataset prep
├── tools/                    # rtsp_smoke, detection_smoke, latency_probe, onnx_inspect
├── tests/                    # 290+ hermetic tests
├── docs/                     # REUSE.md, specs/, mqtt-architecture.md
└── environment.yml           # monitor3d conda env (py3.10, GStreamer, ORT-GPU + TRT)
```

!!! note "Three Python environments, never mixed"
    `monitor3d` conda (runtime), `calibration/.venv-multical` (pins opencv-contrib ≤4.7), `isi-train` conda (ultralytics pulls `opencv-python`).
