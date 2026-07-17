# System tree — who lives where

**WHY** — five apps + one shared core in one repo; knowing the boundary lines is knowing the architecture.
**WHAT** — annotated top-two-level tree (verified against the filesystem).
**HOW** — the shared core is `backbone/`; everything else either *feeds* it (isistream, isical), *consumes* it (monitor_web, isicomms), or *trains for* it (trainer/).

```text
isi_monitor3d/
├── backbone/                 # THE SHARED CORE — single source of truth, exported as a wheel
│   ├── core/                 #   registry + types + the 5 ABC seams (interfaces.py)
│   ├── shared/               #   camera_rig, geometry, zones, timestamps, frame_shm (/dev/shm bus)
│   ├── ingestion/            #   FrameSource plugins (rtsp, replay), FrameSynchronizer, points_in (:9010)
│   ├── detection/            #   Detector plugins + zone_scope.py + tiling.py (SAHI) + pre/postprocess
│   ├── homography/           #   foot proj → cross-cam fusion → disagreement gate → ByteTrack-in-m → stabilizer
│   ├── triangulation/        #   subscription-driven DLT + reprojection gate + 3D Kalman
│   ├── comms/                #   schemas.py (THE wire contract, v6) + udp/mqtt sinks + Publisher
│   └── runtime/              #   Orchestrator — the only registry.create() caller; python -m backbone.runtime
├── isistream/                # perception producer (Direction 1): capture → zone-scoped detect → pose → UDP :9010
├── monitor_web/              # operator dashboard (FastAPI :8000): Pixi map, /ws/video, Settings, START/STOP
│   └── monitor_web/          #   app, camera_hub, zone_worker, zone_projection, pose_overlay, isistream_host
├── isical/                   # calibration Studio (FastAPI :8300): capture → solve → export calibration.json
├── calibration/              # calibration backend: Multical wrappers (.venv-multical), Mode-1 single-cam, schema
├── isicomms/                 # MQTT broker + gateway unit
│   ├── isicomms/             #   mqtt_subscriber → in-memory state → REST api/ (+ /ui probe)
│   └── deploy/               #   docker-compose stacks: onprem/ and cloud/ (TLS + Caddy)
├── trainer/
│   ├── isiGen/               # synthetic data Studio :8200 — SDXL + depth ControlNet → YOLO-seg export
│   └── isidet/               # model trainer (isi-train conda env): YOLO-seg configs, data/, runs/, exports
├── config/                   # backbone.yaml (NEVER commit — credentials), zones.yaml, zone_patches.yaml, ...
├── models/                   # .onnx artifacts + .trt_cache (TensorRT engines, built once per shape)
├── scripts/                  # export_module.sh, dataset prep (labelme_to_yolo, build_pallet3_seg, ...)
├── tools/                    # smoke tests: rtsp_smoke, detection_smoke, latency_probe, onnx_inspect, boards
├── tests/                    # 290+ hermetic tests — real pipeline, stub ONNX, loopback UDP
├── docs/                     # REUSE.md, specs/ (cahier des charges), mqtt-architecture.md, comms-manual.md
└── environment.yml           # the monitor3d conda env (Python 3.10, GStreamer, ORT-GPU + TRT)
```

## The two wire contracts (memorize these)

| # | Contract | Direction | Transport | Notes |
|---|---|---|---|---|
| 1 | `DetectionSetMessage` | isistream → engine | UDP loopback `ingestion.points.listen_port` (:9010) | per camera per tick, explicit-empty heartbeat, `seq` gap counting |
| 1 | tracks / zone_state / passings / observations / proximity | engine → consumers | UDP :50001 (+ MQTT for non-display types) | `schemas.py` v6; observations are UDP-only (display concern) |
| 2 | MQTT topics `isiMonitor3D/v1/<node_id>/<suffix>` | node → broker → gateway | MQTT :1883 (TLS in cloud stack) | gateway re-serves as REST :8080 with Bearer token; `/ui` live probe |

!!! note "Environment isolation, on purpose"
    Three Python environments never mix: **`monitor3d`** conda (runtime + tests, OpenCV 4.13 + GStreamer + ORT-GPU), **`calibration/.venv-multical`** (Multical pins opencv-contrib ≤4.7 — would corrupt the runtime env), **`isi-train`** conda (ultralytics/torch pulls `opencv-python`). Same isolation logic as the module exports: boundaries beat shared state.
