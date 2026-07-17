# ISI **Monitor 3D** — the system in one page

**WHY** — know *what* is *where*, in metres, stable identity, real time (Isitec cahier des charges). No cloud, systemd-supervised.

**WHAT** — 1–2 RTSP cameras → `Track2D` (always) + `Track3D` (on-demand), versioned JSON over UDP + MQTT.

**HOW** — Direction-1 split: two processes joined by a UDP loopback contract.

```mermaid
flowchart TD
  subgraph cams[Cameras]
    direction LR
    A[cam_a RTSP H.264]
    B[cam_b RTSP H.265]
  end
  subgraph isistream[isistream — perception producer]
    direction LR
    CAP[capture + decode] --> ZS[zone-scoped detect seg + global pose]
  end
  subgraph engine[backbone.runtime — metric engine, no CUDA]
    direction LR
    SY[FrameSynchronizer] --> HG[homography: foot→floor → fusion → ByteTrack-in-m]
    HG --> TR[triangulation Mode 2, subscription-driven]
    HG --> PB[Publisher fan-out]
    TR --> PB
  end
  subgraph consumers[Consumers]
    direction LR
    MW[monitor_web dashboard :8000]
    GW[isicomms broker+gateway :1883/:8080]
    AGV[AGV / WMS pollers]
  end
  A & B --> CAP
  ZS -- "DetectionSetMessage UDP :9010" --> SY
  CAP -- "/dev/shm frame bus" --> MW
  PB -- "UDP/JSON tracks + observations" --> MW
  PB -- "MQTT isiMonitor3D/v1/<node>/..." --> GW
  GW -- "REST /nodes /zones /tracks + /ui" --> AGV
```

## The four apps

| App | Launch | Role | Port |
|---|---|---|---|
| **isistream** | `python -m isistream --config config/backbone.yaml` | all pixels: capture, decode, `/dev/shm` bus, detect + pose | → :9010 |
| **backbone** | `python -m backbone.runtime --config config/backbone.yaml` | metric engine, no CUDA, ~190 MB | UDP :50001 |
| **monitor_web** | `python -m monitor_web` (alias `3d`) | dashboard; START/STOP spawns both above | :8000 |
| **isicomms** | `docker compose up -d` / `python -m isicomms` | broker + gateway: MQTT in → REST + `/ui` | :1883 / :8080 |

## KPIs

| Indicator | Target | Status |
|---|---|---|
| Latency capture→publish p95 | < 200 ms | p50 77 / p95 126 ms |
| Homography reprojection | ≤ 2 px | e2e green |
| Triangulation gate per view | ≤ 5–8 px | on-rig pending |
| mAP@0.5 | ≥ 0.90 | `pallet3_yolo_seg` |
| Pallet empty/full P / R | ≥ 0.95 / 0.93 | on the wire |

!!! note
    Latency is always vs `frame.capture_ts`, never publish time. Instrument: `tools/latency_probe.py`.

## Modes

| Mode | Cams | Calibration | Output |
|---|---|---|---|
| **1** `single_cam_homography` | 1 | `calibrate single-cam` (≥5 floor points) | `Track2D` only |
| **2** `dual_cam_homography_triangulation` | 2 | Multical joint BA | `Track2D` + `Track3D` |

Camera dies in Mode 2 ⇒ degradation: solo pairs after 100 ms, `Track3D` halts, `track_id`s survive.

## Seven non-negotiables

1. **One calibration, two queries** — `calibration.json` (`K, D, R, t, H, P`) feeds both methods.
2. **One identity space** — ByteTrack owns `track_id`; triangulation never re-IDs.
3. **Subscription, not polling** — `Track3D` per `config/subscriptions.yaml`.
4. **5 plugin seams**, test-pinned — concrete everywhere else.
5. **Contracts, not shared code** — `backbone/comms/schemas.py`.
6. **Fail honestly** — gated outputs, never silent-bad.
7. **Industrial defaults** — systemd, no cloud.
