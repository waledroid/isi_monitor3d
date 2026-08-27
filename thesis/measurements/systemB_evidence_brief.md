# System B (IsiDetector) — VERIFIED EVIDENCE BRIEF

Everything below was read off disk in `/home/aatanda/logistic` (branch `main`) and
`/home/aatanda/logistic-fps` (branch `fps`, the site branch, 148 commits ahead of `main`,
2026-04-23 to 2026-07-30). **Use only what is here. Do not add numbers, dates, or claims
that are not in this file. Where this file says NOT MEASURED, the manuscript must say so.**

---

## 1. Setting and problem

- Customer: **Celio** (French clothing retailer). Named in `isidet/configs/inference/common.yaml:33`
  ("the Celio automaticien requires...") and in the deck files `Project Celio*.pptx`.
- Problem statement, verbatim (`mkdocs/docs/index.md:14`): "A camera watches the conveyor belt.
  IsiDetector detects each **carton** (green) and **polybag** (orange) in real time, assigns it a
  track ID, and fires a UDP datagram the instant the parcel's leading edge crosses the counting
  line. The sorter PLC receives the event and actuates the gate, no polling, one packet per parcel."
- Verbatim (`DEMO.md:19`): "A camera watches the conveyor; the system tells the sorter, in real
  time, whether each parcel is a carton or a polybag, so the machine can route it without a human."
- Role in the customer's line (from the emission design spec): IsiDetector is the **backup
  classifier behind Celio's barcode sorter**, consumed "only on a barcode no-read today (Phase A)".
- PLC timing constraint (same spec): camera acceptance window **600 to 1100 ms**
  (`Mini_Camera` / `Maxi_Camera`), "500 ms tolerance", stations spaced about 1100 ms, transit
  about 14 s. Automaticien's instruction, verbatim: "tu choisis un endroit fixe et je gere la
  fenetre de temps de mon cote".
- Design constraints (`DEMO.md:24-26`): must run on "whatever PC the site has (GPU or CPU-only
  Intel box)"; must survive "poor lighting / glare on polybags"; must be "operable by non-ML
  engineers, no Python, no terminal".
- Operator UI is bilingual FR/EN; the automaticien's protocol sheet is FR by default, EN with `--en`.

## 2. What is SHARED with System A (explain once in the common-foundation section, then cross-reference)

Byte-identical files between `logistic/isidet/src` and `isi_monitor3d/trainer/isidet/src`
(verified with `diff -rq`): `shared/registry.py`, `preprocess/clahe_engine.py`,
`inference/base_inferencer.py`, `inference/yolo_inferencer.py`, `inference/rfdetr_inferencer.py`,
`inference/tensorrt_inferencer.py`, `inference/remote_rfdetr_inferencer.py`,
`training/hooks/industrial_logger.py`, `rfdetr_service.py`.

Same lineage, diverged: `inference/{export_engine,onnx_inferencer,openvino_inferencer}.py`,
`shared/vision_engine.py`, `utils/event_logger.py`, `training/{base_trainer,trainers/*}.py`.
Only in System B: `shared/{crossing,dedup_gate,digital_out}.py`. Only in System A:
`inference/pose_onnx_inferencer.py`, `training/hooks/memory_cleanup.py`.

Other shared elements: detector families YOLO26 and RF-DETR; the classes `carton` and `polybag`
(System A's three-class corpus is `palette` plus **the same parcel classes**, its train config
comment records the parcel dataset merged in as classes 1 and 2); ONNX-first export with the raw
head, opset 17; OpenVINO and TensorRT execution paths; ByteTrack; the registry / Strategy pattern;
config-driven YAML design; UDP/JSON as the integration contract; Docker with GPU/CPU auto-detect;
the same RTX 5070 development workstation.

## 3. Data and models (System B)

Datasets (COUNTED on disk, not doc claims):
- `isidet/data/isi_3k_dataset` (raw LabelMe pool): **1,236 jpg, 1,213 json**, flat, unsplit;
  28 images have no label file, 5 labels have no image.
- Master annotation `isidet/data/annotations/coco_instance_segmentation.json`: **1,208 images,
  1,788 annotations, carton 741 / polybag 1,047**; 1,203 images with at least one annotation.
- `isidet/data/universal_dataset` (the corpus the shipped model was trained on):
  **train 2,898 images / 2,898 labels, 12 empty label files, 4,197 instances
  (carton 1,701, polybag 2,496); val 242 images / 242 labels, 1 empty, 389 instances
  (carton 174, polybag 215)**. All labels are polygons.
- `isidet/data/dataset_v2`: train **2,484** (2,033 jpg + 451 png), val **259**; 3,520 train
  instances (carton 1,019, polybag 2,501), 359 val; **120 label-free background images**.
  No build script exists for it; provenance undocumented.
- `isidet/data/rfdetr_dataset`: a COCO view of `universal_dataset` built by symlinks; **every
  symlink is currently broken** (stale absolute paths).
- Build path: `extract_frames.py` (ffmpeg `fps=2`, `scale=1280:-2`, `-q:v 2`) then
  `prepare_data.py` (`random.seed(42)`, 80/20 split, `AUGMENTATION_MULTIPLIER = 2` so each train
  image yields original + 2 augmented copies, Albumentations RandomBrightnessContrast / MotionBlur
  / GaussNoise / ImageCompression / HueSaturationValue). 1,208 master images -> 966 train base
  x3 = 2,898 train, 242 val. This reproduces `universal_dataset` exactly.
- DOC ERROR to avoid repeating: `mkdocs/docs/cheatsheet.md` states 4,968/518 for `dataset_v2` and
  2,899/243 for `universal_dataset`; those counts include Windows `:Zone.Identifier` sidecars.
  Use the counted numbers above.

Training runs (all task=segment; AdamW, batch 16, epochs 200 requested, cos_lr, seed 0):

| Run | pretrained | data | imgsz | epochs run | wall time | best ep | box P | box R | box mAP50 | box mAP50-95 | mask mAP50 | mask mAP50-95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `yolo26n_320_200` (SHIPPED) | yolo26n-seg.pt | universal_dataset | 320 | 196 | 13,065 s (3.63 h) | 166 | 0.913 | 0.935 | 0.950 | 0.915 | 0.943 | 0.840 |
| `yolo26n_416_200` | yolo26n-seg.pt | universal_dataset | 416 | 184 | 12,363 s (3.43 h) | 134 | 0.923 | 0.943 | 0.955 | 0.926 | 0.954 | 0.884 |
| `yolo26m_416_200` | yolo26m-seg.pt | universal_dataset | 416 | 200 | 15,238 s (4.23 h) | 193 | 0.929 | 0.975 | 0.970 | 0.955 | 0.969 | 0.912 |
| `yolo26m_512_200` | yolo26m-seg.pt | universal_dataset | 512 | 200 | 17,030 s (4.73 h) | 199 | 0.936 | 0.946 | 0.963 | 0.950 | 0.962 | 0.916 |
| `yolo26m_640_200` (MISLABELLED) | **yolov8m-seg.pt** | universal_dataset | 640 | 147 | 9,596 s (2.67 h) | 97 | 0.919 | 0.957 | 0.967 | 0.953 | 0.965 | 0.926 |
| `22-06-2026 black` | yolo26n-seg.pt | dataset_v2 | 320 | 171 | 10,508 s (2.92 h) | 151 | 0.940 | 0.825 | 0.863 | 0.849 | 0.859 | 0.780 |

Best epoch above = Ultralytics fitness (what `best.pt` stores) = 0.1*mAP50 + 0.9*mAP50-95 over box
and mask. NOTE: the run directory named `yolo26m_640_200` was actually seeded from **YOLOv8-m**,
not YOLO26-m (its `args.yaml` says `model: yolov8m-seg.pt`). Report it as YOLOv8-m or omit it.

RF-DETR runs (`isidet/models/rfdetr/`), per-class AP exists only here:

| Run | variant | best-EMA epoch | ema mAP50 | ema mAP50-95 | (non-EMA best) AP carton | AP polybag | segm mAP50-95 |
|---|---|---|---|---|---|---|---|
| `24-03-2026_0120` | medium | 10 | 0.967 | 0.953 | 0.928 | 0.972 | 0.941 |
| `25-03-2026_0043` | medium | 7 | 0.969 | 0.953 | 0.928 | 0.970 | 0.944 |
| `10-04-2026_0110` | nano | 43 | 0.969 | 0.951 | 0.929 | 0.964 | 0.926 |
| `15-04-2026_1447` | nano | aborted after 1 epoch | 0.954 | 0.917 | 0.875 | 0.946 | 0.895 |

RF-DETR `hparams.yaml` is empty `{}` in every run, so the actual imgsz / batch / lr of those runs
are NOT recorded; only the configured values exist (`rfdetr_optim.yaml`: 100 epochs, batch 2,
416 px, AdamW 1e-4 head / 1e-5 encoder, EMA on, grad accum 8). No RF-DETR wall time recorded.

DEPLOYED MODEL: on a clean clone of `fps`, `isidet/runs/segment/models/yolo/yolo26n_320_200/
weights/openvino/model.xml` (the only model tracked in git). Confirmed by the committed
`settings.json` on `fps` for both web backends.

## 4. Perception and preprocessing (System B)

- Single camera, RTSP. Deployed URL form `rtsp://admin:admin123@192.168.1.108:554/user=admin&
  password=admin123&channel=1&stream=0.sdp?` (main/high-res stream). Camera native resolution,
  fps and codec were NEVER recorded; the ROI coordinates imply a sensor at least about 2304x1620.
- `LiveReader`: `Queue(maxsize=1)`, RTSP prefers TCP with UDP fallback, 5 s open timeout, 1 s
  reconnect loop; file sources paced to native fps.
- ROI: operator drags a rectangle; deployed `roi_points` are `[[632,14],[2282,14],[2282,1620],
  [632,1620]]` (FastAPI copy) / `[[707,5],[2237,5],[2237,1620],[707,1620]]` (local Flask copy).
  Downscale uses `INTER_AREA` so the 320 px the model sees is sharp.
- CLAHE "SpecularGuard" (`preprocess/clahe_engine.py`, byte-identical to System A's copy):
  CLAHE on the **LAB L channel only**, `clip_limit 2.5`, `tile_grid 8x8`, applied after the ROI
  crop and downscale, before inference. Purpose as written: lift deep shadows and stabilise
  polybag glare. **Deployed default is OFF** (see the A/B result below).
- Five inference backends dispatched by file extension: `.engine` TensorRT, `.xml` OpenVINO,
  `.onnx` ONNX Runtime, `.pth` RF-DETR (native, or HTTP to a sidecar inside Docker), `.pt`
  Ultralytics. Allowlists per mode: CPU mode permits only `.xml`/`.onnx` and only the YOLO family,
  and refuses RF-DETR with an operator-facing message.
- OpenVINO **hard-refuses RF-DETR IR** at load time because OpenVINO 2026 mistranslates the
  transformer's Einsum ops; the settings API rejects such a file with HTTP 400.
- ONNX preload: a throwaway CUDA session is built at boot to warm the cuDNN kernel cache, then
  discarded. Rationale recorded in code: a shared cross-thread CUDA session caused multi-second
  stream-sync stalls; per-thread rebuild costs about 2 s per swap instead of 30 to 80 s.
  These swap-latency figures are PROSE, not stored measurements.
- Site inference config: `yolo_imgsz 320`, `skip_masks: true`, `skip_traces: true`,
  OpenVINO `performance_hint LATENCY`, `num_streams 1`, `inference_num_threads 8`,
  `max_det 300`, `nms_iou 0.7`, `yolo_conf` 0.7 (Flask) / 0.75 (FastAPI).

## 5. Decision layer (System B) — this is where it diverges from System A

- ByteTrack in **image space** (`sv.ByteTrack`): `track_activation_threshold = conf`,
  `lost_track_buffer = track_buffer` (60), `minimum_matching_threshold = match_thresh`
  (CPU 0.70, GPU 0.85), `frame_rate` calibrated from the measured capture fps when non-zero.
  Tuning history: 0.9 -> 0.7 justified by "at 18 fps with 1 m/s belt and 25 cm parcels,
  frame-to-frame IoU lands in the 70 to 85 % range"; a later raise to 0.85 was reverted for CPU
  because "looser matching lets a finished track absorb the next parcel -> missed crossings".
- Counting line: `line_orientation horizontal`, `line_position 0.71`, `belt_direction
  top_to_bottom` as deployed. The line is rebuilt whenever the frame size changes so it follows
  the ROI crop.
- **Leading-edge anchor.** `_ANCHOR_MAP`: vertical+left_to_right -> CENTER_RIGHT;
  vertical+right_to_left -> CENTER_LEFT; horizontal+top_to_bottom -> BOTTOM_CENTER;
  horizontal+bottom_to_top -> TOP_CENTER. Deployed belt is horizontal top_to_bottom, so the
  anchor is BOTTOM_CENTER. Default `trigger_anchor: leading_edge`; `center` is the opt-in
  alternative. Reason recorded verbatim: the Celio automaticien requires the send at the start of
  the object ("debut de l'objet"), because the PLC's timing window expects the leading edge;
  centre fires later and fell outside the window.
- **CrossingDetector** (`shared/crossing.py`): supervision's `LineZone` fires on the instantaneous
  frame-to-frame side flip of the anchor; at low fps a fast parcel can move more than its own size
  between frames and the flip is missed. The detector instead latches each track once its anchor
  is seen strictly before the line and fires once when the same track is later seen after the line
  in belt order. It is OR-ed with `LineZone` and feeds the same dedup and `seq` path, so a crossing
  caught by both is still counted once. Enabled by `count_interpolate` (deployed: true).
- **DedupGate** (`shared/dedup_gate.py`): emit if and only if (track id not yet counted) AND
  (time guard off OR at least `interval_ms` since the last emit). Track-id layer always on; the
  time guard defaults to **off** (`dedup_time_enabled: false`, `dedup_interval_ms: 300`).
  Pruned at 50,000 entries by keeping the newer half.
- **`seq`**: a monotonic counter incremented only after dedup, so suppressed crossings never
  consume a number. It is gap-free by construction, therefore any gap the receiver sees is a
  genuinely lost datagram. The same value is written to the event CSV and sent on the wire, so the
  log can be reconciled against what was received. Resets to 1 on stream restart.

## 6. Delivery and integration (System B)

- UDP datagram, exactly as built:
  `{"class": "carton", "seq": 17, "id": 42, "ts": "2026-03-31T14:23:45.312847"}`
  (`class` is `carton` or `polybag`; `id` is the ByteTrack id and is NOT sequential; `ts` is ISO
  with microseconds). About 70 bytes. One datagram per crossing, including two parcels crossing in
  the same frame. Single `SOCK_DGRAM` socket created once and reused; `publish()` returns the
  trigger-to-wire latency in nanoseconds.
- Target: deployed default `10.0.0.1:9502`. Configuration priority: `settings.json`, then
  `UDP_HOST`/`UDP_PORT` env, then the YAML, then the hardcoded default; `POST /api/udp` retargets
  live with no restart. INCONSISTENCY worth one honest sentence: the site documents describe the
  PLC at `10.0.0.10` and the network drawing shows 10.0.0.2, 10.0.0.3 and 10.0.0.5, so the
  `10.0.0.1` default is not justified anywhere in the repository.
- Measured emission latency, quoted in the operator sheet the automaticien receives:
  **median 78 us, p99 474 us, worst observed 637 us**, and a stated maximum rate of about
  **3,000 events per hour**. The live histogram (p50/p95/p99/max, thresholds green under 500 us)
  is computed in the performance monitor and served from `/api/performance`, but is NEVER
  persisted to disk.
- Digital output (the electrical twin of the datagram): on every crossing a relay channel is
  driven ON for `pulse_ms` (default 50 ms, validated range 5 to 5000) then OFF, which the PLC
  reads like a photocell. Class-to-channel map `{carton: 1, polybag: 2}`. Drivers: serial
  (USB `/dev/ttyUSB0` or Ethernet-serial `socket://IP:PORT`; protocols numato, lcus, custom
  template), Modbus TCP (function code 05 write-single-coil, the 12-byte echo used as the ack the
  wire path otherwise lacks), and a simulator. A worker thread serialises pulses so a slow write
  never touches the inference loop; the queue is bounded at 64 so a dead device cannot grow memory;
  reconnect every 2 s with log rate-limiting; `test_pulse` fires even while the feature is off so
  the automaticien can commission the wiring. A hardware-free board emulator prints the measured
  pulse widths.
- Event log CSV: header `ts,class,id,seq`, one row per crossing, daily file with midnight rollover
  and 30-day retention, bind-mounted to the host so it survives container recreation ("the audit
  trail the automaticien's loss disputes depend on").
- Web platform: Flask and FastAPI peer backends sharing one inference core and one settings schema;
  FastAPI adds `/ws/video` (binary JPEG) and `/ws/stats` (500 ms JSON) replacing polling. Model
  hot-swap preserves class totals, counted ids, the ByteTrack instance, the line, and the event
  logger; only the inferencer and the palette-indexed annotators are rebuilt. Display JPEG encoding
  is throttled 2:1 while inference, tracking, crossing and UDP still run on every frame.

## 7. Packaging and deployment (System B)

- Two containers plus a docs server: `web` (Flask or FastAPI + ONNX Runtime + Ultralytics +
  OpenVINO, and TensorRT on GPU hosts; ports 9501 TCP and 9502 UDP) and an `rfdetr` sidecar
  (isolated PyTorch + rfdetr, port 9510, GPU profile only, skipped on CPU-only hosts).
  Image sizes recorded in the Dockerfile header: CUDA image about 5 GB, CPU image about 1.2 GB.
- Mode selection: explicit flag, then a deployment marker file written at install, then an
  `nvidia-smi` probe; if GPU mode is selected but the GPU is not visible inside Docker it falls
  back to CPU automatically.
- Branch model: `main` is the full project; a runtime-only branch drops the compression tool, the
  docs, the trainers and the scripts (the initial cut removed 152 files and about 98,000 lines);
  site PCs run the site branch (`fps`).
- Site install (about 60 minutes, per the site checklist): three network interfaces configured one
  at a time (camera subnet 192.168.1.x with the camera at 192.168.1.108, automate subnet 10.0.0.x
  with the site PC at 10.0.0.5 and the PLC documented at 10.0.0.10, plus a DHCP internet link),
  camera connected, remote access installed, photographs sent to the office, office validates the
  stream remotely.
- Headless boot: a systemd oneshot unit (`ExecStart=docker compose up -d`,
  `ExecStop=docker compose down`, `RemainAfterExit=yes`, after `docker.service` and
  `network-online.target`) plus `auto_start: true` in both settings files. Kiosk mode (auto-login,
  fullscreen browser) was implemented and then REMOVED as problematic on site. Recorded failure
  that motivated the rewrite: "because the operator avoided enabling autostart, a PC reboot left
  detection stopped until someone opened the browser and clicked Start".
- Network lock-down tool: freezes the DHCP-issued address into a static NetworkManager profile,
  refuses addresses inside the Docker bridge range, and runs five checks including a **live UDP
  egress probe** (tcpdump on the interface while the container sends a probe datagram). It also
  prints a bilingual FR/EN protocol sheet for the automaticien containing the payload shape, the
  measured latency figures, the maximum rate, three receiver recipes, firewall rules, and a
  three-step validation handshake.
- Remote support: Tailscale plus RustDesk, installed by one script.
- Compression tool (office side only): stages `fp16` (float16 conversion with an op blocklist and
  a repair sweep for orphan casts), `int8` (static QDQ, per-channel, MinMax, calibrated on
  **8 synthetic random samples**), `int8_qdq` (static QDQ, Percentile calibration on **32 real
  images**, DFL head excluded), `sim` (constant folding), `openvino_fp16`. Benchmark and accuracy
  validators exist and print to the terminal but **persist nothing**.

## 8. MEASURED RESULTS available for System B

**8.1 Detector accuracy** — the table in section 3 above (validation split of `universal_dataset`,
242 images, 389 instances). Per-class AP exists only for RF-DETR.

**8.2 Line-placement study** (`webapp/isitec_app/uploads/line_result.txt`, `line_result_le.txt`):
3 site clips x 1,000 frames, OpenVINO `yolo26n_320_200`, conf 0.55, imgsz 320, ByteTrack
(activation 0.55, buffer 60, matching 0.85). Results: **35 tracks, 2,609 detections**;
observations per track median 8.0, mean 74.5, min 1, max 995; 86 % of tracks have at least 2
observations, 77 % at least 3. With the leading-edge anchor the detection height fraction has
p10/median/p90 = **0.70 / 0.71 / 0.72**, and the best line position is **0.71, straddled by 15 of
35 tracks**. With the centre anchor: p10/median/p90 = 0.45 / 0.45 / 0.46, best 0.45, straddled by
16 of 35. Straddle counts per candidate line (leading edge): 0.1 -> 11, 0.2 -> 1, 0.3 -> 2,
0.4 -> 7, 0.5 -> 10, 0.6 -> 10, 0.7 -> 14, 0.8 -> 13, 0.9 -> 12. **This is the empirical basis for
the deployed line position.**

**8.3 Counting A/B study** (2026-06-03, clip `cam_20260602_105526.mp4`, operator ROI, horizontal
line at 0.71, top to bottom). Method note, verbatim in substance: the raw 25 fps clip cannot be
processed in real time on the CPU/OpenVINO path, frames drop non-deterministically ("the same
config gave 61, then 23 cartons"), so the clip was re-timed to a deterministic every-Nth-frame
copy at the same resolution to make counts repeatable. Results:
- Old configuration (ByteTrack default 30 fps, time-dedup on, interpolation off) versus new
  (fps-calibrated, dedup off, interpolation on): **identical counts, 23 and 23 at 12.5 fps,
  9 and 9 at 6.25 fps**. Conclusion recorded: the recall toggles are a safe no-op on cleanly
  processed footage; their benefit is meant for the live frame-drop regime and was not validated
  there.
- **CLAHE ON gives 19 cartons, CLAHE OFF gives 23 cartons** on the same clip and settings,
  repeatable, about +21 %. Conclusion recorded: run carton lines with CLAHE off; the site had it
  enabled. This moved the count more than any tracking toggle.
- **Every run counted 0 polybags**, CLAHE on or off, although the operator confirms polybags cross
  in that clip. Recorded as an OPEN detection or line-placement issue, never resolved.
- Corroborating session telemetry from that night: fps 6.0 to 11.7, average confidence 0.558 to
  0.921, id ratio 2.61 to 4.56, carton counts exactly 19, 9, 9, 9, 9, 23, 23, 23, 23.

**CRITICAL HONESTY PIN:** this A/B has **no labelled ground truth**. The numbers are counts from
one configuration versus another, not counts against a truth tally. The "+21 %" is a relative
count delta, NOT a measured recall. The evaluation harness that would produce a miss rate exists
but was never run against real labelled truth (its example truth values are placeholders), and the
planned hand-labelling of the site clips was never done. The manuscript must state this.

**8.4 Deployed operation** (session log, untracked but on disk):
- 2026-06-16 12:55, 2.01 h, 23.0 fps, 824 carton and 110 polybag, mean confidence 0.825, id ratio 1.95
- 2026-06-16 16:31, 3.17 h, 22.2 fps, 1,613 carton and 357 polybag, mean confidence 0.935, id ratio 2.41
- 2026-06-17 09:17, 2.69 h, 22.9 fps, 1,391 carton and 313 polybag, mean confidence 0.933, id ratio 2.41
- 2026-06-03 10:24, 0.20 h, 21.8 fps, 148 carton and 19 polybag
- 2026-07-24 10:01, 0.02 h, 22.8 fps, 9 carton and 3 polybag
- 2026-08-17 16:13 (office, CPU), 0.55 h, 21.5 fps, 200 carton and 215 polybag, mean confidence 0.952
- id ratio is defined as unique track ids divided by crossings, so it is a measured id-churn figure.
- Event CSV `events_2026-07-24.csv`: 12 crossings in 49 s (9 carton, 3 polybag), `seq` 1 to 12
  gap-free, tracker ids non-monotonic (1, 4, 8, 12, ...), which demonstrates the seq versus id
  distinction on real data.
- Site footage: three consecutive 10-minute captures on 2026-06-02 at 10:55, 11:05 and 11:15.
- A customer-side screenshot exists showing the PLC-side receiver application logging real
  datagrams on **2026-05-13** ("class=carton id=475 ts=2026-05-13T14:03:30.931851" through
  id=488), with "UDP Ecoute sur port 9502" and "UDP Status: UDP Online". The site PC address in
  that log is 192.168.2.49.

**8.5 Artifact sizes (the only compression measurements that exist)** for `yolo26n_320_200`:
FP32 ONNX 10,955,995 B (10.96 MB); FP16 ONNX 5,528,711 B (5.53 MB, ratio 0.505); OpenVINO IR
weights 5,391,622 B; INT8 IR weights 2,780,872 B (2.78 MB, 0.254 of the FP32 ONNX). For a larger
model: FP32 ONNX 109.24 MB, simplified 109.15 MB, INT8 QDQ 28.48 MB (0.261).
One narrated validation: the FP16 conversion halved the file (10 MB to 5.3 MB), 240 of 242 weight
tensors converted, and top-5 confidences matched the FP32 baseline within 2e-4 on a synthetic
input. **No accuracy-after-quantization mAP exists for any stage.**

## 9. THINGS NEVER MEASURED (the manuscript must say so, not paper over)

1. Counting accuracy against labelled ground truth: no miss rate, no recall, no precision for the
   deployed pipeline. The "+21 %" is a count delta between configurations.
2. Why polybags counted zero on the June clip. Open.
3. On-site validation of the recall toggles in the live frame-drop regime they were built for.
4. End-to-end trigger latency measured at the PLC, and compliance with the 600 to 1100 ms window.
   Only the emission-side latency figures (78 us median) exist, and only as quoted values in the
   operator sheet, never as a stored artifact.
5. `seq` transport-loss rate on the real link (the mechanism exists, no receiver-side gap count).
6. Camera native resolution, fps and codec.
7. Any accuracy or speed measurement of the INT8 model on the site PC.
8. A controlled GPU-versus-CPU or backend-versus-backend benchmark. Every throughput figure in the
   project documentation (TensorRT "1.5 to 3x", OpenVINO "2 to 5x", CPU fps tables, per-stage
   latency breakdowns) is an unbacked estimate, and some of them contradict each other. Do not
   quote any of them. There is not a single TensorRT engine file in the repository.
9. Model swap latency (about 2 s, 30 to 80 s before) is narrated, never stored.
10. Any go-live or acceptance date. The pre-site test checklist and the task list are entirely
    unticked. The evidence of production use is circumstantial: multi-hour sessions in June, site
    footage in June, an event CSV in July, the customer-side receiver screenshot in May, and a
    July design document describing on-site operator behaviour in the past tense.
11. Test coverage: one plain-python file with 7 assertions covering the crossing latch only. No
    pytest, no CI, no linting configuration.
