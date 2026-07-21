# Master's Thesis Plan — ISI Monitor 3D (CONFIDENTIAL, Isitec)

**Format (UGA requirements, from the Recommendations PDF):** scientific article, English,
**≤ 15 pages**, abstract ≤ 15 lines, fixed structure (journal name on p.1 → Title → Authors
→ Affiliations → Abstract → Introduction → Materials & Methods → Results → Discussion →
Conclusions → References). **Deadline: August 31** (email to P. Mossuz + masters address).
**Confidentiality notice on page 1.** Defense by Zoom, second half of September; final mark
= mean(thesis, defense).

**Framing (decided):** one integrated **system article** — every module covered
proportionately, depth only where the work is ours.

---

## 1. Title & thesis statement

Working title:

> *A Modular Real-Time Multi-Camera Vision System for Metric 3D Monitoring of
> Warehouse Logistics: Synthetic-Data-Assisted Detection, Dual-Method Geometric
> Localization, and Edge Deployment*
> <!-- "Driven"→"Assisted" 2026-07-21: post-G0/G6, "driven" overclaims (all
> reported detectors trained on real data; synthetic is the assist). -->

Authors: **Abdullahi Adewale ATANDA** (student, UGA) · **Yang NIU** (mentor,
ISITEC International, France). Journal: **Computers in Industry** (confirmed
2026-07-21, no further validation needed).

Thesis statement (the one sentence everything must serve): *a complete industrial
vision pipeline — from operator-guided calibration to MQTT delivery — can meet
sub-200 ms metric 3D monitoring KPIs on a single edge GPU, with a diffusion-based
pipeline that expands ~50 real photos of a class into labeled synthetic data that
cannot replace real-data detector training but is consistent with a small
augmentation benefit (robust effect: recall +0.050).*
<!-- reworded 2026-07-20 per FULL.REVIEW-1 A1: the earlier "trained largely on
synthetic data" version is falsified by G0 (all-real training corpus) + G6
(synthetic-only transfers at 0.22 mAP). -->

Affiliations: Université Grenoble Alpes + ISITEC International, France. Page 1
carries the red confidentiality notice.

## 2. Journal shortlist (mentor validates; name goes on p.1)

1. **Computers in Industry (Elsevier)** — industrial vision systems, applied
   framing fits best. *(recommended)*
2. **Journal of Real-Time Image Processing (Springer)** — if the mentor prefers to
   lead with the < 200 ms real-time engineering.
3. **IEEE Access** — broad-scope, common for system papers, IEEE template.
4. **MDPI Sensors** — multi-camera sensing systems; pragmatic fallback.

## 3. Article skeleton with page budget (15 pp hard cap)

| § | Section | Pages | Content (module → where it appears) |
|---|---|---|---|
| — | Title/authors/affil./abstract | 1.0 | Abstract ≤15 lines: problem (manual warehouse monitoring), limitation (cost of per-site data + multi-camera metric perception), approach (modular 5-part system), validation (real rig + val-set metrics), numbers (p50 40.3 / p95 78.1 ms, ~26 fps — G3; box mAP@0.5 0.977 / mAP@50-95 0.948 val — G1; calib RMS 1.62 px; one 12 GB GPU), contribution list |
| 1 | Introduction + literature | 2.5 | Context: warehouse automation, AGV safety; gap: integrated calibration→detection→metric-3D→comms systems vs isolated papers; related work: multi-camera tracking, homography ground-plane methods, 2-view triangulation, synthetic data (SDXL/ControlNet/LoRA, sim-to-real), YOLO/RF-DETR, edge inference (ONNX Runtime/TensorRT). Ends with numbered contributions (see §4) |
| 2 | Materials & Methods | 5.0 | 2.1 System architecture (0.75 p): five modules + split-process Direction-1 design, decode-once /dev/shm frame bus, points-mode metric engine, wire contracts (fig. 1). 2.2 Calibration — isical (1 p): ChArUco intrinsics → multi-board extrinsics → floor anchor; operator Studio; math of K,D,R,t→H,P. 2.3 Perception — isistream + isidet models (1 p): zone-scoped detection (polygon→per-camera crops), pose, TRT EP. 2.4 Geometric core — backbone (1.5 p): one-calibration-two-queries; per-frame homography chain (foot point→undistort→H→fusion→disagreement gate→ByteTrack-in-meters→stabilizer) + subscription-driven triangulation (DLT, reprojection gate, 3D Kalman, shared identity). 2.5 Synthetic data — isiGen (0.5 p): 10-stage pipeline, SDXL+ControlNet, LoRA r16 on 53 reals → 500 synthetic. 2.6 Comms + deployment — isicomms (0.25 p): UDP/JSON + MQTT schema v6, gateway REST for AGVs; hardware (RTX 5070 dev / Jetson Orin NX target) |
| 3 | Results | 3.5 | See §5 — tables T1–T6 + figures F3–F5, observations only |
| 4 | Discussion | 1.5 | Interpretation: why split-process wins (GIL/ORT contention 55 ms vs 2,200 ms); zone-scoping economics; synthetic-vs-real trade-off; degraded-mode behavior; **Limitations**: val-split metrics vs deployment frames, 2-cam DLT exactly-determined (gate blind to cross-cam disagreement — S5 gotcha), no on-site tape-measure ground truth yet, single-site validation |
| 5 | Conclusions | 0.5 | KPIs met + numbered contributions restated + future work (Jetson port, ≥3-cam aniposelib, pose-mode S5.5) |
| — | References | 1.0 | ~25–35 verified refs (researcher agent: never invented; WebSearch/IEEE/arXiv only) |

## 4. Contribution list (Introduction, numbered)

<!-- MENTOR DECISION: contributions merged 5→1 per review; revert if mentor prefers 5 separate -->
<!-- 2026-07-21: list REPLACED with the author's tightened wording (§1-restructure
     guideline, item 7). Dropped from the list per author (values remain in §3/abstract):
     ~26 fps aggregate; intrinsics 0.60/0.45 px; 53→500 counts; 0.96–0.98 mAP range;
     "explicit failure gating" clause. Prior version retrievable from git history. -->

1. A **modular five-module architecture** with frozen communication contracts,
   enabling single-decode multi-consumer processing and measured real-time
   performance (capture-to-publish latency of 40 ms p50 / 78 ms p95 under full
   production load on one 12 GB edge GPU — live latency measurement, G3 artifact).
2. A **dual-method localization framework** combining always-on ground-plane
   homography and on-demand stereo triangulation from a shared calibration and
   identity space.
3. An **operator-guided calibration workflow** (isical) achieving a 1.62 px
   reprojection consensus on the deployed rig using printed calibration boards only.
4. A **synthetic-data generation pipeline** (isiGen: SDXL + ControlNet + LoRA),
   together with a quantitative evaluation of synthetic versus real training data
   for industrial object detection (G6 ablation, §3.4; detectors trained on the
   ALL-REAL pallet3 corpus reach 0.96–0.98 mAP@0.5 val — kept here as evidence
   context, no longer stated inside the article's contribution list).

## 5. Results section plan — evidence table

### Already measured (usable as-is; sources verified in repo)

| Table/Fig | Content | Source |
|---|---|---|
| T1 Detector accuracy | yolo26l-seg 0.977/0.948 (box mAP@50/50-95); yolo26n-seg\@320 0.962/0.895; RF-DETR medium 0.973/0.938 (best-EMA by val box mAP@0.5:0.95 — the adopted checkpoint rule) + per-class AP; mask mAPs. ALL VAL-SPLIT (no held-out test split exists — G1) | `trainer/isidet/runs/**/report.md`, `results.csv`, `models/rfdetr/*/metrics.csv`, `thesis/measurements/G1_test_split_eval.md` |
| T2 Datasets | pallet3: 3 classes, 5540/1049 splits, ALL-REAL (G0; coco "test" duplicates valid — no held-out split); synthetic provenance (isiGen counts: 500 scaffolds, 553 captions, 500 generated, 267 pass CLIP filter into the export; LoRA r16/768px/2000 steps/53 reals, loss 0.127) | `trainer/isidet/data/*/data.yaml`, `trainer/isiGen/**`, `thesis/measurements/G0_data_provenance.md` |
| T3 Calibration | intrinsic RMS 0.603/0.451 px; extrinsic consensus RMS 1.621 px (KPI ≤ 2 px ✅); board specs; 25 shots/cam, 8 pairs | `isical/data/c1/*` |
| T4 Runtime | p50 40.3 / p95 78.1 / p99 94.0 ms, 25.8 fps aggregate (G3, primary); TRT vs CUDA: 3.0×/2.8× isolated, 2.6×/1.4× e2e, VRAM deltas (G4); whole-GPU 7.6/12.2 GiB under production load (context only — WSL2 has no per-process VRAM). Dev-log prose (p50 77/126, VRAM 2.5-vs-5.2 GB, tick 55-vs-2,200 ms) = pre-campaign July-split measurements → §4 design rationale ONLY, not contributions/abstract | `thesis/measurements/G3_*`, `G4_trt_vs_cuda.md`; `CLAUDE.md` (dev log) |
| T5 Synthetic-accuracy bounds | e2e synthetic: homography ≤1 mm zero-noise / <10 cm @2 px noise; triangulation ≤1 mm; gate 5 px; 721 tests (collected 2026-07-20) | `tests/test_e2e_*.py` |

### Measurement campaign (fills gaps; run before drafting Results — week 2)

| Gap | Experiment | Command / method |
|---|---|---|
| G1 Real-frame detection accuracy | ~~held-out test split eval~~ SUPERSEDED by G0: no held-out split exists (coco test duplicates valid). Ran instead: independent val re-eval (reproduction) of the headline model | `thesis/measurements/G1_test_split_eval.md` |
| G2 Real triangulation reprojection | Run Mode-2 live with subscriptions, log `ReprojectionGate` errors over ≥10 min; report distribution vs 5 px gate | live rig session + orchestrator logs |
| G3 Latency artifact | Produce a probe artifact (p50/p95/p99, n) instead of prose numbers | `python tools/latency_probe.py online --config config/backbone.yaml --seconds 300` → save to `thesis/measurements/` |
| G4 TRT vs CUDA EP table | Same model, both EPs, imgsz 320/640: latency + VRAM (run while live stack is STOPPED — GPU-exclusivity rule) | `tools/detection_smoke.py` timing runs ×N |
| G5 Metric ground truth (stretch) | Tape-measure 5–10 floor positions, compare Track2D/Track3D output | on-site session; if not possible → Limitations |
| G6 Real-only vs real+synthetic training ablation | Train the same small model config (e.g., yolo26n-seg) on the real-only subset vs the merged corpus; compare val mAP — backs contribution 4's synthetic-data claim | isidet in `isi-train` env (external, training-isolation rule) |

Rule for the section (enforced by `researcher`/`cv-reviewer` agents): **observations
separated from interpretation; every number traces to a file or a logged run; no
val-split number presented as deployment accuracy.**

## 6. Figures (target 6; each earns its page space)

1. **F1 System architecture** — five modules + wire contracts (adapt the existing
   cheatsheet topology illustration / `illu.png`).
2. **F2 Geometric pipeline** — the homography chain + triangulation branch with gates.
3. **F3 Calibration** — board layout photo + reprojection-error visualization (Multical viewer export).
4. **F4 isiGen samples** — real photo → scaffold/control map → synthetic variants (confidential-safe).
5. **F5 Latency** — CDF or box plot from G3 probe artifact.
6. **F6 Dashboard** — REINSTATED 2026-07-21 (user request; material now exists):
   live operator UI captured headless during the G3-era production run —
   `thesis/figures/F6_dashboard.png`; callout added in §3.3. No RTSP/IPs visible.

## 7. Writing workflow & tooling

- Drafts live in **`thesis/`** (this folder): `thesis/draft/` (one .md per section →
  assembled), `thesis/measurements/` (campaign artifacts), `thesis/figures/`.
- **`researcher` agent** drafts each section (grounding rules already enforce
  repo-verified numbers, "Measurement needed." otherwise).
- **`cv-reviewer` agent** reviews each draft (structured review: major/minor concerns,
  recommended experiments, verdict) → `researcher` revises. Two full loops minimum
  (after v1 draft, after mentor feedback).
- Final formatting: LaTeX template of the chosen journal (elsarticle / Springer svjour3
  / IEEEtran) once the mentor validates the journal; content stays in markdown until then.
- Confidential: nothing from `thesis/` is published, uploaded, or pushed to public remotes.

## 8. Timeline (today = July 20 → deadline Aug 31)

| Week | Dates | Milestone |
|---|---|---|
| 1 | Jul 20–26 | Journal validated with mentor; skeleton + F1/F2 figures; **Methods draft** (§2.1–2.6) via researcher agent |
| 2 | Jul 27–Aug 2 | **Measurement campaign G1–G4** (G5 if site access); T1–T6 tables assembled from artifacts |
| 3 | Aug 3–9 | Full **draft v1** (all sections + abstract); cv-reviewer **review pass 1**; revise |
| 4 | Aug 10–16 | **Mentor review**; incorporate; references verified (no invented citations) |
| 5 | Aug 17–23 | Journal-template formatting; figures final; cv-reviewer **pass 2**; length ≤ 15 pp enforced |
| 6 | Aug 24–31 | Freeze, proofread, PDF build, **submit by Aug 31** (both email addresses); keep 3-day buffer |
| Sept | 2nd half | Defense prep: 15-min deck from F1–F6 + KPI table; dry run vs cv-reviewer |

## 9. Risks

- **Rig availability** for G2/G5 → fallback: report gate threshold + synthetic bounds,
  move real 3D accuracy to Limitations (already the honest repo state).
- **15-page cap** vs 5 modules → page budget above is binding; anything over goes to
  a cited internal tech report, not the article.
- **Val-only accuracy numbers** → G1 test-split eval is the cheapest credibility fix; do it first.
- **Deadline** → v1 complete end of week 3 leaves two full review cycles.
