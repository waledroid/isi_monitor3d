# FULL.REVIEW-3 — Verification pass on the author-requested revision (7 corrections)

**Reviewed:** `thesis/MANUSCRIPT.md` (522 lines), `thesis/latex/isimonitor3d.tex` /
`isimonitor3d.pdf` (13 pages, built 2026-07-21), `thesis/draft/references.md`.
**Method:** grep-style compliance checks, PDF text extraction (zero unresolved
references), artifact-by-artifact number verification against
`thesis/measurements/`, `trainer/isidet/` run files, `isical/data/c1/`, and the
repo source; online spot-verification of new references [24]–[28].

**Verdict: MINOR** — all seven corrections are applied in the reader-facing
article (the typeset PDF is fully compliant); the residual findings are
housekeeping in the working files and two accepted-risk observations, none
requiring a content change before submission.

---

## 1. Item-by-item compliance with the 7 requested corrections

### (1) "isiMonitor3d" naming — APPLIED ✅
- Zero occurrences of "ISI Monitor 3D" (or "ISI Monitor") in MANUSCRIPT.md, the
  tex, or the extracted PDF text.
- "isiMonitor3d" appears in the abstract, §1, §2.2, and §5 (4 prose
  occurrences, consistent casing).
- The capital-D form "isiMonitor3D" survives in exactly one place: the literal
  MQTT topic example in §2.7 (`isiMonitor3D/v1/zone_a/track2d/person`). This is
  the permitted exception and it is **correct against the repo** — the deployed
  prefix is `isiMonitor3D/v1/...` (`backbone/comms/mqtt_sink.py:77`,
  `config/backbone.yaml:106`). Changing it would falsify the wire contract.

### (2) De-repetition — APPLIED ✅
- **Five-module enumeration:** the full named/role enumeration appears exactly
  twice — §1 (role form: "operator-guided calibration, a perception producer, a
  metric-geometry engine, a synthetic-data/training pair, and a communication
  gateway") and the head of §2.2 (named form: isical / isistream / backbone
  engine / isicomms / isiGen-isidet, with life-cycle rationale). All other
  mentions ("five-module architecture", abstract's "five modules spanning…")
  are references, not enumerations. Compliant as specified.
- **KPI list:** single full statement in §1 (¶ "This article describes
  isiMonitor3d…"); §2.2 back-refs it ("the five industrial acceptance criteria
  fixed in the customer specification and listed in Section 1"); §3 restates
  each individual target inline exactly where its measurement is compared
  (T1 note: ≥ 0.90; T3: ≤ 2 px; T4: < 200 ms) — the reader never has to hunt.
- **Capture-clock caveat:** full statement once in §2.4 (Capture ¶); §3.3 and
  §4.1 back-ref it while restating the load-bearing ~100 ms figure; Limitation 5
  is a one-liner pointing at §4.1. Good balance — see item 4 below.
- **One calibration, two queries:** full statement at the head of §2.5; §2.3
  back-refs it by name. No third full statement found.

### (3) New §2.1 Background subsection — APPLIED ✅ (quality assessed in §2 below)
- Present, ~2 typeset pages (PDF pages 2–4), nine technology paragraphs plus a
  closing data-flow summary that maps each technology to its Methods subsection.
- References [24]–[28] present in text, references.md, and `isimonitor3d.bib`.

### (4) Design rationale in every Methods subsection — APPLIED ✅
Spot-verified per subsection: §2.2 (split-process rationale + frame-bus
rationale), §2.3 (two-stage solve separation, frozen-K justification, consensus
plane vs single placement, ≥5 Mode-1 pairs for an armable residual gate), §2.4
(newest-frame-only latency bound, motion-gate premise, zone-scope scene prior,
ONNX-not-engine portability), §2.5 (foot-point validity, tracking-in-meters,
on-demand triangulation economics, fail-honestly gates), §2.6 (why isiGen
exists; trainer-env isolation), §2.7 (one transport per consumer class; why the
REST gateway exists). Every "what" now carries a "why".

### (5) High-level→low-level flow — APPLIED ✅
§2.1 explicitly promises "each at a high level first, then the specific
property that motivates its role" and delivers it per paragraph; every Methods
subsection opens with a one-sentence purpose statement before mechanism
("Calibration turns a fresh installation into a metrically usable rig…", "The
metric engine is where pixels become meters…", etc.). §2.1's closing paragraph
gives the pixels→detections→tracks→data order that §§2.2–2.7 then follow.

### (6) Confidentiality notice shortened — APPLIED ✅
One sentence, front matter only ("CONFIDENTIAL — This work describes a private
industrial project of ISITEC International, France."), rendered via `\corres`
in the tex. No other confidentiality prose found in the body.

### (7) G0–G6 campaign labels removed — APPLIED in the PDF ✅ / residue in MANUSCRIPT.md ⚠️
- The tex/PDF contains **zero** G-tokens: both archive-path sentences were
  rewritten to "archived with the measurement artifacts" (tex lines 242, 264,
  396).
- MANUSCRIPT.md still carries two literal artifact paths:
  `thesis/measurements/raw/g1_val_yolo26l/` (line 217) and
  `thesis/measurements/G6_ablation/runs/` (line 295). Both are quotations of
  on-disk measurement filenames — inside the stated exception — but the .md and
  .tex have now drifted (the tex dropped them). Recommend making MANUSCRIPT.md
  match the tex wording so the reassembled source stays the single source of
  truth. **Minor m1.**

## 2. §2.1 Background quality and reference verification

**Structure:** each paragraph is genuinely high-level-first (what the
technology is → what property it contributes → forward pointer to the Methods
subsection that uses it). The closing summary paragraph is a good addition.

**Technical accuracy — checked claim by claim, no errors found:**
- YOLO one-pass/dense-grid framing, NMS description, and YOLO26's NMS-free
  edge orientation: accurate per [12]/[13]/[14].
- DETR set prediction + bipartite matching; RT-DETR's "efficient hybrid encoder
  decoupling intra-scale interaction from cross-scale fusion, uncertainty-minimal
  query selection, decoder-layer speed tuning without retraining": these are the
  actual mechanisms of [16], correctly summarized. RF-DETR as NAS-derived,
  fine-tuning-oriented: matches [17].
- Latent diffusion (denoising in autoencoder latent space) [24]; SDXL as scaled
  backbone + strengthened text conditioning [18]; ControlNet as a trainable
  encoder copy injecting spatial conditioning [19]; LoRA as frozen weights +
  low-rank attention updates, LLM origin hedged [20]. All accurate.
- SAM2 "transformer with streaming memory, point or box prompts, images and
  video": accurate per [25].
- GStreamer element-graph/pads/negotiation framing and the "policy as pipeline
  structure" observation; NVDEC as the dedicated decode engine offloading CPU:
  both accurate and correctly tied to §2.4.
- ONNX Runtime EP model and TensorRT engine-per-GPU-and-shape: accurate; the
  shape-discipline consequence correctly foreshadows the SAHI bucket padding.
- Kalman as recursive minimum-variance estimator with predict/update and the
  graceful-degradation property; SORT recipe; ByteTrack's confidence-split
  second pass. Accurate; the §2.5 forward pointer to the metric-space
  transposition is right.
- Geometry: P = K[R|t], Zhang planar calibration, 8-DoF/4-correspondence
  homography, DLT, fiducials-turn-correspondence-into-detection. All textbook-
  correct.

**Reference spot-verification (online, this session):**
- [24] Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion
  Models," CVPR 2022, arXiv:2112.10752 — confirmed against the arXiv abstract.
- [25] Ravi, Gabeur, Hu et al., "SAM 2: Segment Anything in Images and Videos,"
  arXiv:2408.00714 (2024) — confirmed against the arXiv abstract.
- [26] Kalman 1960, Trans. ASME J. Basic Engineering 82(1):35–45, DOI
  10.1115/1.3662552 — bibliographic details consistent with the ASME record.
- [27] GStreamer, [28] NVIDIA Video Codec SDK/NVDEC — official project URLs
  correct.
- **SAM2 role claim vs repo: VERIFIED.** `trainer/isiGen/src/stages/masking/
  sam2_masker.py` registers a SAM2 masker defaulting to
  `facebook/sam2.1-hiera-small`, with a box-prompted `segment_prompted` path
  (one predict per box) and an automatic-mask-generator fallback. §2.1's
  "detector boxes prompt SAM2 to segment each real photograph's objects" and
  §2.6's "automatic detection (box prompts that seed mask generation)" match
  the implementation.

## 3. Numbers vs artifacts (nothing broke) — ALL VERIFIED ✅

| # | Claim (abstract/§3) | Artifact | Match |
|---|---|---|---|
| 1 | Latency p50 40.3 / p95 78.1 / p99 94.0 ms; 61 heartbeats / 310 s; 25.8 fps; ranges | `G3_summary.md` | exact |
| 2 | ×2.6 margin; prior p50 77 / p95 126 ms as dated config | `G3_summary.md` notes | exact |
| 3 | Calibration consensus RMS 1.621 px both cams | `isical/data/c1/calibration_refined.json` (`reprojection_rms_px: 1.621` ×2) | exact |
| 4 | Intrinsic RMS 0.603 / 0.451 px | `c1/work/intrinsic_rms.json` (0.6034 / 0.4506) | exact (rounded) |
| 5 | YOLO26l-seg 0.950/0.951/0.977/0.948, mask 0.972/0.921; speed 1.2/10.3/1.2 ms; train-time 0.960/0.939, 0.947 | `G1_test_split_eval.md` | exact |
| 6 | T2 per-class rows (palette/carton/polybag) | `G1_test_split_eval.md` per-class table | exact |
| 7 | YOLO26n-seg 0.915/0.899/0.962/0.895, mask 0.953/0.846, best epoch 89/100 | run `results.csv` (best mAP50-95 epoch 89: 0.91521/0.89946/0.96175/0.89495/0.95327/0.84581) | exact |
| 8 | RF-DETR best-EMA epoch 23/41: 0.953/0.933/0.973/0.938, mask 0.962/0.906; non-EMA 0.971/0.930; per-class 0.916/0.905/0.971 | `rfdetr-medium-seg_e41_432px/metrics.csv` | exact |
| 9 | Ablation T6 all three rows; syn-val ≈0.936; deltas +0.021/+0.023/+0.020/+0.050; 0.26/0.50 h; md5 0 overlaps; 238+29=267 CLIP-filtered | `G6_synth_ablation.md` | exact (0.262→0.26, 0.497→0.50) |
| 10 | Provenance: 5,540/1,049; zero synthetic; test = byte-identical dup of valid; 500 generated / 553 items | `G0_data_provenance.md` | exact |
| 11 | TRT vs CUDA T5, all 12 cells + VRAM deltas + ≈41 ms CPU-side + 1.0–1.9 s warm-cache build + CUDA-640 p95 62.6 vs median 16.4 | `G4_trt_vs_cuda.md` | exact |

**Cross-references after the §2 renumbering:** the PDF text contains zero "??".
All hardcoded "Section 2.x" pointers were checked against the new structure
(2.1 Background / 2.2 Architecture / 2.3 Calibration / 2.4 Perception /
2.5 Geometric core / 2.6 isiGen / 2.7 Comms): every occurrence points at the
right subsection (12× "Section 2.4" all perception-related, 6× "Section 2.5"
all geometry/occupancy, etc.). **Figures:** 6 figure files present
(`latex/figures/F1…F6`), 6 labels, all six `\figref`'d from the body
(fig:arch, fig:calibration, fig:pipeline, fig:isigen, fig:latency,
fig:dashboard). Tables T1–T7 all labeled and referenced.

## 4. De-repetition context check — PASS

Each back-reference retains enough local context to be read standalone: §2.2's
KPI back-ref names them as "the five industrial acceptance criteria"; §3.3
restates the ~100 ms clock lag inline rather than only pointing; §2.3 names the
invariant it back-refs; §4.1 restates the pre-campaign 55 ms/2,200 ms rationale
figures with their provenance rather than pointing into a log. No orphaned
pointer found where the reader would need to flip back to parse the sentence.

## 5. Page/word budget — PASS (13 of 15 pages)

Layout: front matter + §1 (pp. 1–2), §2.1 (pp. 2–4, ≈2 pages as requested),
§§2.2–2.7 (pp. 4–8), §3 (pp. 8–11), §4–§5 (pp. 11–12), references (p. 12),
final figure page (p. 13). Two pages of headroom for figure growth at
journal-template time. No section reads thin; §2.1's ByteTrack and geometry
paragraphs are the densest but stay purposeful (every sentence earns a forward
pointer). No bloat flagged.

---

## Remaining items (all minor)

- **m1 — MANUSCRIPT.md / tex drift on archive paths.** MANUSCRIPT.md lines 217
  and 295 retain `raw/g1_val_yolo26l/` and `G6_ablation/runs/`; the tex says
  "archived with the measurement artifacts". Align the .md to the tex.
- **m2 — Stale § pointers in references.md annotations** (internal notes, not
  reader-facing): [7] "adapted in §2.4"→§2.5; [10]/[11] "§2.2"→§2.3; [14]/[17]
  "used in §2.3"→§2.4; [18]/[19]/[20] "§2.5"→§2.6; and the FORMATTING FLAG's
  "as pinned in §2.6"→§2.7. Worth fixing so the working file doesn't mislead a
  future editing pass.
- **m3 (carried from FULL.REVIEW-1 m7, still open by design):** software
  citations [22]/[23] ([27]/[28] now too) need version numbers and access dates
  at journal-template time — already flagged in references.md; extend the flag
  to [27]/[28].
- **Accepted-risk observations (no action requested):** (a) the two remaining
  repository-path mentions in §3's opening ("thesis/measurements/",
  "trainer/isidet/", "isical/data/c1/") are unusual for a journal article and
  will likely be replaced by a data-availability statement at submission;
  (b) RF-DETR "epoch 23 of 41" reads the 0-indexed CSV (epochs 0–40) as 41
  epochs — internally consistent, fine.

## Verdict

**MINOR.** All seven requested corrections are fully applied in the
reader-facing article; every checked number traces exactly to its artifact;
the new Background subsection is technically accurate with verified
references; cross-references and figures are intact at 13/15 pages. The three
minor items above are working-file housekeeping and can be batched into the
next editing pass.
