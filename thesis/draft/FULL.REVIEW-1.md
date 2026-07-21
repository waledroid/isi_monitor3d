# Full peer review — pass 1 with complete evidence set (2026-07-20)

Reviewer role: senior CV researcher / journal editor. Scope: `01_introduction.md`,
`02_methods.md`, `03_results.md`, `references.md`, against PLAN.md and the complete
measurement set (G0, G1, G3, G4, G6 + raw artifacts). First review since the
measurement campaign and the G0-driven corrections.

---

## Summary

The manuscript describes ISI Monitor 3D, a five-module industrial vision system
(calibration, perception producer, metric engine, synthetic-data/training pair,
comms gateway) validated on a deployed two-camera rig. Claimed evidence:
capture→publish p50 40.3 / p95 78.1 ms under full production load (G3),
calibration consensus RMS 1.621 px (c1 rig), detector box mAP@0.5 0.962–0.977 on
the pallet3 validation split (G1 + run artifacts), a TensorRT-vs-CUDA EP
benchmark (G4), a single-class real-vs-synthetic training ablation (G6), and
synthetic geometric verification bounds from the hermetic test suite. The
post-G0 corrections (all-real training corpus, no held-out test split) have been
applied and, on my scan, applied completely. The package is honest and unusually
well-traced; the remaining problems are (a) three numbers in contribution 1 that
never appear in Results, (b) one acceptance KPI (pallet empty/full P/R) that is
stated twice and measured nowhere, (c) an Introduction economics claim that the
system's own G6 ablation undercuts, and (d) an ablation whose headline margin is
inside its own stated noise floor.

---

## Verdicts

| Section | Verdict |
|---|---|
| §1 Introduction | **Minor revision** (economics overclaim; contribution-1 orphan numbers; KPI list omits the triangulation gate) |
| §2 Methods | **Minor revision** (384 vs 320 input size; one clarifying clause on the merged corpus; otherwise ready) |
| §3 Results | **Minor revision** (empty/full KPI must be explicitly de-scoped; two leaked interpretations; give the VRAM/tick numbers a home or move them) |
| **Overall** | **Minor revision — conditional.** Escalates to Major if the three mandatory items (MC-1, MC-2, MC-3 below) are not addressed before submission. |

---

## 1. Cross-section number consistency checklist

Legend: ✓ consistent everywhere checked; ✗ mismatch (locations given).

| # | Number | §1 | §2 | §3 | Artifact | Verdict |
|---|---|---|---|---|---|---|
| 1 | Latency p50 40 / p95 78 (p99 94) ms | 40/78 | — | 40.3/78.1/94.0 | G3 identical | ✓ drafts; ✗ **PLAN §3 abstract row still says "p95 126 ms"** (PLAN.md:45) — stale, will corrupt the abstract when drafted |
| 2 | Throughput ~26 fps aggregate (25.8) | ~26 | — | 25.8 / 13.4–13.8 per cam | G3 25.8 | ✓ drafts; ✗ PLAN T4 "~20 fps" stale |
| 3 | −52 % VRAM (2.5 vs 5.2 GB) | claimed as measured | — | **absent** | **no campaign artifact** (CLAUDE.md prose only, pre-perf-lever config) | ✗ **orphan number** — see MC-2 |
| 4 | Tick 55 ms vs 2,200 ms in-process | claimed as measured | — | **absent** | CLAUDE.md prose only | ✗ **orphan number** — see MC-2 |
| 5 | TRT speedup 3.0×/2.8× isolated, 2.6×/1.4× e2e | — | (no number, good) | matches | G4 identical | ✓; ✗ PLAN T4 "2.1–2.3×" stale (G4 supersedes) |
| 6 | mAP@0.5 range 0.96–0.98 | 0.96–0.98 | — | 0.962–0.977 | ✓ | ✓ |
| 7 | mAP@0.5:0.95 range 0.89–0.95 | ✓ | — | 0.895–0.948 | ✓ | ✓ |
| 8 | yolo26l 0.977/0.948 (report.md 0.947; G1 re-eval 0.948, delta explained) | — | — | ✓ with explanation | G1 ✓ | ✓ |
| 9 | yolo26n@320 0.962/0.895 | — | — | ✓ | results.csv ✓ | ✓ |
| 10 | RF-DETR 0.973/0.938 best-EMA ep 23/41 (non-EMA 0.971/0.930) | — | — | ✓ + rule stated | metrics.csv ✓; PLAN T1 updated ✓ | ✓ (the earlier 0.975/0.933 discrepancy is resolved) |
| 11 | RF-DETR per-class AP 0.916/0.905/0.971 | — | — | ✓ | metrics.csv ✓ | ✓ |
| 12 | Calibration 1.621 px (1.62); intrinsics 0.603/0.451 (0.60/0.45) | 1.62, 0.60/0.45 | gates 0.5/2.0 px | 1.621, 0.603/0.451 | c1 files | ✓ |
| 13 | 25 shots/cam, 8 pairs, 8 floor placements/cam | — | ✓ | ✓ | counted | ✓ |
| 14 | pallet3 5,540/1,049; 1,436 val instances; 3 classes | 5,540 | 5,540/1,049 | ✓ | G0/G1 ✓ | ✓; ✗ minor: PLAN T2 "(/1050)" and G1's header "1,050" vs G0's counted 1,049 — the phantom test-split count; scrub from PLAN |
| 15 | isiGen: 53 reals → 500 generated; 553 captions; 500 scaffolds; LoRA r16/768px/2000 steps/lr 1e-4 | 53→500 | ✓ all | — | G0 ✓ | ✓ drafts; ✗ PLAN T2 "2001 scaffolds" stale (file count, not scaffold count — §2 traceability already flags it; fix PLAN) |
| 16 | Ablation arms 238/238/476; test 186 img / 215 inst | — | — | ✓ | G6 ✓; 238 syn train counted on disk ✓ | ✓ (test set 186/215 also matches T1b polybag row — good) |
| 17 | G6 table (all 24 cells) | — | — | verbatim | G6 identical | ✓ |
| 18 | Deltas +0.021/+0.023/+0.020/+0.050; P 0.922→0.906; syn-val 0.936; syn→real 0.223 | — | — | ✓ | G6 ✓ | ✓ |
| 19 | Reprojection gate 5 px default, 5–8 px allowed | (omitted) | ✓ ×2 | ✓ T5 | reprojection_gate.py | ✓ (see Minor m1 for §1 omission) |
| 20 | Synthetic bounds ≤1 mm / <10 cm@2px / ≤1 mm; 721 tests | — | — | ✓ | tests ✓; PLAN T5 updated ✓ | ✓ |
| 21 | Zone inference input size | — | **"default 384 px"** | **"production input size (320)"** | config/backbone.yaml `zone_imgsz: 320` | ✗ **mismatch** — see MC-4 |
| 22 | G3 conditions: 61 heartbeats, 310 s, n=2048, motion gate on, TRT, full load | — | — | ✓ | G3 ✓ | ✓ |
| 23 | Motion gate 32×32 / >2 % / >15 levels / 2 s refresh; pose stride | — | ✓ | referenced | motion_gate.py (verified pass 1) | ✓ |
| 24 | Fusion 0.8/1.6 m; agreement 0.4/0.8 m; conf 0.5/0.1; skew 33 ms; grace 100 ms | — | ✓ | — | code (verified pass 1) | ✓ |
| 25 | KPI targets: <200 ms p95 / ≤2 px / 5–8 px / mAP ≥0.90 / P/R ≥0.95/0.93 | 4 of 5 | all 5 | 3 verified, **empty/full absent** | CLAUDE.md KPI table | ✗ coverage gap — see MC-1 |

**Net:** the drafts are internally consistent to a degree I rarely see. All
mismatches that remain are (a) PLAN.md staleness (rows 1, 2, 5, 14, 15 — fix
before the abstract is drafted from PLAN's row), (b) the two orphan numbers in
contribution 1 (rows 3–4), and (c) the 384/320 input-size conflict (row 21).

---

## 2. Residual false-claim scan (post-G0)

**Result: clean, with two hardening suggestions.**

- **"Merged corpus trained the reported detectors"** — no trace remains. §1
  contribution 4 attributes accuracy to "the real 5,540-image site corpus";
  §2.5 states the reported detectors trained on the all-real corpus; §3.1 opens
  with the G0 provenance audit. Correct everywhere.
- **Held-out test split** — no residual claim. §3.1 states explicitly that the
  COCO test folder is a byte-identical duplicate of val and that *all* reported
  accuracy is validation-split accuracy. §3.4's "test set" usage is legitimate
  (it *is* held out from all three ablation arms) and the text discloses its
  origin (pallet3 val) in the same sentence. Acceptable as written.
- **Val presented as deployment accuracy** — none found. §3.1's title carries
  "(validation split)", the KPI observation says "on the validation split", §1
  says "(validation)".
- Hardening 1 (§2.5): "the synthetic exports are packaged into a separate
  merged corpus" — append "(not used for any training reported here)". Without
  the clause, a fast reader can still infer usage; the clause costs six words
  and closes the last inferential path to the old claim.
- Hardening 2 (unchecked leakage channel, new): G0 verified the *synthetic
  PNGs* share zero bytes with pallet3, but nobody has checked whether any of the
  **53 real LoRA-source photos** overlap the 186-image G6 test set. If any do,
  the generator saw the test images and Arms S and R+S have soft leakage. One
  hash intersection (53 × 186 files, minutes) closes it; run it and state the
  result in G6 and §3.4.

---

## 3. Results discipline (§3) and the §4 drafting brief

### 3a. Interpretation leaks in §3 (should move to §4 or be reworded as measurement conditions)

- §3.4: "differences of ~0.02 mAP are within single-seed training noise …
  **so the R+S over R margins should be read with that uncertainty**" — the
  bolded clause is interpretation. Keep the noise-floor statement as a
  measurement condition; move the "how to read it" to §4.
- §3.4: "a measured **sim-to-real gap** of ~0.71" — naming the difference is a
  (mild) interpretive act. Acceptable if §4 owns the explanation; alternatively
  "a real-transfer drop of ~0.71 mAP@0.5".
- §3.3: "the figures **reflect steady-state production behavior**" and "medians
  are **therefore the robust statistic**" — borderline; both defensible as
  measurement-condition statements. Leave, but do not add more of this register.
- §3.2: equating "joint extrinsic consensus reprojection RMS 1.621 px" with the
  "homography reprojection error ≤ 2 px" KPI (T3's "assembly gate = KPI") is an
  interpretive mapping — the measured quantity is board-corner BA residual, not
  the runtime floor-projection error. §3 may report the number; §4 must own the
  argument that the BA residual bounds/proxies the homography KPI.

Everything else in §3 is properly observational. The G0 provenance paragraph,
the reproduction paragraph, and the explicit condition statements (motion gate
included in the latency distribution; dashboard load during G4) are exemplary.

### 3b. §4 drafting brief — interpretations §4 MUST make

Per result:

- **G3 latency (T4):** why the split-process + motion-gate + 720p + TRT
  configuration lands at p95 78 ms; what the 77/126→40/78 change decomposes
  into (which lever bought what, to the extent known); why the process split is
  necessary (GIL/ORT contention, 55 vs 2,200 ms tick — with provenance caveat,
  see MC-2); what fraction of headroom remains for the Jetson port.
- **Capture-clock definition:** the KPI clock starts at the appsink callback,
  ~100 ms (RTSP jitter buffer) + decode after the optical event. State plainly
  that optical-event-to-publish is therefore roughly 100 ms larger than the
  reported figures and would still meet the 200 ms KPI only if the KPI is
  understood against the defined clock (as the spec defines it). Do not let the
  "×2.6 margin" stand without this qualification.
- **G4 (T4b):** why TRT's isolated gain (2.8–3.0×) collapses to 1.4× end-to-end
  at 640 (CPU-side letterbox/NMS/mask decode ≈41 ms) and what that implies —
  the production choice of 320 px zone crops is what keeps the EP gain
  realizable; further model speedups are pointless at 640 without moving
  postprocess off the CPU. Also: benchmark ran beside a ~5 GB dashboard;
  medians robust, p95s contaminated — say so.
- **T1/T1b accuracy:** what val-split-only means — checkpoint selection
  (`best.pt`, best-EMA) optimized *on the same split that is reported*, so
  numbers carry optimistic selection bias in addition to lacking a test split;
  per-class support imbalance (carton: 74 images) widens per-class confidence
  intervals; the mAP ≥ 0.90 KPI is met on val, deployment-frame accuracy is
  unmeasured.
- **G6 ablation:** (i) synthetic-only transfers poorly (0.22) — the pipeline's
  current value is as an *augmenter*, not a replacement for real data; discuss
  likely causes (appearance domain gap, single site/camera family, CLIP filter
  limits). (ii) The R+S gains (+0.02 mAP) are within the single-seed noise
  floor; only the recall +0.050 is plausibly outside it — if seeds are not
  re-run (see §4 of this review), the claim must be stated as "consistent with
  a small benefit, not established". (iii) The R+S arm has 2× the training
  images of R — the comparison confounds "synthetic data" with "more data";
  defend the count-matched design as the scarce-real-data scenario, and admit
  737 real images existed (an R-full arm would have separated the confound).
  (iv) Single class, single site, polybag-positive-only test set → no
  false-positive measurement on empty scenes.
- **Mandatory limitations list** (each one sentence minimum):
  1. All detector accuracy is validation-split (no held-out test split exists;
     G0), with checkpoint selection on the same split.
  2. G6 ablation: single seed per arm; margins ≈ noise floor; single class;
     238-image arms; 186-image single-site test set; data-budget confound.
  3. No field metric ground truth: geometric accuracy is synthetic-bound-only
     (T5); no tape-measure campaign (planned G5 not run); no live triangulation
     reprojection distribution (planned G2 not run) — the 5 px gate is a
     configured threshold, not a measured error.
  4. 2-cam DLT is exactly determined → the reprojection gate cannot detect
     cross-camera disagreement at the 3D stage; mitigation is the upstream 2D
     disagreement gate; gate becomes informative only at ≥3 cameras.
  5. Capture-clock definition (above) — the reported latency excludes ~100 ms
     of pre-appsink pipeline.
  6. **Pallet empty/full P/R KPI (≥0.95/0.93) is unvalidated** — the mechanism
     exists (§2.4) but no measurement was made (see MC-1).
  7. All measurements on the dev workstation (RTX 5070, WSL2), not the Jetson
     Orin NX production target; the port is argued portable, not demonstrated.
  8. Single deployment site, single rig, one camera family per finding.
  9. Latency distribution includes motion-gate cached re-emissions — the gated
     (cheap) ticks share the distribution with inferred ticks; worst-case
     always-inferring latency was not isolated.

---

## 4. Ablation robustness (G6) — would I accept it?

**As a headline quantification claim: no. As an honestly-caveated pilot: yes.**
The design is clean where it matters (identical hyperparameters, count-matched
arms, disjoint splits, byte-level leakage check, real-only test set, artifacts
archived), and the caveats are already disclosed. But:

- **Single seed, and the margin is the noise.** +0.021 box mAP@0.5 with a
  self-declared ~0.02 single-seed noise floor is, by the artifact's own
  standard, not a finding. Contribution 4 currently leans on G6 as its
  quantification.
- **Data-budget confound.** R+S (476 imgs) vs R (238 imgs) — see §3b. An
  R-full (737 real) arm would cost ~0.7 h and bound what "just more real data"
  buys.
- **Selection asymmetry (minor):** each arm's `best.pt` is selected on its own
  val split — Arm S selects on *synthetic* val, R on real val. Different
  selection signals across arms; one sentence of disclosure suffices.
- **No FP measurement** on object-free frames (20 background frames exist in
  the isiGen data; a cheap negative-frame eval is possible).

**Cheapest strengthening, and it is worth it:** re-run Arms R and R+S with two
additional seeds (4 runs × 0.26–0.50 h ≈ **1.5–2 h GPU**), report mean ± std
over 3 seeds. Either the +0.02 margin survives (claim becomes defensible) or it
doesn't (the text honestly reports recall as the only robust effect — still a
publishable observation). Arm S re-seeding is optional (its result is 0.22 vs
0.94; no seed will change the conclusion). Second-cheapest: the R-full arm
(+0.7 h) and the 53-source-vs-test-set hash check (minutes). Total < 3 h GPU
for a categorically stronger §3.4. **Do this before submission.**

---

## 5. Reviewer-attack surface — top 3 attacks

**A1 — "Your own ablation contradicts your economics pitch."** §1's Deployment
economics paragraph claims an installation needs "on the order of fifty real
photographs per object class — instead of a per-site data collection and
annotation campaign". The evidence: reported detectors trained on a 5,540-image
real campaign corpus; synthetic-only from 53 reals transfers at 0.22 mAP; even
Arm R uses 238 real images. PLAN's thesis statement ("detectors trained largely
on synthetic data generated from ~50 real photos") is *directly falsified* by
G0+G6 as the description of this system. **The text does not withstand this
attack as written.** Fix: reword the economics paragraph to the supported
claim (synthetic augmentation from ~50 photos measurably improves a detector
trained on modest real data — pending seeds; reducing, not replacing, the
campaign), and rewrite PLAN's thesis statement — it is the one sentence
"everything must serve," and it currently serves a falsified claim. The
decision-relevant missing experiment, if rig time allows: an arm of ~53 real ±
synthetic — that is the actual 50-photo scenario.

**A2 — "You state five acceptance KPIs and verify three."** Empty/full P/R
(≥0.95/0.93): stated in §1 and §2.1, mechanism described in §2.4, measured
nowhere. Triangulation: gate threshold configured, error never measured live.
A results section structured around acceptance targets invites a
requirements-traceability read; the two silent rows will be found. **Partially
withstands** (the triangulation gap is covered by §3.5's honest framing) —
the empty/full gap is not. Fix: MC-1.

**A3 — "Validation-split accuracy with val-selected checkpoints, single site."**
The package withstands this *because it says it first* — G0/§3.1 disclosure is
the strongest part of the manuscript — provided §4 adds the checkpoint-selection
bias sentence and the deployment-frame caveat. Without those, a reviewer writes
the paragraph for you, unkindly.

(Runner-up: G4 measured under a ~5 GB dashboard load — pre-empted by
disclosure, but expect a "rerun on an idle GPU" request; a clean rerun is ~10
minutes if convenient.)

---

## 6. Major concerns (mandatory before submission)

- **MC-1 — Empty/full KPI unvalidated and unacknowledged in §3.** Either
  measure it (even a small labeled clip) or add one explicit sentence in §3
  ("the empty/full classification KPI is not validated in this work") + the
  limitation in §4. Silence is the only unacceptable option.
- **MC-2 — Contribution 1 contains two numbers with no Results home and no
  campaign artifact:** −52 % VRAM (2.5 vs 5.2 GB) and 55 vs 2,200 ms tick.
  Both are pre-campaign prose figures (CLAUDE.md), measured under a different
  configuration than G3's. Options: (a) re-measure VRAM under the G3
  configuration (one `nvidia-smi` session, minutes) and add a T4 row; (b) move
  both numbers out of the contribution list into §4's design-rationale
  discussion with explicit "measured during development, prior configuration"
  provenance. Do not leave "measured" claims in the contribution list that §3
  never substantiates.
- **MC-3 — G6 single-seed margin vs the claim structure** (see §4 above): run
  the 2 extra seeds, or downgrade the §1/§3.4 language to
  "consistent-with-improvement" and lead with the recall effect.
- **MC-4 — 384 vs 320 input size.** §2.3 says crops are batched "to a fixed
  inference size (default 384 px)"; the deployed config and G4 use 320. State
  the deployed value in §2.3 ("default 384 px; 320 px in the deployed
  configuration and all measurements") — otherwise §2 and §3 describe different
  systems.
- **MC-5 — PLAN.md staleness will poison the abstract.** PLAN §3's abstract
  row still carries p95 126 ms; T4 carries 77/126, ~20 fps, TRT 2.1–2.3×; T2
  carries "2001 scaffolds" and the phantom "(/1050)" test split. The abstract
  is drafted from that row in week 3 — fix PLAN now.

## 7. Minor concerns

- m1: §1's KPI list omits the triangulation 5–8 px gate that §2.1 includes —
  align the two lists (4 vs 5 items).
- m2: §2.5 merged-corpus clause — add "(not used for any training reported
  here)" (see §2 of this review).
- m3: §3.4 wording "the 238 CLIP-filtered … generations of the isiGen export"
  — the export holds 267 synthetic (238 train + 29 val); say "the 238 training
  images among the 267 CLIP-filtered generations" or equivalent, so the count
  reconciles with G0's 267.
- m4: G1's header says "1,050 images" for the phantom test split; G0 counts
  1,049. Harmonize the artifacts (and PLAN T2's "(/1050)").
- m5: §3.3 "×2.6 margin" — pair it with the capture-clock caveat in the same
  breath in §4 (already flagged in the brief; noting here so it is not lost).
- m6: Ablation checkpoint-selection asymmetry (arm-local val splits) — one
  disclosure sentence in §3.4 or G6.
- m7: References [22]/[23] need versions + access dates at journal formatting
  (already deferred; carry the flag).
- m8: §3.1 "authoritative by construction" appears twice across draft+artifact
  registers; fine once, tic if repeated in final assembly.

## 8. Recommended experiments (ranked by cost/benefit)

1. G6 seeds ×2 for Arms R and R+S (~1.5–2 h GPU) — converts contribution 4's
   quantification from within-noise to defensible (MC-3).
2. 53-LoRA-source vs 186-test-set hash intersection (minutes) — closes the last
   leakage channel (§2, Hardening 2).
3. VRAM measurement under the G3 configuration (minutes) — repairs MC-2(a).
4. R-full (737 real) arm (~0.7 h) — separates the data-budget confound.
5. Negative-frame FP eval on the 20 background frames (minutes) — addresses the
   polybag-positive-only caveat.
6. (If rig time) ~53-real ± synthetic arm — the actual 50-photo economics
   scenario (A1); otherwise soften the §1 claim.
7. (Optional) idle-GPU G4 rerun — pre-empts the contention objection.

## 9. Final recommendation

**Minor revision, conditional.** The evidence base is now genuinely strong and
the honesty discipline (G0 disclosure, val-only framing, condition statements)
is the package's best defense. The revision is "minor" because every mandatory
fix is either textual or < 2 h of GPU time; it becomes **major** if MC-1
(unmeasured KPI), MC-2 (orphan contribution numbers), or MC-3 (within-noise
ablation margin) reach submission unaddressed — each of those is the kind of
finding that flips a journal reviewer from sympathetic to adversarial.
