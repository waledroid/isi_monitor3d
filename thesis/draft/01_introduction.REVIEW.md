# Review — §1 Introduction (cv-reviewer pass 1, 2026-07-20)

Reviewed against: `thesis/PLAN.md` (framing, §4 contributions, 2.5-page budget),
`thesis/draft/references.md`, `thesis/draft/02_methods.md` (roadmap consistency),
repo evidence (`isical/data/c1/*`, `CLAUDE.md` KPI/latency facts).
Citations spot-verified this session: [2], [5], [14], [17] (WebFetch/WebSearch);
[4], [6], [7], [11], [12], [13], [15], [16], [20], [21] checked against reviewer
knowledge of the primary sources.

## Summary

The draft frames infrastructure-side metric warehouse monitoring (context), argues an
integration gap — calibration → synthetic-data-trained detection → metric multi-camera
localization → machine-consumable delivery not found described as one measured system —
states three stakes (safety / inventory / deployment economics), lists the five PLAN §4
contributions verbatim, and gives a roadmap matching the actual 5-section skeleton.
1,502 words (budget ~1,700). Flow context → gap → stakes → system → contributions →
roadmap is logical and each literature cluster ends with a scoped, per-cluster gap
statement rather than one broad claim.

## Strengths

- **Gap discipline is above average.** Each gap claim is scoped to the works actually
  cited ("developed and evaluated on fixed benchmark rigs", "these trackers operate in
  the image plane"), and the integration-gap sentence is hedged ("Across the literature
  reviewed here, we did not find…"). The system-paper framing is stated explicitly
  ("this integration gap, rather than any single algorithmic component, is the subject").
- **Citation hygiene.** Spot-checks pass on existence + author/venue/year for all sampled
  entries, including the two post-2025 references: [14] YOLO26 (arXiv 2606.03748, Jocher
  et al., June 2026, NMS-free dual-head, edge-oriented — matches the draft's attribution)
  and [17] RF-DETR (arXiv 2511.09554, Robinson et al., Nov 2025; abstract confirms the
  fine-tuning/target-dataset emphasis the draft attributes to it). [5] MVDet confirmed:
  feature projection to ground plane, evaluated on Wildtrack/MultiviewX — exactly what
  the draft claims.
- **Contribution numbers trace to the repo.** 1.62 px ↔ `calibration_refined.json`
  (`reprojection_rms_px: 1.621` both cams); 0.60/0.45 px ↔ `work/intrinsic_rms.json`
  (0.6034/0.4506); p50 77 / p95 126 ms, −52 % VRAM (2.5 vs 5.2 GB), tick 55 vs
  2,200 ms ↔ CLAUDE.md. The five contributions match PLAN §4 verbatim.
- **Roadmap is consistent** with `02_methods.md` (subsections 2.1–2.6 named correctly)
  and with the PLAN's 5-section skeleton; the Results preview matches T1–T5.
- **Tone.** Banned-adjective scan clean; KPI targets stated as customer acceptance
  criteria, not achievements; observation/claim separation generally respected.

## Major Concerns

**M1 — Citation [2] misattribution ("load analysis").** The draft credits Rutinowski
et al. [2] with identifying "entity re-identification, multi-view localization on the
shop floor, and load analysis." Verified against the paper: its three approaches are
(i) re-identification of logistical entities, (ii) multi-view pose estimation for
tracking/localization, (iii) **category-agnostic segmentation of items in bins for
robotic grasping** — not load analysis. Two of three attributions hold; the third is a
claim the paper does not make. Fix: replace "load analysis" with the paper's actual
third area, or drop it.

**M2 — Contribution 4 makes a causal claim the Results section cannot yet back.**
"A synthetic-data production pipeline … yielding detectors at 0.93–0.95 mAP@0.5:0.95"
attributes the detector accuracy to the synthetic pipeline, but per Methods §2.5 the
detectors are trained on a **merged real+synthetic corpus** (5,540 train images), and no
synthetic-vs-real ablation exists in the PLAN §5 evidence tables or the G1–G5 campaign.
Additional soft spots inside the same sentence: (a) "~50 real photos **per class**" —
repo evidence documents one class (black polybag, 53 reals, 500 generated); palette/
carton provenance is not in T2; (b) "thousands of labeled **instances**" — 500 generated
images are documented; the instance count is not traced anywhere; (c) the 0.93–0.95
range silently excludes yolo26n-seg@320 at 0.892, which T1 will print. Fix options
(pick one): rephrase to attribute the mAP to detectors "trained on the resulting
real+synthetic corpus"; add a small real-only baseline to the G-campaign (the only fix
that makes the causal claim survivable); scope "per class" to what T2 documents. The
"(validation)" qualifier is correctly present — keep it.

**M3 — Reviewer-attack surface (the three most likely hits):**
1. *"System paper — where is the novelty?"* — **Mostly withstood.** The explicit
   integration-gap framing and the measured-numbers-in-contributions style are the
   right defense for Computers in Industry / JRTIP. Residual risk: the dual-method
   claim rests on negative evidence ("an arrangement we did not find described in the
   works above") over only 9 works [3]–[11]; a reviewer who knows one hybrid
   homography+triangulation system lands a hit. Consider leaning on the survey [3]'s
   taxonomy ("surveyed approaches adopt either…") to make the absence claim positive
   and citable.
2. *"Contributions 1 and 5 overlap."* — **Not withstood, but it is a PLAN-level
   issue.** Contribution 1 already contains performance measurements (−52 % VRAM,
   tick cost); contribution 5 is purely a performance result of the architecture in 1.
   A reviewer will call 5 "a result, not a contribution." Since the draft reproduces
   PLAN §4 verbatim (as instructed), do not fix in the draft — raise with the mentor
   whether 5 should be recast as "demonstration that the KPIs are met end-to-end"
   (validation claim) or merged into 1. Second-order variant of the same attack:
   contribution 5's numbers were measured in points mode with the motion gate enabled
   (cached re-emission) — Results must state the measurement conditions or a reviewer
   will ask what "capture→publish" includes.
3. *"ISO standard cited but no compliance claimed."* — **Withstood.** The draft uses
   ISO 3691-4 [1] only to establish that vehicle-centric safety is codified and
   infrastructure-side supervision is complementary; it never claims conformity.
   Keep it that way — do not let later sections drift into "meets ISO" phrasing.

**M4 — KPI/contribution metric mismatch.** The article's stated acceptance target is
mAP@0.5 ≥ 0.90 (¶6), but contribution 4 reports only mAP@0.5:0.95. The number that
directly answers the KPI (mAP@0.5, e.g. 0.977 per T1) never appears in the
Introduction. Either add the mAP@0.5 figure to contribution 4 or ensure Results T1
leads with it; as written, a careful reader cannot check the headline KPI from the
contribution list.

## Minor Concerns

1. **[20] LoRA over-attribution.** The cited paper is about large language models; the
   draft's "LoRA [20] adapts **the generator** to a specific object class from a few
   dozen examples" attributes a diffusion-adaptation claim the paper does not make.
   One clause fixes it: "low-rank adaptation, introduced for language models [20] and
   since applied to diffusion backbones, …" (or add a diffusion-LoRA reference).
2. **[2] URL is dead** (301 → `proc.logistics-journal.de`). Update references.md to
   `https://proc.logistics-journal.de/article/view/1050`.
3. **[9] Zhang 2000 sourcing** via a SCIRP reference page and a personal-site PDF
   mirror looks unprofessional in a journal bibliography. Use the IEEE TPAMI DOI
   (10.1109/34.888718 — verify once before adding) and drop the mirror links.
4. **[22]/[23] software citations** should carry version + access date at
   format time (Methods already pins ORT 1.23.2 / TRT 10.16 — mirror those).
5. Tone nits: "a simple, strong recipe" (colloquial); "The combination is attractive"
   (mildly promotional — "well-suited to per-site training because…" is flatter);
   "changes what such systems cost to roll out" is an economic claim with no
   measurement — acceptable as motivation, but keep it out of Results/Conclusions.
6. Roadmap wording: "Section 2 details the five modules" then enumerates six
   subsections, of which 2.1 (architecture) is not a module. Trivial rewording.
7. Contribution 3 says "on the deployed rig"; the Limitations section must keep the
   matching caveat (single site, 2-cam live validation pending) so the adjective
   "deployed" is not challenged.

## Recommended fixes (priority order)

1. M1: correct the [2] attribution (one clause).
2. M2: rewrite contribution-4 causality ("trained on the resulting real+synthetic
   corpus") + scope "per class"; propose a real-only baseline for the week-2 campaign.
3. M4: add the mAP@0.5 number (or an explicit pointer) so the headline KPI is checkable.
4. M3.2: flag the 1-vs-5 overlap to the mentor before the PLAN freezes the list.
5. Minors 1–3 in references.md (LoRA hedge, dead URL, Zhang DOI).

## Verdict

**Minor revision.** Structure, length, tone, roadmap, and number-traceability are
publication-track; nothing requires re-architecting the section. The two substantive
items are one verified citation misattribution (M1) and one contribution whose causal
phrasing outruns the planned evidence (M2); both are sentence-level fixes plus, ideally,
one cheap added experiment (real-only baseline) that would convert contribution 4 from
the weakest to a defensible one.
