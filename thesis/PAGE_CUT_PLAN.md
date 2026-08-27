# Deferred pass: cutting the article from 24 pages to the mandated 15

The UGA recommendations state the limit as a rule, not a guideline: *"The article length should
not exceed 15 pages."* The structure pass of 2026-08-21 deliberately left the length alone
(author's decision: restructure first, cut later). This file records the plan so the cut can be
made in one sitting.

## Where the pages are now (24 total)

| Block | Pages | Note |
|---|---|---|
| Front matter and abstract | 0.4 | abstract is 14 lines of the 15 allowed, do not grow it |
| 1 Introduction | 3.3 | 1.2 alone is about 3.0 of it |
| 2 Materials and Methods | 11.9 | System A about 7.0, System B about 3.6, shared core about 1.3 |
| 3 Results | 4.4 | |
| 4 Discussion | 2.7 | |
| 5 Conclusions | 0.6 | |
| References | 1.6 | 38 entries |

System A outweighs System B by about 1.9 to 1, so the cuts should fall mostly on System A.
That keeps the two systems looking like siblings rather than making System B the appendix.

## Untouchable, whatever the pressure

- The abstract limit, the confidentiality notice, the mandated section names, the journal name
  at the top of page 1.
- The data-provenance paragraph in 3.1.1 (no held-out test split exists; the COCO test folder
  duplicates validation). Every mAP in the article is misreadable without it.
- Every "what has not been measured" statement: the rack feature being switched off in
  production; no labelled counting ground truth, so the 21 % is a configuration delta and not a
  recall; the unexplained zero-polybag result; no trigger latency measured at the controller;
  no accuracy after quantization; no acceptance date; the empty RF-DETR hyper-parameter files;
  the single-seed ablation at its own noise floor with the data-budget confound; the roughly
  100 ms pre-appsink clock offset; the deployed 60 px reprojection gate against a 5 px default.
- The note under the merged detector table saying its rows are not comparable.
- The evidence tables: calibration, detectors, ablation, latency, sessions.

## The cut list, largest yield first

1. **Introduction 1.2, about 1.0 page.** Delete the six third-level headings and rewrite as
   three paragraphs. Keep every citation and every "the literature does X, we do Y" sentence;
   cut the how-it-works exposition (the LoRA algebra, the diffusion forward and reverse
   description, ControlNet's zero-convolution mechanics, the MQTT quality-of-service tutorial,
   the quantization tutorial).
2. **System A methods, about 4.0 pages.** Drop the message-catalogue and topic-tree listings,
   the shared-memory frame bus, the plugin-interface enumeration, the capture-acceptance
   criteria, the ten-step synthetic pipeline detail, the decode-ambiguity story, the SAHI and
   TensorRT bucketing paragraph, the degenerate-synchronizer story. Keep all four displayed
   equations: they are the only real geometry in the article.
3. **System B methods, about 1.3 pages.** Keep the datagram literal, the relay in one sentence
   and the audit log in one sentence. Drop the three relay driver families, the container and
   branch detail, the lock-down tool's five checks, the compression tool's five paths.
4. **Results, about 1.1 pages.** Drop the compression subsection (3.3.4) entirely: its own text
   says no accuracy measurement exists and only file sizes remain, so two clauses elsewhere
   carry it. Convert the execution-provider table and the synthetic-bounds table to prose,
   keeping every number.
5. **Discussion, about 1.2 pages.** Three points per system instead of five and four. Merge the
   17 limitation items to about 12 by grouping, never by deleting an admission.
6. **Figures, 13 to about 7.** Merge the two architecture figures into one two-system diagram;
   merge the two operator-interface screenshots into one two-panel figure, which doubles as a
   resonance device; drop the calibration board montage, the latency chart (the table already
   carries its range), the rack diagram, and the site network drawing.
7. **References, about 0.6 page.** Drop URLs where a DOI exists, and the few references that
   supported only the cut tutorial passages.

That totals roughly 10 pages of recovery against the 9 required, leaving about one page of
slack for two-column float packing, which will consume it.

## Overflow route

`thesis/PLAN.md` already nominates a cited internal technical report for anything that will not
fit. The repository documentation (the MkDocs site, the communications manual, the AGV guide,
the hardware sheet) is the substance; cite it once and point at it from the places above.
Do not invent a document that does not exist.
