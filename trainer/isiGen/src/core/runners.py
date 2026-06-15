"""Phase runners — the SINGLE implementation of each phase's batch loop.

Both the CLI scripts (scripts/run_*.py) and the Studio JobRunner call these,
so there is exactly one behavior. Each runner is resumable: it only processes
records whose target field is missing (unless force=True), loads each model
once, and saves the manifest atomically at the end.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from ..stages.captioning.base import CAPTIONERS
from ..stages.control_maps.base import CONTROL_MAP_EXTRACTORS
from ..stages.curate.importer import ingest
from ..stages.masking.base import MASKERS
from .manifest import Manifest
from .project import ProjectConfig, load_project

logger = logging.getLogger(__name__)


def run_curate(project_dir: Path, *, source: str, class_name: str | None = None,
               auto_class: bool = False) -> dict:
    """Phase 1 — see stages/curate/importer.ingest."""
    project = load_project(project_dir)
    return ingest(Path(project_dir), project, Path(source),
                  class_name=class_name, auto_class=auto_class)


def run_control_maps(project_dir: Path, *, stages: list[str] | None = None,
                     force: bool = False) -> dict:
    """Phase 2 (generation side) — depth/canny control maps for every active record.

    ``stages``: subset of the project's configured extractors (default: all)."""
    project = load_project(project_dir)
    project_dir = Path(project_dir)
    cfg = project.phase("control_maps")
    wanted = stages or list(cfg.get("extractors", ["depth_anything_v2", "canny"]))
    manifest = Manifest.load(project_dir)
    counts: dict[str, int] = {}
    for name in wanted:
        extractor = CONTROL_MAP_EXTRACTORS.create(name, **(cfg.get(name) or {}))
        field = f"{extractor.map_name}_map"        # depth_map | canny_map
        todo = [r for r in manifest.active()
                if force or getattr(r, field, None) is None]
        if not todo:
            counts[name] = 0
            continue
        logger.info("maps[%s]: %d image(s) to process", name, len(todo))
        extractor.load()
        done = 0
        try:
            for rec in todo:
                img = cv2.imread(str(project_dir / rec.image))
                if img is None:
                    logger.warning("maps[%s]: unreadable %s — skipped", name, rec.image)
                    continue
                out = extractor.extract(img)
                rel = f"maps/{extractor.map_name}/{rec.id}.png"
                (project_dir / rel).parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(project_dir / rel), out)
                setattr(rec, field, rel)
                manifest.upsert(rec)
                done += 1
        finally:
            extractor.close()
        counts[name] = done
        logger.info("maps[%s]: %d done", name, done)
    manifest.save()
    return counts


def _composite_mask(project: ProjectConfig, shape_hw: tuple[int, int],
                    class_masks: dict[str, np.ndarray]) -> np.ndarray:
    """Paint class masks in project colors (BGR canvas). Paint order = the
    project's classes order, so later classes overwrite on overlap (cartons on
    a palette stay cartons)."""
    canvas = np.zeros((shape_hw[0], shape_hw[1], 3), dtype=np.uint8)
    for spec in project.classes:
        mask = class_masks.get(spec.name)
        if mask is None or not mask.any():
            continue
        r, g, b = spec.color
        canvas[mask] = (b, g, r)
    return canvas


def _pick_main_mask(masks: list[np.ndarray], shape_hw: tuple[int, int]) -> np.ndarray:
    """From SAM2's automatic masks, pick the single *main subject*: the largest
    central object. Near-full-frame masks (background/floor/walls) and tiny
    specks are ignored; among the rest, score = area-times-centrality. Falls back to
    the largest mask if everything is filtered out. Never unions the scene."""
    h, w = shape_hw[:2]
    total = float(h * w) or 1.0
    cx, cy = w / 2.0, h / 2.0
    maxd = (cx * cx + cy * cy) ** 0.5 or 1.0
    best, best_score = None, -1.0
    for m in masks:
        frac = float(m.sum()) / total
        if frac < 0.005 or frac > 0.9:            # speck or whole-frame background
            continue
        ys, xs = np.nonzero(m)
        if xs.size == 0:
            continue
        dist = (((xs.mean() - cx) ** 2 + (ys.mean() - cy) ** 2) ** 0.5) / maxd
        score = frac * (1.0 - dist)               # large AND central (area * centrality)
        if score > best_score:
            best, best_score = m, score
    if best is None:                              # all filtered → largest by area
        best = max(masks, key=lambda m: int(m.sum()))
    return best.astype(bool)


def run_masks(project_dir: Path, *, force: bool = False) -> dict:
    """Phase 2 (ground-truth side) — SAM2 masks, composited to the color-coded PNG.

    Prompted records use their Studio prompts; promptless ones fall back to the
    automatic generator (single-class → assigned; multi-class → needs_review)."""
    project = load_project(project_dir)
    project_dir = Path(project_dir)
    cfg = project.phase("masking")
    masker_name = cfg.get("masker", "sam2")
    masker = MASKERS.create(masker_name, **(cfg.get(masker_name) or {}))
    fallback_auto = bool((cfg.get(masker_name) or {}).get("fallback_auto", True))
    single_class = len(project.classes) == 1

    manifest = Manifest.load(project_dir)
    todo = [r for r in manifest.active() if force or r.mask is None]
    if not todo:
        return {"masked": 0}
    logger.info("masks: %d image(s) to process", len(todo))
    masker.load()
    done = 0
    try:
        for rec in todo:
            img = cv2.imread(str(project_dir / rec.image))
            if img is None:
                logger.warning("masks: unreadable %s — skipped", rec.image)
                continue
            class_masks: dict[str, np.ndarray] = {}
            if rec.mask_prompts:
                class_masks = masker.segment_prompted(img, rec.mask_prompts)
                rec.mask_source = "prompted"
                rec.needs_review = False
            elif fallback_auto:
                auto = masker.segment_auto(img)
                if auto:
                    # Auto-guess the MAIN subject (largest central object), not the
                    # union of every region — that masked the whole scene.
                    main = _pick_main_mask(auto, img.shape[:2])
                    if single_class:
                        class_masks = {project.classes[0].name: main}
                        rec.needs_review = False
                    else:
                        # Multi-class: assign the guess to the record's own class
                        # but flag for operator review (which class is ambiguous).
                        class_masks = {rec.class_name: main}
                        rec.needs_review = True
                rec.mask_source = "auto"
            else:
                continue
            if not class_masks:
                logger.warning("masks: nothing segmented for %s", rec.id)
                continue
            out = _composite_mask(project, img.shape[:2], class_masks)
            rel = f"maps/mask/{rec.id}.png"
            (project_dir / rel).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(project_dir / rel), out)
            rec.mask = rel
            manifest.upsert(rec)
            done += 1
    finally:
        masker.close()
    manifest.save()
    logger.info("masks: %d done", done)
    return {"masked": done}


# ---------------------------------------------------------------------------
# Phases 4-8
# ---------------------------------------------------------------------------

def _scaffold_index_path(project_dir: Path) -> Path:
    return Path(project_dir) / "scaffolds" / "index.jsonl"


def load_scaffold_index(project_dir: Path) -> list[dict]:
    import json
    path = _scaffold_index_path(project_dir)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def save_scaffold_index(project_dir: Path, entries: list[dict]) -> None:
    import json
    import os
    path = _scaffold_index_path(project_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w") as fh:
        for e in entries:
            fh.write(json.dumps(e) + "\n")
    os.replace(tmp, path)


def run_scaffolds(project_dir: Path, *, count: int | None = None) -> dict:
    """Phase 6 — materialize (control map, ground-truth mask) pairs under
    scaffolds/ + an index the generation phase consumes. Count splits evenly
    across the configured sources."""
    from ..stages.scaffolds.base import SCAFFOLD_SOURCES
    project = load_project(project_dir)
    project_dir = Path(project_dir)
    cfg = project.phase("scaffolds")
    gen_cfg = project.phase("generation")
    sources = list(cfg.get("sources", ["box3d_procedural"]))
    total = int(count if count is not None else cfg.get("count", 100))
    per = max(1, total // max(1, len(sources)))
    entries = load_scaffold_index(project_dir)
    seq = len(entries)
    made = 0
    for name in sources:
        sub = dict(cfg.get(name) or {})
        sub.setdefault("width", int(gen_cfg.get("width", 1024)))
        sub.setdefault("height", int(gen_cfg.get("height", 1024)))
        sub["project_dir"] = str(project_dir)        # depth_remix reads the manifest
        source = SCAFFOLD_SOURCES.create(name, **sub)
        for control, mask, meta in source.generate(project, per):
            sid = f"sc{seq:06d}"
            seq += 1
            ctrl_rel = f"scaffolds/{sid}_control.png"
            mask_rel = f"scaffolds/{sid}_mask.png"
            cv2.imwrite(str(project_dir / ctrl_rel), control)
            cv2.imwrite(str(project_dir / mask_rel), mask)
            entries.append({"id": sid, "control": ctrl_rel, "mask": mask_rel,
                            "classes": meta.get("classes", []),
                            "source": name, "status": "pending"})
            made += 1
        logger.info("scaffolds[%s]: %d pair(s)", name, per)
    save_scaffold_index(project_dir, entries)
    return {"scaffolds": made, "total": len(entries)}


def _build_prompt(project: ProjectConfig, classes: list[str], rng) -> str:
    """Phase-7 prompt: every present class's trigger+phrase joined, plus a random
    background from the captioning bank — the same anti-bleed structure the LoRA
    was trained on."""
    cap_cfg = (project.phase("captioning").get("template") or {})
    phrases = cap_cfg.get("class_phrases") or {}
    backgrounds = cap_cfg.get("backgrounds") or ["an industrial environment"]
    parts = []
    for name in classes:
        try:
            spec = project.class_by_name(name)
        except KeyError:
            continue
        parts.append(f"{spec.trigger} {phrases.get(name, name)}")
    subject = " stacked with ".join(parts) if parts else "industrial goods"
    return f"a photo of {subject}, {rng.choice(backgrounds)}"


def run_generation(project_dir: Path, *, limit: int | None = None) -> dict:
    """Phases 5+7 — init the SDXL ControlNet pipeline once (load()), then mint
    one synthetic image per pending scaffold. Each output lands in generated/
    with its prompt, and joins the manifest as a synthetic record whose MASK is
    the scaffold's ground truth (aligned by construction)."""
    import random

    from ..stages.generation.base import IMAGE_GENERATORS
    project = load_project(project_dir)
    project_dir = Path(project_dir)
    cfg = dict(project.phase("generation"))
    name = cfg.pop("generator", "sdxl_controlnet")
    seed_cfg = int(cfg.pop("seed", -1))
    generator = IMAGE_GENERATORS.create(name, **cfg)
    entries = load_scaffold_index(project_dir)
    todo = [e for e in entries if e.get("status") == "pending"]
    if limit:
        todo = todo[: int(limit)]
    if not todo:
        return {"generated": 0}
    manifest = Manifest.load(project_dir)
    rng = random.Random(seed_cfg if seed_cfg >= 0 else None)
    generator.load()                                   # Phase 5
    done = 0
    try:
        for e in todo:
            control = cv2.imread(str(project_dir / e["control"]), cv2.IMREAD_GRAYSCALE)
            if control is None:
                logger.warning("generation: unreadable scaffold %s — skipped", e["id"])
                continue
            prompt = _build_prompt(project, e.get("classes", []), rng)
            seed = seed_cfg if seed_cfg >= 0 else rng.randint(0, 2**31 - 1)
            image = generator.generate(prompt, control, seed=seed)
            rid = f"syn{e['id'][2:]}"
            img_rel = f"generated/{rid}.png"
            cap_rel = f"captions/{rid}.txt"
            (project_dir / "generated").mkdir(exist_ok=True)
            cv2.imwrite(str(project_dir / img_rel), image)
            (project_dir / cap_rel).write_text(prompt + "\n")
            from .manifest import ManifestRecord
            classes = e.get("classes") or [project.classes[0].name]
            manifest.upsert(ManifestRecord(
                id=rid, sha256="", source_path=e["id"], image=img_rel,
                class_name=classes[0],
                width=image.shape[1], height=image.shape[0],
                mask=e["mask"], mask_source="prompted", caption_path=cap_rel,
                synthetic=True, prompt_seed=seed))
            e["status"] = "generated"
            done += 1
            logger.info("generation: %s done (%d/%d) — %s",
                        rid, done, len(todo), prompt[:80])
            save_scaffold_index(project_dir, entries)  # resumable after every image
            manifest.save()
    finally:
        generator.close()
    return {"generated": done}


def run_filter(project_dir: Path, *, force: bool = False) -> dict:
    """Phase 8a — CLIP-score every synthetic record against its prompt; exclude
    those below phases.filtering.min_score (hallucination guard)."""
    from ..stages.filtering.base import QUALITY_FILTERS
    project = load_project(project_dir)
    project_dir = Path(project_dir)
    cfg = dict(project.phase("filtering"))
    name = cfg.pop("filter", "clip_score")
    min_score = float(cfg.pop("min_score", 0.25))
    qf = QUALITY_FILTERS.create(name, **cfg)
    manifest = Manifest.load(project_dir)
    todo = [r for r in manifest.records.values()
            if getattr(r, "synthetic", False)
            and (force or getattr(r, "clip_score", None) is None)]
    if not todo:
        return {"scored": 0, "excluded": 0}
    qf.load()
    scored = excluded = 0
    try:
        for rec in todo:
            img = cv2.imread(str(project_dir / rec.image))
            prompt = ""
            if rec.caption_path and (project_dir / rec.caption_path).exists():
                prompt = (project_dir / rec.caption_path).read_text().strip()
            if img is None or not prompt:
                continue
            s = qf.score(img, prompt)
            rec.clip_score = round(s, 4)               # extra field (extra=allow)
            if s < min_score:
                rec.excluded = True
                excluded += 1
            manifest.upsert(rec)
            scored += 1
    finally:
        qf.close()
    manifest.save()
    logger.info("filter: %d scored, %d excluded (< %.2f)", scored, excluded, min_score)
    return {"scored": scored, "excluded": excluded}


def run_export(project_dir: Path) -> dict:
    """Phase 8b — package every active record that has BOTH an image and a mask
    (synthetic + optionally the real curated ones) into the configured formats."""
    from ..stages.exporting.base import DATASET_EXPORTERS
    project = load_project(project_dir)
    project_dir = Path(project_dir)
    cfg = dict(project.phase("export"))
    include_real = bool(cfg.pop("include_real", True))
    exporters = cfg.pop("exporters", ["yolo_seg"])
    manifest = Manifest.load(project_dir)
    records = [r for r in manifest.active() if r.mask and r.image
               and (include_real or getattr(r, "synthetic", False))]
    out: dict = {"records": len(records)}
    for name in exporters:
        exporter = DATASET_EXPORTERS.create(name, **cfg)
        root = exporter.export(project, records, project_dir / "export")
        out[name] = str(root)
        logger.info("export[%s]: %d record(s) → %s", name, len(records), root)
    return out


def run_lora(project_dir: Path) -> dict:
    """Phase 4 — train the project LoRA; returns the weights path."""
    from datetime import datetime

    from ..stages.lora.base import LORA_TRAINERS
    project = load_project(project_dir)
    project_dir = Path(project_dir)
    cfg = dict(project.phase("lora"))
    name = cfg.pop("trainer", "diffusers_sdxl")
    cfg.setdefault("base_model", project.phase("generation").get("base_model"))
    cfg["project_dir"] = str(project_dir)
    trainer = LORA_TRAINERS.create(name, **cfg)
    isigen_root = Path(__file__).resolve().parents[2]
    run_dir = isigen_root / "runs" / "lora" / (
        f"{project.name}_r{cfg.get('rank', 16)}_"
        f"{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}")
    weights = trainer.train(project, run_dir)
    return {"weights": str(weights)}


def run_captions(project_dir: Path, *, force: bool = False) -> dict:
    """Phase 3 — caption every active record. NEVER overwrites caption_edited."""
    project = load_project(project_dir)
    project_dir = Path(project_dir)
    cfg = project.phase("captioning")
    name = cfg.get("captioner", "template")
    captioner = CAPTIONERS.create(name, **(cfg.get(name) or {}))
    manifest = Manifest.load(project_dir)
    done = skipped_edited = 0
    for rec in manifest.active():
        if rec.caption_edited:
            skipped_edited += 1
            continue
        if rec.caption_path is not None and not force:
            continue
        text = captioner.caption(rec, project)
        rel = f"captions/{rec.id}.txt"
        (project_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        (project_dir / rel).write_text(text + "\n")
        rec.caption_path = rel
        manifest.upsert(rec)
        done += 1
    manifest.save()
    logger.info("captions: %d written, %d edited (preserved)", done, skipped_edited)
    return {"captioned": done, "preserved_edited": skipped_edited}


# ---------------------------------------------------------------------------
# Per-phase reset — delete a phase's outputs so it can be re-run cleanly
# ---------------------------------------------------------------------------

# Phases whose outputs can be wiped from the Studio (curate is excluded — use
# the gallery's exclude or delete the whole project instead).
RESETTABLE = ("maps", "masks", "captions", "lora", "scaffolds", "generate", "export")


def _rm_dir_files(d: Path) -> int:
    n = 0
    if d.exists():
        for f in d.glob("*"):
            if f.is_file():
                f.unlink()
                n += 1
    return n


def reset_phase(project_dir: Path, phase: str, *, runs_dir: Path | None = None) -> dict:
    """Delete a phase's artifacts (files + manifest fields) so it re-runs clean.

    Mask reset keeps saved SAM2 prompts; caption reset keeps hand-edited
    captions. Returns a small summary of what was cleared."""
    import shutil

    project_dir = Path(project_dir)
    if phase not in RESETTABLE:
        raise ValueError(f"phase {phase!r} is not resettable (one of {RESETTABLE})")
    m = Manifest.load(project_dir)
    cleared = 0

    if phase == "maps":
        files = _rm_dir_files(project_dir / "maps" / "depth") + \
                _rm_dir_files(project_dir / "maps" / "canny")
        for r in m.records.values():
            if r.depth_map or r.canny_map:
                r.depth_map = r.canny_map = None
                cleared += 1
        m.save()
        return {"phase": phase, "files_deleted": files, "records_cleared": cleared}

    if phase == "masks":
        files = _rm_dir_files(project_dir / "maps" / "mask")
        for r in m.records.values():
            if r.mask or r.mask_source or r.needs_review:
                r.mask = None
                r.mask_source = None
                r.needs_review = False        # prompts (r.mask_prompts) kept on purpose
                cleared += 1
        m.save()
        return {"phase": phase, "files_deleted": files, "records_cleared": cleared}

    if phase == "captions":
        files = 0
        for r in m.records.values():
            if r.caption_path and not r.caption_edited:      # keep hand-edited
                fp = project_dir / r.caption_path
                if fp.exists():
                    fp.unlink()
                    files += 1
                r.caption_path = None
                cleared += 1
        m.save()
        return {"phase": phase, "files_deleted": files, "records_cleared": cleared}

    if phase == "lora":
        from .project import save_project
        removed = 0
        project = load_project(project_dir)
        if runs_dir is not None:
            for dpath in (Path(runs_dir) / "lora").glob(f"{project.name}_*"):
                shutil.rmtree(dpath, ignore_errors=True)
                removed += 1
        gen = project.phase("generation")
        if gen.get("lora_weights"):
            gen["lora_weights"] = None
            save_project(project_dir, project)
        return {"phase": phase, "runs_deleted": removed}

    if phase == "scaffolds":
        d = project_dir / "scaffolds"
        files = sum(1 for _ in d.glob("*")) if d.exists() else 0
        if d.exists():
            shutil.rmtree(d)
        return {"phase": phase, "files_deleted": files}

    if phase == "generate":                  # minted synthetics
        files = _rm_dir_files(project_dir / "generated")
        syn = [rid for rid, r in m.records.items() if getattr(r, "synthetic", False)]
        for rid in syn:
            del m.records[rid]
        m.save()
        entries = load_scaffold_index(project_dir)
        reset = 0
        for e in entries:
            if e.get("status") == "generated":
                e["status"] = "pending"
                reset += 1
        if reset:
            save_scaffold_index(project_dir, entries)
        return {"phase": phase, "files_deleted": files,
                "synthetic_removed": len(syn), "scaffolds_repending": reset}

    if phase == "export":
        d = project_dir / "export"
        if d.exists():
            shutil.rmtree(d)
        return {"phase": phase, "export_deleted": True}

    return {"phase": phase}                   # unreachable (guarded above)
