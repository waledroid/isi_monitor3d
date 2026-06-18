"""Visual galleries for phases 5-7: scaffold pairs, LoRA runs (report + loss
curve), and the media that backs them. (The mint gallery reuses /records +
/media, since minted images are normal synthetic manifest records.)"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from ...core.project import load_project, resolve_synthesis
from ...core.runners import load_scaffold_index
from .deps import project_dir

router = APIRouter()


# ---- scaffolds (phase 6) ----

@router.get("/api/p/{name}/scaffolds")
async def scaffolds(request: Request, name: str) -> dict:
    d = project_dir(request, name)
    project = load_project(d)
    cfg = project.phase("scaffolds")
    # Resolved synthesis path (copy_paste vs depth) so the page can show which
    # generator will run and why (background count / mode).
    sources, _gen, syn = resolve_synthesis(project, d)
    cp = cfg.get("copy_paste") or {}
    return {"scaffolds": load_scaffold_index(d), "sources": sources,
            "paste_count": cp.get("paste_count", 1),
            "placement": cp.get("placement", "random"),
            "count": int(cfg.get("count", 500)), "synthesis": syn}


@router.get("/api/p/{name}/generation")
async def generation_info(request: Request, name: str) -> dict:
    """Phase-7 settings for the mint page: current inpaint strength + whether the
    resolved path even uses it (only sdxl_inpaint / copy_paste does)."""
    d = project_dir(request, name)
    project = load_project(d)
    gcfg = project.phase("generation")
    _sources, generator, syn = resolve_synthesis(project, d)
    return {"strength": float(gcfg.get("strength", 0.45)),
            "generator": generator, "is_inpaint": generator == "sdxl_inpaint",
            "synthesis": syn}


@router.get("/api/p/{name}/strength-montage")
async def strength_montage(request: Request, name: str):
    """Serve the latest strength-sweep montage (_strength_compare/<project>_montage.png)."""
    d = project_dir(request, name)
    montage = d / "_strength_compare" / f"{d.name}_montage.png"
    if not montage.exists():
        raise HTTPException(status_code=404, detail="no strength montage yet")
    return FileResponse(str(montage))


@router.get("/media/{name}/scaffold/{sid}/{layer}")
async def scaffold_media(request: Request, name: str, sid: str, layer: str):
    if layer not in ("control", "mask"):
        raise HTTPException(status_code=404, detail=f"unknown layer {layer!r}")
    d = project_dir(request, name)
    ids = {e["id"] for e in load_scaffold_index(d)}        # validate id (no traversal)
    if sid not in ids:
        raise HTTPException(status_code=404, detail=f"scaffold {sid!r} not found")
    p = d / "scaffolds" / f"{sid}_{layer}.png"
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"no {layer} for {sid!r}")
    return FileResponse(p)


# ---- LoRA runs (phase 5) ----

def _lora_dir(request: Request, name: str) -> Path:
    return Path(request.app.state.settings.runs_dir) / "lora"


@router.get("/api/p/{name}/lora-runs")
async def lora_runs(request: Request, name: str) -> dict:
    d = project_dir(request, name)                          # 404s on unknown project
    from ...core.project import load_project
    max_steps = int((load_project(d).phase("lora") or {}).get("max_steps", 2000))
    root = _lora_dir(request, name)
    runs = []
    if root.is_dir():
        for dpath in sorted(root.glob(f"{name}_*"), reverse=True):
            if not dpath.is_dir():
                continue
            report = dpath / "report.md"
            runs.append({
                "run": dpath.name,
                "report": report.read_text() if report.is_file() else "",
                "has_plot": (dpath / "loss_curve.png").is_file(),
                "has_weights": (dpath / "pytorch_lora_weights.safetensors").is_file(),
            })
    return {"runs": runs, "max_steps": max_steps}


@router.get("/media/{name}/lora/{run}/plot")
async def lora_plot(request: Request, name: str, run: str):
    project_dir(request, name)
    if not run.startswith(f"{name}_") or "/" in run or ".." in run:
        raise HTTPException(status_code=404, detail=f"run {run!r} not found")
    p = _lora_dir(request, name) / run / "loss_curve.png"
    if not p.exists():
        raise HTTPException(status_code=404, detail="no loss curve for this run")
    return FileResponse(p)
