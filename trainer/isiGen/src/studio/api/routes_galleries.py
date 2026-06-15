"""Visual galleries for phases 5-7: scaffold pairs, LoRA runs (report + loss
curve), and the media that backs them. (The mint gallery reuses /records +
/media, since minted images are normal synthetic manifest records.)"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from ...core.runners import load_scaffold_index
from .deps import project_dir

router = APIRouter()


# ---- scaffolds (phase 6) ----

@router.get("/api/p/{name}/scaffolds")
async def scaffolds(request: Request, name: str) -> dict:
    d = project_dir(request, name)
    return {"scaffolds": load_scaffold_index(d)}


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
    project_dir(request, name)                              # 404s on unknown project
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
    return {"runs": runs}


@router.get("/media/{name}/lora/{run}/plot")
async def lora_plot(request: Request, name: str, run: str):
    project_dir(request, name)
    if not run.startswith(f"{name}_") or "/" in run or ".." in run:
        raise HTTPException(status_code=404, detail=f"run {run!r} not found")
    p = _lora_dir(request, name) / run / "loss_curve.png"
    if not p.exists():
        raise HTTPException(status_code=404, detail="no loss curve for this run")
    return FileResponse(p)
