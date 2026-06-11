"""Shared router helpers — project resolution from app state."""

from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException, Request

from ...core.manifest import Manifest
from ...core.project import PROJECT_YAML, ProjectConfig, load_project


def project_dir(request: Request, name: str) -> Path:
    d = Path(request.app.state.settings.data_dir) / name
    if not (d / PROJECT_YAML).exists():
        raise HTTPException(status_code=404, detail=f"project {name!r} not found")
    return d


def project_cfg(request: Request, name: str) -> tuple[Path, ProjectConfig]:
    d = project_dir(request, name)
    return d, load_project(d)


def manifest(request: Request, name: str) -> tuple[Path, Manifest]:
    d = project_dir(request, name)
    return d, Manifest.load(d)
