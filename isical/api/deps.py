"""Shared route dependencies."""

from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException, Request

from ..core.project import CALIB_YAML, CalibConfig, load_project


def project_dir(request: Request, name: str) -> Path:
    d = Path(request.app.state.settings.data_dir) / name
    if not (d / CALIB_YAML).exists():
        raise HTTPException(status_code=404, detail=f"calibration {name!r} not found")
    return d


def project_cfg(request: Request, name: str) -> tuple[Path, CalibConfig]:
    d = project_dir(request, name)
    return d, load_project(d)
