"""Health endpoint — never touches the broker."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/healthz")
async def healthz() -> JSONResponse:
    """Liveness probe.  Always 200; no broker dependency."""
    return JSONResponse({"ok": True})
