"""GET /api/gateway/nodes — proxy to the isi-gateway's /nodes endpoint.

Fetches the cross-warehouse node list from the central gateway and returns it
to the dashboard.  When gateway_url is unset the endpoint returns immediately
with ``{"configured": false, "nodes": []}``.  Every remote-call error is caught
and surfaced as ``{"configured": true, "error": "<msg>", "nodes": []}``; this
route **never** returns an HTTP 500 (errors are part of the normal contract so
the dashboard can show "gateway unreachable" without crashing the poll loop).

The handler is a SYNC ``def`` so its network I/O runs in the FastAPI threadpool
rather than on the async event loop, matching the established monitor_web
convention for blocking endpoints.

**Resolution order for gateway_url / gateway_token:**
  1. UI-settings store (``config/monitor_web_ui.yaml``) — set from Settings →
     Communication so the operator never needs to edit env vars.
  2. Env / pydantic-settings fallback (``MONITOR_WEB_GATEWAY_URL`` /
     ``MONITOR_WEB_GATEWAY_TOKEN``).
  If neither source provides a non-empty URL the endpoint returns
  ``{"configured": false}``.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ..api.routes_config import _read_ui_settings

logger = logging.getLogger(__name__)

router = APIRouter()


def _resolve_gateway(cfg) -> tuple[str | None, str | None]:
    """Return (gateway_url, gateway_token) using ui-settings first, then env fallback."""
    ui = _read_ui_settings(cfg)
    ui_url = ui.get("gateway_url", "") or ""
    ui_token = ui.get("gateway_token", "") or ""
    if ui_url.strip():
        return ui_url.strip(), (ui_token.strip() or None)
    # Fall back to the env/Settings value.
    return cfg.gateway_url, cfg.gateway_token


@router.get("/api/gateway/nodes")
def gateway_nodes(request: Request) -> JSONResponse:
    """Return the node list from the isi-gateway, or a safe fallback.

    Never raises; always HTTP 200.
    """
    cfg = request.app.state.settings

    gateway_url, gateway_token = _resolve_gateway(cfg)

    if not gateway_url:
        return JSONResponse({"configured": False, "nodes": []})

    url = gateway_url.rstrip("/") + "/nodes"
    req = urllib.request.Request(url)
    if gateway_token:
        req.add_header("Authorization", f"Bearer {gateway_token}")

    try:
        with urllib.request.urlopen(req, timeout=cfg.gateway_timeout_s) as resp:
            raw = resp.read()
            data = json.loads(raw)
    except urllib.error.HTTPError as exc:
        # HTTPError is a subclass of URLError — must be caught first.
        short = f"HTTP {exc.code}"
        logger.debug("gateway_nodes: HTTPError fetching %s: %s", url, exc)
        return JSONResponse({"configured": True, "error": short, "nodes": []})
    except urllib.error.URLError as exc:
        short = str(exc.reason) if hasattr(exc, "reason") else str(exc)
        logger.debug("gateway_nodes: URLError fetching %s: %s", url, exc)
        return JSONResponse({"configured": True, "error": short, "nodes": []})
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        short = str(exc)
        logger.debug("gateway_nodes: error fetching %s: %s", url, exc)
        return JSONResponse({"configured": True, "error": short, "nodes": []})

    # Gateway returns {"nodes": [...], "count": N} — normalise to list.
    if isinstance(data, dict):
        nodes = data.get("nodes", [])
    elif isinstance(data, list):
        nodes = data
    else:
        nodes = []

    return JSONResponse({"configured": True, "gateway_url": gateway_url, "nodes": nodes})
