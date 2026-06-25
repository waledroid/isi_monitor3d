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
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/api/gateway/nodes")
def gateway_nodes(request: Request) -> JSONResponse:
    """Return the node list from the isi-gateway, or a safe fallback.

    Never raises; always HTTP 200.
    """
    cfg = request.app.state.settings

    if not cfg.gateway_url:
        return JSONResponse({"configured": False, "nodes": []})

    url = cfg.gateway_url.rstrip("/") + "/nodes"
    req = urllib.request.Request(url)
    if cfg.gateway_token:
        req.add_header("Authorization", f"Bearer {cfg.gateway_token}")

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

    return JSONResponse({"configured": True, "gateway_url": cfg.gateway_url, "nodes": nodes})
