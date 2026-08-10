"""GET /clients — who is consuming the gateway (and how many MQTT clients).

Two sources with different fidelity, reported honestly side by side:

* **REST consumers** — tracked by the gateway itself (middleware in
  ``app.py``): every API request is recorded per client, keyed by the
  ``X-Client-Name`` header when the client sends one (recommended for AGVs,
  e.g. ``X-Client-Name: agv_07``) or by client IP otherwise.
* **MQTT consumers** — Mosquitto only exposes a connected-client *count*
  (``$SYS/broker/clients/connected``); individual client identities are not
  visible to ordinary clients. The count includes the gateway's own
  connection and every publishing Backbone node.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from .auth import require_token

router = APIRouter()

# A REST client is "active" if it made a request within this window.
ACTIVE_WITHIN_S = 30.0


@router.get("/clients", dependencies=[Depends(require_token)])
async def clients(request: Request) -> JSONResponse:
    """List known REST consumers + the broker's connected-client count."""
    store: dict[str, dict] = getattr(request.app.state, "api_clients", {})
    now = time.time()
    rows = [
        {**entry, "active": (now - entry["last_seen"]) <= ACTIVE_WITHIN_S}
        for entry in store.values()
    ]
    rows.sort(key=lambda r: r["last_seen"], reverse=True)
    return JSONResponse({
        "api_clients": rows,
        "count": len(rows),
        "mqtt_connected": request.app.state.subscriber.mqtt_connected(),
    })
