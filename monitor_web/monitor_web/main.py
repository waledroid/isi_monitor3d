"""Uvicorn entry point — `python -m monitor_web`."""

from __future__ import annotations

import logging

import uvicorn

from .app import create_app
from .config import Settings


class _AccessLogNoiseFilter(logging.Filter):
    """Drop uvicorn access-log lines for the frequently-polled, zero-information
    endpoints so the terminal shows real events instead of a flood of
    ``GET /api/status 200 OK``. The dashboard polls these every 1-5 s."""

    # Every 1-5 s poll the dashboard makes belongs here. This is not only
    # cosmetic: uvicorn access lines are written from the EVENT LOOP, and a
    # frozen/backlogged Windows terminal makes those writes BLOCK — measured
    # 23 s request latency while the console was stuck. Fewer lines = the
    # loop stays decoupled from console health.
    _NOISY_PATHS = (
        "/api/status", "/api/logs", "/api/calibrate/status",
        "/api/zones",            # also covers /api/zones/state
        "/api/zone-patches",     # also covers /api/zone-patches/state
        "/api/gateway/",         # nodes + zones cards
        "/api/ui-settings",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if any(p in msg for p in self._NOISY_PATHS):
            return False
        # Static-asset revalidations (304 Not Modified) are pure noise too.
        return not ("/static/" in msg and " 304" in msg)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    logging.getLogger("uvicorn.access").addFilter(_AccessLogNoiseFilter())
    cfg = Settings()
    app = create_app(cfg)
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="info")


if __name__ == "__main__":
    main()
