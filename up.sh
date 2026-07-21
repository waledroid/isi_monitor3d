#!/usr/bin/env bash
# Bring up the full ISI Monitor 3D stack with docker compose.
#
#   ./up.sh                 build (cached) + start everything + show status
#   ./up.sh down            stop the stack
#   ./up.sh logs -f app     ...any other args are passed to `docker compose`
#
# Services: app (dashboard :8000, spawns backbone+isistream on START),
# isical (:8300), mosquitto (:1883), gateway (:8080).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export ISI_UID="$(id -u)"
export ISI_GID="$(id -g)"

# Any argument => raw docker compose passthrough (down / logs / ps / ...).
if [[ $# -gt 0 ]]; then
    exec docker compose "$@"
fi

# --- preflight ---------------------------------------------------------------
command -v docker >/dev/null || { echo "ERROR: docker not found" >&2; exit 1; }
docker compose version >/dev/null 2>&1 || { echo "ERROR: docker compose v2 not found" >&2; exit 1; }
if ! docker info 2>/dev/null | grep -q nvidia; then
    echo "WARNING: NVIDIA runtime not registered with Docker — isistream will have no GPU." >&2
    echo "         Install nvidia-container-toolkit and run: sudo nvidia-ctk runtime configure --runtime=docker" >&2
fi

# Warn when a needed port is held by something OTHER than this stack (a
# natively-run dashboard, the old isicomms/deploy/onprem compose, ...).
# A port clash at `up` leaves half-created containers (no network endpoint).
running="$(docker compose ps --services --status running 2>/dev/null || true)"
port_check() {  # port_check <port> <service> <hint>
    if ! grep -qx "$2" <<<"$running" && ss -ltn 2>/dev/null | grep -q ":$1 "; then
        echo "WARNING: port $1 is already in use by something outside this stack ($3)." >&2
    fi
}
port_check "${MONITOR_WEB_PORT:-8000}"      app       "a natively-run dashboard, e.g. the '3d' alias?"
port_check "${ISICAL_PORT:-8300}"           isical    "a natively-run isical studio?"
port_check "${ISICOMMS_MQTT_PORT:-1883}"    mosquitto "a native broker or the old isicomms/deploy/onprem stack?"
port_check "${ISICOMMS_GATEWAY_PORT:-8080}" gateway   "the old isicomms/deploy/onprem stack?"

# --- build + up --------------------------------------------------------------
docker compose build
docker compose up -d
echo
docker compose ps
echo
echo "Dashboard   http://localhost:${MONITOR_WEB_PORT:-8000}/          (START launches backbone + isistream)"
echo "isical      http://localhost:${ISICAL_PORT:-8300}/          (calibration studio)"
echo "Gateway UI  http://localhost:${ISICOMMS_GATEWAY_PORT:-8080}/ui        (Swagger at /docs)"
echo "MQTT        localhost:${ISICOMMS_MQTT_PORT:-1883}"
echo
echo "Tail logs:  ./up.sh logs -f app     Stop:  ./up.sh down"
