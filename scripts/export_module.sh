#!/usr/bin/env bash
# export_module.sh — produce a SELF-CONTAINED, copy-portable folder for one
# module app, launchable in any other project with ./launch.sh (Python apps)
# or docker compose (isicomms). See docs/REUSE.md.
#
#   scripts/export_module.sh <isical|isistream|isigen|isidet|isicomms> <dest-dir> [onprem|cloud]
#
# The shared core (backbone* + calibration* + isistream*) travels as a WHEEL
# built fresh from this checkout into <export>/wheels/ — no vendored source
# copies (no drift), no git access needed at the destination. isiGen/isidet
# have zero repo coupling, so their export is a plain folder copy.
set -euo pipefail

MODULE="${1:?usage: export_module.sh <module> <dest-dir> [onprem|cloud]}"
DEST_ROOT="${2:?usage: export_module.sh <module> <dest-dir> [onprem|cloud]}"
FLAVOR="${3:-onprem}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python3}"

DEST="$DEST_ROOT/$MODULE-portable"
mkdir -p "$DEST"

build_wheels() {   # backbone wheel (+ optionally the isicomms wheel) → $DEST/wheels
  mkdir -p "$DEST/wheels"
  echo "· building isi-monitor3d-backbone wheel (backbone + calibration + isistream)…"
  "$PY" -m pip wheel "$REPO" --no-deps --no-build-isolation -q -w "$DEST/wheels" \
    || "$PY" -m pip wheel "$REPO" --no-deps -q -w "$DEST/wheels"
  if [ "${WITH_GATEWAY:-0}" = "1" ]; then
    echo "· building isicomms wheel…"
    "$PY" -m pip wheel "$REPO/isicomms" --no-deps --no-build-isolation -q -w "$DEST/wheels" \
      || "$PY" -m pip wheel "$REPO/isicomms" --no-deps -q -w "$DEST/wheels"
  fi
}

write_launcher() {  # $1 = command line to exec after venv setup
  cat > "$DEST/launch.sh" << EOF
#!/usr/bin/env bash
# Self-contained launcher — creates .venv and installs requirements on first run.
set -euo pipefail
cd "\$(dirname "\$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  ./.venv/bin/pip install --upgrade pip -q
  if compgen -G "wheels/*.whl" > /dev/null; then
    ./.venv/bin/pip install wheels/*.whl          # the bundled shared core
  fi
  ./.venv/bin/pip install -r requirements.txt
fi
exec $1 "\$@"
EOF
  chmod +x "$DEST/launch.sh"
}

case "$MODULE" in

isical)
  build_wheels
  rsync -a --exclude ".venv*" --exclude "__pycache__" --exclude "runs/" \
        --exclude "data/" "$REPO/isical/" "$DEST/isical/"
  cp "$REPO/calibration/setup_multical.sh" "$DEST/"
  cat > "$DEST/requirements.txt" << 'EOF'
fastapi>=0.115
uvicorn>=0.30
jinja2>=3.1
pydantic>=2.6
pydantic-settings>=2.0
opencv-python>=4.8
numpy>=1.26
pyyaml>=6.0
EOF
  write_launcher './.venv/bin/python -m isical'
  cat > "$DEST/README-DETACH.md" << 'EOF'
# isical (portable)
Copy this folder anywhere → `./launch.sh` → Studio on http://localhost:8300.
- Paths (data dir, backbone.yaml target, mode2 calibration output) are
  configurable via `ISICAL_*` env vars (see isical/config.py) — point them at
  your project's locations.
- Multical extrinsics need the isolated venv once: `./setup_multical.sh`
  (system deps: cmake libopencv-dev libeigen3-dev). ChArUco-only flows don't.
- The `wheels/` wheel carries backbone + calibration from the source repo —
  rebuild it there with scripts/export_module.sh when you want updates.
EOF
  ;;

isistream)
  build_wheels
  cat > "$DEST/requirements.txt" << 'EOF'
opencv-python>=4.8
numpy>=1.26
pyyaml>=6.0
pydantic>=2.6
onnxruntime>=1.17
EOF
  write_launcher './.venv/bin/python -m isistream'
  cp "$REPO/config/backbone.yaml.example" "$DEST/config.example.yaml" 2>/dev/null || \
    cat > "$DEST/config.example.yaml" << 'EOF'
# Minimal isistream config — copy to config.yaml, fill your cameras, run:
#   ./launch.sh --config config.yaml
calibration_path: ./calibration.json
zones_path: ./zones.yaml
cameras:
  cam_a:
    source: { name: rtsp, url: "rtsp://user:pass@camera-ip:554/1", output_wh: [1280, 720] }
detection:
  plugin: yolo_onnx_seg
  onnx_path: ./models/your-model.onnx
  class_names: [palette, carton, polybag]
  confidence_threshold: 0.1
  zone_imgsz: 320
ingestion:
  mode: points
  points: { listen_host: 127.0.0.1, listen_port: 9010 }
EOF
  cat > "$DEST/README-DETACH.md" << 'EOF'
# isistream (portable)
The app code ships INSIDE the wheel (`python -m isistream`) — this folder is
the launcher + config, deliberately without a source copy (it would shadow
the installed package).
1. GStreamer is a SYSTEM dependency (RTSP decode): conda `gstreamer gst-plugins-base
   gst-plugins-good gst-plugins-bad pygobject` or the distro equivalents.
2. `cp config.example.yaml config.yaml`, fill cameras/model/calibration.
3. `./launch.sh --config config.yaml`
It publishes DetectionSetMessages (UDP :9010) + the /dev/shm frame bus —
the same contract the isiMonitor3d metric engine consumes.
EOF
  ;;

isigen)
  rsync -a --exclude ".venv*" --exclude "__pycache__" --exclude "runs/" \
        "$REPO/trainer/isiGen/" "$DEST/"
  ;;

isidet)
  rsync -a --exclude "__pycache__" --exclude "runs/" --exclude "mytest_*" \
        "$REPO/trainer/isidet/" "$DEST/"
  ;;

isicomms)
  WITH_GATEWAY=1 build_wheels
  rsync -a "$REPO/isicomms/deploy/$FLAVOR/" "$DEST/"
  cat > "$DEST/Dockerfile.portable" << 'EOF'
# isicomms gateway image built ONLY from the bundled wheels — no repo checkout,
# no network beyond PyPI for fastapi/uvicorn/paho.
FROM python:3.10-slim
WORKDIR /app
COPY wheels/ /wheels/
RUN pip install --no-cache-dir /wheels/*.whl fastapi uvicorn "paho-mqtt>=1.6,<3" \
    pydantic pydantic-settings pyyaml
EXPOSE 8080
CMD ["python", "-m", "isicomms"]
EOF
  # point the compose at the local portable Dockerfile (context = this folder)
  "$PY" - "$DEST/docker-compose.yml" << 'PYEOF'
import re, sys
p = sys.argv[1]
s = open(p).read()
s = re.sub(r"build:\n\s+context: [^\n]+\n\s+dockerfile: [^\n]+",
           "build:\n      context: .\n      dockerfile: Dockerfile.portable", s)
open(p, "w").write(s)
PYEOF
  cat > "$DEST/README-DETACH.md" << EOF
# isicomms (portable, $FLAVOR)
Self-contained broker + gateway. On the target machine (docker + compose):
$( [ "$FLAVOR" = cloud ] && echo '1. ./gen-certs.sh       # mint CA + server certs
2. cp .env.example .env  # set MQTT_USERNAME/PASSWORD, API_TOKEN
3. docker compose up -d  # mosquitto :8883 (TLS) + gateway (:443 via caddy)' \
  || echo '1. docker compose up -d  # mosquitto :1883 + gateway REST :8080' )

CONTRACT: any producer publishing the versioned MQTT JSON messages of
backbone.comms.schemas (SCHEMA_VERSION 6 — zone_state / tracks / diagnostics /
config) feeds this gateway; consumers poll REST /nodes /zones (Bearer token).
Schema evolution = expand backbone/comms/schemas.py, never share code.
EOF
  ;;

*)
  echo "unknown module: $MODULE (isical|isistream|isigen|isidet|isicomms)" >&2
  exit 2
  ;;
esac

echo "✔ exported → $DEST"
