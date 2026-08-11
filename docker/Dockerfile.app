# The CPU "app" image — monitor_web dashboard for the CPU deployment branch:
#
#   app → python -m monitor_web   (dashboard; its START button spawns
#         backbone.runtime + isistream INSIDE this container, exactly as on
#         the host — /dev/shm frame bus and the 127.0.0.1 UDP hops all stay
#         in-container). ALL inference is OpenVINO IR on CPU — no CUDA, no
#         onnxruntime, no TensorRT anywhere in this image.
#
# Build from the REPO ROOT (the .dockerignore allowlists what enters):
#
#   docker build -f docker/Dockerfile.app -t isi-monitor3d/app-cpu .
#
# The image bakes the source at /opt/isi (editable-installed). Config /
# models / calibration data are NOT baked — compose mounts the host repo and
# points the apps at it via MONITOR_WEB_* env vars.
FROM mambaorg/micromamba:2.1.1

USER root

# Runtime-only system deps: procps → pgrep (isistream stray-reaper).
RUN apt-get update && apt-get install -y --no-install-recommends \
        procps ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# The conda env — the heavy, rarely-changing layer. Static libs and bytecode
# caches are dead weight at runtime; delete them IN THIS LAYER (a later RUN
# could not reclaim the space).
COPY docker/environment.app.yml /tmp/environment.app.yml
RUN micromamba install -y -n base -f /tmp/environment.app.yml \
    && micromamba clean -a -y \
    && find /opt/conda -name '*.a' -delete \
    && find /opt/conda -name '__pycache__' -type d -prune -exec rm -rf {} +

# Make every following RUN execute inside the activated env.
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# Bake the source at /opt/isi (repo layout preserved so every
# Path(__file__)-relative lookup — monitor_web._REPO_ROOT, calibration
# schema — resolves inside the image).
COPY pyproject.toml README.md /opt/isi/
COPY backbone /opt/isi/backbone
COPY isistream /opt/isi/isistream
COPY calibration /opt/isi/calibration
COPY monitor_web /opt/isi/monitor_web
COPY tools /opt/isi/tools
COPY config/backbone.yaml.example /opt/isi/config/backbone.yaml.example

# Deps are already satisfied by the env → --no-deps keeps pip from dragging in
# the opencv-python PyPI wheel (would collide with conda's opencv).
RUN pip install --no-cache-dir --no-deps -e /opt/isi -e /opt/isi/monitor_web

ENV LD_LIBRARY_PATH=/opt/conda/lib

WORKDIR /opt/isi
EXPOSE 8200

CMD ["python", "-m", "monitor_web"]
