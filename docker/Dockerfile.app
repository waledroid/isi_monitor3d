# The GPU "app" image — one image, two compose services:
#
#   app     → python -m monitor_web   (dashboard; its START button spawns
#             backbone.runtime + isistream INSIDE this container, exactly as
#             on the dev box — /dev/shm frame bus, NAL sockets and the
#             127.0.0.1 UDP hops all stay in-container)
#   isical  → python -m isical        (calibration studio, :8300)
#
# Build from the REPO ROOT (the .dockerignore allowlists what enters):
#
#   docker build -f docker/Dockerfile.app -t isi-monitor3d/app .
#
# The image bakes the source at /opt/isi (editable-installed) plus the
# isolated Multical venv. Config / models / calibration data are NOT baked —
# compose mounts the host repo (or a site data dir) and points the apps at it
# via MONITOR_WEB_* / ISICAL_* env vars.
FROM mambaorg/micromamba:2.1.1

USER root

# Runtime-only system deps: procps → pgrep (isistream stray-reaper);
# libgl1/libglib2.0-0 → the Multical venv's non-headless opencv-contrib wheel.
# (The apriltags BUILD toolchain is installed + purged in its own layer below.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        procps ca-certificates \
        libgl1 libglib2.0-0 \
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

# GPU inference stack — mirrors the live dev env (pip wheels replaced the
# conda onnxruntime to gain TensorrtExecutionProvider; see CLAUDE.md).
# The tensorrt_libs wheel ships ~2.8 GB of WINDOWS builder resources
# (libnvinfer_builder_resource_win_*.so) inside the Linux wheel — delete them
# in the same layer. The per-arch LINUX resources are all kept so the image
# can build engines on any site GPU (sm75…sm120), not just the dev RTX 5070.
RUN pip install --no-cache-dir \
        onnxruntime-gpu==1.23.2 \
        tensorrt-cu12==10.16.1.11 \
    && rm -f /opt/conda/lib/python3.10/site-packages/tensorrt_libs/libnvinfer_builder_resource_win_*

# GStreamer's CUDA elements (cudaconvertscale — the `decoder: nvdec` chain)
# dlopen the UNVERSIONED "libnvrtc.so"; conda ships only libnvrtc.so.12. On
# the dev box the system CUDA toolkit provides the unversioned name via
# ldconfig — in the image, this symlink is that parity. Without it nvh264dec
# still registers but _nvdec_available() fails and every source silently
# falls back to software decode.
RUN ln -s libnvrtc.so.12 /opt/conda/lib/libnvrtc.so

# Bake the source at /opt/isi (repo layout preserved so every
# Path(__file__)-relative lookup — monitor_web._REPO_ROOT, isical data dirs,
# calibration/.venv-multical — resolves inside the image).
COPY pyproject.toml README.md /opt/isi/
COPY backbone /opt/isi/backbone
COPY isistream /opt/isi/isistream
COPY calibration /opt/isi/calibration
COPY monitor_web /opt/isi/monitor_web
COPY isical /opt/isi/isical
COPY tools /opt/isi/tools
COPY config/backbone.yaml.example /opt/isi/config/backbone.yaml.example

# Deps are already satisfied by the env → --no-deps keeps pip from dragging in
# the opencv-python PyPI wheel (would collide with conda's opencv).
RUN pip install --no-cache-dir --no-deps -e /opt/isi -e /opt/isi/monitor_web

# Multical's isolated venv (pins opencv-contrib <=4.7; never imported by the
# runtime — calibrate.py invokes its binary by absolute path).
# One layer: install the apriltags build toolchain (git + cmake + opencv/eigen
# headers), build the venv, then purge the toolchain — keeping only the system
# opencv/lapack/tbb RUNTIME libs the compiled apriltags_eth.so links against
# (apt-mark manual protects them from autoremove). MULTICAL_VIEWER=0 skips the
# ~1.1 GB Qt/VTK 3D-viewer stack (headless container; solves don't need it).
RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake build-essential libeigen3-dev libopencv-dev \
    && MULTICAL_VIEWER=0 PYTHON=/opt/conda/bin/python3.10 \
        bash /opt/isi/calibration/setup_multical.sh \
    && test -f /opt/isi/calibration/.venv-multical/lib/python3.10/site-packages/apriltags_eth*.so \
    && dpkg-query -W -f='${Package}\n' \
        | grep -E '^(libopencv-(core|imgproc|calib3d|features2d|flann|stitching)|liblapack3|libblas3|libtbb)' \
        | xargs apt-mark manual \
    && apt-get purge -y git cmake build-essential libeigen3-dev libopencv-dev \
    && apt-get autoremove -y --purge \
    && rm -rf /var/lib/apt/lists/* /root/.cache

# `video` capability → NVDEC/NVENC (libnvcuvid) for GStreamer's nvcodec.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    LD_LIBRARY_PATH=/opt/conda/lib

WORKDIR /opt/isi
EXPOSE 8000 8300

CMD ["python", "-m", "monitor_web"]
