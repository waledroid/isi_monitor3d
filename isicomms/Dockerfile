# Central isicomms (gateway) image — aggregates all Backbone nodes off the MQTT broker
# and serves the polling REST API. Build from the REPO ROOT (it needs the
# backbone package for backbone.metadata.schemas + backbone.shared.zones):
#
#   docker build -f isicomms/Dockerfile -t isicomms .
#
# The gateway only imports schemas + zones, so the backbone's base deps
# (numpy + pyyaml) are enough — no CUDA / OpenCV / GStreamer in this image.
FROM python:3.10-slim

WORKDIR /app

# Install the backbone (base deps only) so its schemas + zones are importable.
COPY pyproject.toml README.md ./
COPY backbone ./backbone
RUN pip install --no-cache-dir .

# Install the gateway (fastapi, uvicorn, paho-mqtt, pydantic-settings).
COPY isicomms ./isicomms
RUN pip install --no-cache-dir ./isicomms

ENV ISI_GATEWAY_HOST=0.0.0.0 \
    ISI_GATEWAY_PORT=8080
EXPOSE 8080

CMD ["python", "-m", "isicomms"]
