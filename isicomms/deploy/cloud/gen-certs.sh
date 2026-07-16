#!/usr/bin/env bash
# gen-certs.sh — mint a self-signed CA + broker server cert + API cert.
#
# Usage:
#   CERT_HOST=my.server.com ./gen-certs.sh
#   ./gen-certs.sh my.server.com
#
# Outputs to deploy/cloud/certs/:
#   ca.key ca.crt          — trust root (distribute ca.crt to every node + AGV/WMS client)
#   server.key server.crt  — Mosquitto TLS listener (8883)
#   api.key api.crt        — Caddy HTTPS listener (443 → gateway:8080)
#
# Re-running overwrites existing certs (idempotent-ish: dir is created, files replaced).

set -euo pipefail

CERT_HOST="${1:-${CERT_HOST:-localhost}}"
OUTDIR="$(dirname "$0")/certs"
DAYS=825          # ~2y — max broadly accepted by mobile/browser trust stores
BITS=2048

echo "==> Generating certs for host: ${CERT_HOST}"
echo "    Output directory: ${OUTDIR}"

if [[ -d "${OUTDIR}" && -n "$(ls -A "${OUTDIR}" 2>/dev/null)" ]]; then
    echo "    WARNING: ${OUTDIR} already contains files — they will be overwritten."
fi
mkdir -p "${OUTDIR}"

# ---------------------------------------------------------------------------
# 1. CA (self-signed)
# ---------------------------------------------------------------------------
echo ""
echo "--> CA key + certificate"
openssl genrsa -out "${OUTDIR}/ca.key" ${BITS} 2>/dev/null
openssl req -new -x509 -days ${DAYS} \
    -key "${OUTDIR}/ca.key" \
    -out "${OUTDIR}/ca.crt" \
    -subj "/CN=ISI-Monitor3D-CA/O=ISI/OU=Monitor3D"

# ---------------------------------------------------------------------------
# 2. Broker (Mosquitto) server cert — SAN covers DNS + IP
# ---------------------------------------------------------------------------
echo "--> Broker server key + CSR + certificate"
openssl genrsa -out "${OUTDIR}/server.key" ${BITS} 2>/dev/null

# Build the SAN extension inline via a temp config
BROKER_EXT=$(mktemp)
cat >"${BROKER_EXT}" <<EOF
[req]
distinguished_name = dn
req_extensions = v3_req
prompt = no
[dn]
CN = ${CERT_HOST}
O  = ISI
OU = Monitor3D-Broker
[v3_req]
subjectAltName = DNS:${CERT_HOST},IP:127.0.0.1
keyUsage = digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth
EOF

openssl req -new \
    -key "${OUTDIR}/server.key" \
    -out "${OUTDIR}/server.csr" \
    -config "${BROKER_EXT}"

openssl x509 -req -days ${DAYS} \
    -in "${OUTDIR}/server.csr" \
    -CA "${OUTDIR}/ca.crt" \
    -CAkey "${OUTDIR}/ca.key" \
    -CAcreateserial \
    -out "${OUTDIR}/server.crt" \
    -extensions v3_req \
    -extfile "${BROKER_EXT}" 2>/dev/null

rm -f "${BROKER_EXT}" "${OUTDIR}/server.csr"

# ---------------------------------------------------------------------------
# 3. API (Caddy) cert — same trust root, same SAN approach
# ---------------------------------------------------------------------------
echo "--> API server key + CSR + certificate"
openssl genrsa -out "${OUTDIR}/api.key" ${BITS} 2>/dev/null

API_EXT=$(mktemp)
cat >"${API_EXT}" <<EOF
[req]
distinguished_name = dn
req_extensions = v3_req
prompt = no
[dn]
CN = ${CERT_HOST}
O  = ISI
OU = Monitor3D-API
[v3_req]
subjectAltName = DNS:${CERT_HOST},IP:127.0.0.1
keyUsage = digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth
EOF

openssl req -new \
    -key "${OUTDIR}/api.key" \
    -out "${OUTDIR}/api.csr" \
    -config "${API_EXT}"

openssl x509 -req -days ${DAYS} \
    -in "${OUTDIR}/api.csr" \
    -CA "${OUTDIR}/ca.crt" \
    -CAkey "${OUTDIR}/ca.key" \
    -CAcreateserial \
    -out "${OUTDIR}/api.crt" \
    -extensions v3_req \
    -extfile "${API_EXT}" 2>/dev/null

rm -f "${API_EXT}" "${OUTDIR}/api.csr"

# ---------------------------------------------------------------------------
# 4. Verify
# ---------------------------------------------------------------------------
echo ""
echo "==> Verification"
echo -n "    broker cert: "
openssl verify -CAfile "${OUTDIR}/ca.crt" "${OUTDIR}/server.crt"
echo -n "    api cert:    "
openssl verify -CAfile "${OUTDIR}/ca.crt" "${OUTDIR}/api.crt"

echo ""
echo "==> Generated files in ${OUTDIR}:"
ls -lh "${OUTDIR}"
echo ""
echo "Next steps:"
echo "  1. Copy certs/ca.crt to every Backbone node + AGV/WMS HTTP client."
echo "  2. Create broker credentials: docker run --rm -v \"\$PWD:/w\" eclipse-mosquitto:2 mosquitto_passwd -b -c /w/passwd <user> <pass>"
echo "  3. cp .env.example .env  and fill MQTT_USERNAME, MQTT_PASSWORD, API_TOKEN, CERT_HOST."
echo "  4. docker compose up -d --build"
