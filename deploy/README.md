# ISI Monitor 3D — central server deployment

The central server runs a Mosquitto MQTT broker and the `isi-gateway` REST API.
All warehouse-PC Backbone nodes publish to this broker; AGVs and WMS systems pull
from the gateway API. Two deployment profiles are provided.

---

## On-prem profile (LAN, plaintext)

For a trusted private network where all traffic is physically controlled.

```bash
docker compose -f deploy/onprem/docker-compose.yml up -d --build
```

- Mosquitto on `:1883` (anonymous, no TLS)
- Gateway API on `http://<server>:8080`

Optional: add broker credentials without TLS by setting `allow_anonymous false` and
a `password_file` in `deploy/onprem/mosquitto.conf` and uncommenting
`ISI_GATEWAY_MQTT_USERNAME`/`ISI_GATEWAY_MQTT_PASSWORD` in the compose file.

**Never use this profile on a network reachable from the internet.**

---

## Cloud profile (internet-facing: TLS + auth + API token)

For deployments where the broker or API is reachable over the public internet.

### 1. Generate certificates

```bash
cd deploy/cloud
CERT_HOST=<server-ip-or-dns> ./gen-certs.sh
```

This mints a self-signed CA (`certs/ca.crt`) plus a broker cert (`server.crt`) and
an API cert (`api.crt`), both carrying a SAN for `CERT_HOST`. Run once per server;
re-running overwrites existing certs (update all clients afterwards).

**Distribute `certs/ca.crt`** to:
- each Backbone node (set `ca_cert: /etc/isi/ca.crt` in the node's `backbone.yaml`
  mqtt sink block — or `ISI_GATEWAY_MQTT_CA_CERT` if using the gateway in TLS mode)
- every AGV / WMS HTTP client (install as a trusted CA, or pass as `--cacert` to
  curl; use `-k` / `--insecure` only for quick smoke tests, never in production)

### 2. Create broker credentials

```bash
# Using Docker (no local mosquitto_passwd needed):
docker run --rm -v "$PWD:/w" eclipse-mosquitto:2 \
    mosquitto_passwd -b -c /w/passwd <username> <password>

# Or, if mosquitto_passwd is installed locally:
mosquitto_passwd -b -c deploy/cloud/passwd <username> <password>
```

Use the same username/password in each node's `backbone.yaml` mqtt sink block.

### 3. Configure the environment

```bash
cp deploy/cloud/.env.example deploy/cloud/.env
# Edit .env — fill MQTT_USERNAME, MQTT_PASSWORD, API_TOKEN, CERT_HOST
```

### 4. Start the stack

```bash
docker compose -f deploy/cloud/docker-compose.yml up -d --build
```

- Mosquitto on `:8883` (MQTTS — TLS + password auth)
- Gateway behind Caddy on `https://<server>/` (port 443)
- AGVs / WMS: `GET https://<server>/tracks` with `Authorization: Bearer <API_TOKEN>`

### Backbone node config for the cloud profile

In each node's `config/backbone.yaml`, update the mqtt sink block:

```yaml
metadata:
  sinks:
    - plugin: mqtt
      host: <cloud-server-ip-or-dns>
      port: 8883
      tls: true
      ca_cert: /etc/isi/ca.crt      # the distributed ca.crt
      username: <MQTT_USERNAME>
      password: <MQTT_PASSWORD>
      prefix: isiMonitor3D/<node_id>
```

### Let's Encrypt (real domain)

To replace the self-signed cert with a trusted one from Let's Encrypt, edit
`deploy/cloud/Caddyfile`: replace `:443` with your domain name and remove the
`tls /certs/api.crt /certs/api.key` line. Caddy handles ACME automatically
(port 80 must be reachable for the HTTP-01 challenge). The broker cert still
requires manual renewal via `gen-certs.sh` or a proper PKI.

---

## Security checklist

Before exposing the system to the internet, confirm ALL of the following:

- [ ] Using the **cloud** profile (not on-prem)
- [ ] Mosquitto TLS enabled (`listener 8883`, certs mounted)
- [ ] `allow_anonymous false` + `password_file` set in `mosquitto.conf`
- [ ] `MQTT_USERNAME` and `MQTT_PASSWORD` are non-empty, rotatable secrets
- [ ] `API_TOKEN` is a non-empty, high-entropy secret
- [ ] `ca.crt` distributed to all clients; `-k`/`--insecure` not used in production
- [ ] Ports 1883 (plaintext MQTT) and 8080 (gateway HTTP) are NOT exposed externally
- [ ] `.env`, `passwd`, and `certs/` are NOT committed to git (covered by `.gitignore`)
