"""/clients — REST-consumer tracking (middleware) + broker client count.

The middleware records every data-endpoint request per client, keyed by the
optional ``X-Client-Name`` header (how AGVs self-identify) or client IP.
Page shells / docs / health are untracked. ``mqtt_connected`` mirrors the
broker's ``$SYS/broker/clients/connected`` count (None before the first
$SYS publish).
"""

from __future__ import annotations

from isicomms.config import API_VERSION


def test_middleware_records_api_clients(client):
    client.get("/v1/zones")        # tracked (data endpoint)
    client.get("/healthz")         # untracked
    client.get("/ui")              # untracked (page shell)
    d = client.get("/clients").json()
    assert d["count"] == 1         # one client key (the test client's IP)
    row = d["api_clients"][0]
    assert row["name"] is None
    assert row["ip"] == "testclient"
    assert row["active"] is True
    # /v1/zones + the /clients request itself; never /healthz or /ui
    assert row["requests"] == 2
    assert row["last_path"] == "/clients"


def test_client_name_header_becomes_key(client):
    client.get("/v1/zones", headers={"X-Client-Name": "agv_07"})
    d = client.get("/clients").json()
    agv = next(r for r in d["api_clients"] if r["name"] == "agv_07")
    assert agv["requests"] == 1
    assert agv["ip"] == "testclient"
    # the unnamed /clients request tracks separately under the IP key
    assert d["count"] == 2


def test_mqtt_connected_none_before_sys_publish(client):
    assert client.get("/clients").json()["mqtt_connected"] is None


def test_clients_requires_token(authed_client):
    assert authed_client.get("/clients").status_code == 401
    r = authed_client.get(
        "/clients", headers={"Authorization": "Bearer secret"})
    assert r.status_code == 200


def test_clients_versioned_and_bare(client):
    # Both mounts answer. (Shape equality is not asserted — the middleware
    # legitimately increments the counters between the two calls.)
    assert client.get("/clients").status_code == 200
    assert client.get(f"/{API_VERSION}/clients").status_code == 200
