"""REST API versioning — /v1 prefix on all resource routers + bare aliases."""

from __future__ import annotations

import pytest

from isi_gateway.config import API_VERSION

_RESOURCE_PATHS = [
    "/nodes",
    "/tracks",
    "/diagnostics",
    "/passings",
    "/zones",
    "/config",
]


def test_api_version_constant_is_v1():
    assert API_VERSION == "v1"


@pytest.mark.parametrize("path", _RESOURCE_PATHS)
def test_versioned_path_available(client, path):
    r = client.get(f"/{API_VERSION}{path}")
    assert r.status_code == 200


@pytest.mark.parametrize("path", _RESOURCE_PATHS)
def test_bare_alias_still_available(client, path):
    r = client.get(path)
    assert r.status_code == 200


@pytest.mark.parametrize("path", _RESOURCE_PATHS)
def test_versioned_and_bare_same_shape(client, path):
    bare = client.get(path)
    versioned = client.get(f"/{API_VERSION}{path}")
    assert bare.json() == versioned.json()


def test_healthz_unprefixed():
    """/healthz stays available un-prefixed."""
    from fastapi.testclient import TestClient

    from isi_gateway.app import create_app
    from isi_gateway.config import Settings

    app = create_app(Settings(mqtt_port=1884))
    with TestClient(app) as c:
        assert c.get("/healthz").status_code == 200


def test_healthz_also_under_v1():
    from fastapi.testclient import TestClient

    from isi_gateway.app import create_app
    from isi_gateway.config import Settings

    app = create_app(Settings(mqtt_port=1884))
    with TestClient(app) as c:
        assert c.get(f"/{API_VERSION}/healthz").status_code == 200
