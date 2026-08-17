from backbone.comms.schemas import EtagereCellState, EtagereStateMessage


def _msg(zone_id="et_1"):
    cells = tuple(EtagereCellState(r=r, c=c, state="filled" if c == 1 else "empty",
                                   confidence=0.9) for r in (1, 2, 3) for c in (1, 2, 3))
    return EtagereStateMessage(ts=100.0, camera_id="cam_a", zone_id=zone_id, name="A",
                               cells=cells, stabilized=True)


def test_etagere_empty(client):
    r = client.get("/etagere")
    assert r.status_code == 200 and r.json() == {"etageres": [], "count": 0}


def test_etagere_listed_with_matrix(client):
    client.app.state.subscriber.update_from_message("node_a", _msg())
    r = client.get("/etagere")
    body = r.json()
    assert body["count"] == 1
    e = body["etageres"][0]
    assert e["node_id"] == "node_a" and e["zone_id"] == "et_1" and e["name"] == "A"
    assert e["matrix"] == [["filled", "empty", "empty"]] * 3
    assert len(e["cells"]) == 9 and e["ts"] == 100.0


def test_etagere_by_id_and_404(client):
    client.app.state.subscriber.update_from_message("node_a", _msg())
    assert client.get("/etagere/et_1").json()["zone_id"] == "et_1"
    assert client.get("/etagere/nope").status_code == 404
    assert client.get("/v1/etagere").status_code == 200
