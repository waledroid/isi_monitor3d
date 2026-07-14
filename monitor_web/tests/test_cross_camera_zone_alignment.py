"""Cross-camera zone alignment — THE guarantee behind the ghost overlays.

A zone drawn on either camera's RAW (distorted) live stream must outline the
SAME physical floor region in the other camera, regardless of which camera it
was authored on. This exercises the full production chain on a synthetic
Mode-2 rig with realistic geometry and STRONG lens distortion (k1 = -0.4,
like the site cameras):

    ground-truth floor polygon
      → distorted pixels in the authoring camera   (what the operator clicks)
      → POST /api/zone-patches (stored patch)
      → GET  /api/zone-patches  → ghost pixels in the OTHER camera
      → authoring math of the other camera (undistort + H) → floor metres
      → must equal the ground-truth floor polygon to sub-centimetre.

Also pins the world-zone path (/api/project/pixel-to-floor + floor-to-pixel)
both ways. Any convention/scale/densification bug anywhere in the chain shows
up here as metres of error.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml
from backbone.shared.geometry import (
    densify_polygon,
    floor_homography_from_K_R_t,
    floor_to_pixel_distorted,
    pixel_to_floor,
    projection_from_K_R_t,
    undistort_points,
)
from calibration.schema import CALIBRATION_VERSION
from fastapi.testclient import TestClient

from monitor_web.app import create_app
from monitor_web.config import Settings

K = np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]])
D = np.array([-0.4, 0.15, 0.0, 0.0, 0.0])      # strong barrel — site-like
R_LOOK_DOWN = np.diag([1.0, -1.0, -1.0])
IMG_WH = (1920, 1080)
# Ground-truth zone: a square on the shared floor between the two cameras.
TRUTH = np.array([[0.6, -0.4], [1.4, -0.4], [1.4, 0.4], [0.6, 0.4]])


def _cam_dict(camera_id: str, center_xy: tuple[float, float]) -> dict:
    t = np.array([center_xy[0], center_xy[1], 2.5])
    return {
        "camera_id": camera_id,
        "image_size_wh": list(IMG_WH),
        "K": K.tolist(), "D": D.tolist(),
        "R": R_LOOK_DOWN.tolist(), "t": t.tolist(),
        "H": floor_homography_from_K_R_t(K, R_LOOK_DOWN, t).tolist(),
        "P": projection_from_K_R_t(K, R_LOOK_DOWN, t).tolist(),
        "reprojection_rms_px": 0.5,
    }


def _rig_views():
    from monitor_web.api.routes_projection import _load_rig_cached
    return _load_rig_cached


def _build_app(tmp_path: Path):
    cal = tmp_path / "mode2" / "calibration.json"
    cal.parent.mkdir(parents=True, exist_ok=True)
    cal.write_text(json.dumps({
        "version": CALIBRATION_VERSION,
        "created_at": "2026-07-02T00:00:00Z",
        "floor_anchor_method": "synthetic",
        "floor_origin_note": "alignment test",
        "calibration_mode": "multical_full",
        "cameras": {"cam_a": _cam_dict("cam_a", (0.0, 0.0)),
                    "cam_b": _cam_dict("cam_b", (2.0, 0.0))},
    }))
    backbone_yaml = tmp_path / "backbone.yaml"
    backbone_yaml.write_text(yaml.safe_dump({
        "cameras": {"cam_a": {}, "cam_b": {}},
        "calibration_path": str(cal),
        "metadata": {"sinks": []},
    }))
    cfg = Settings(backbone_config_path=backbone_yaml, udp_port=0, port=0,
                   ui_settings_path=tmp_path / "monitor_web_ui.yaml")
    return create_app(cfg), cal


def _operator_clicks(cam: dict, floor_poly: np.ndarray) -> list[list[float]]:
    """What the operator clicks: the floor polygon as seen on the RAW
    (distorted) live stream of the authoring camera."""
    t = np.asarray(cam["t"], dtype=np.float64)
    uv = floor_to_pixel_distorted(floor_poly, K, D, R_LOOK_DOWN, t, IMG_WH)
    return [[float(u), float(v)] for u, v in uv]


def _ghost_to_floor(ghost: dict, cal_path: Path) -> np.ndarray:
    """Map the ghost pixels back to floor metres via the TARGET camera's
    authoring math (undistort + H) — the same math a zone drawn there uses."""
    rig = _rig_views()(str(cal_path.resolve()), cal_path.stat().st_mtime_ns)
    view = rig[ghost["camera"]]
    pts = np.asarray(ghost["polygon"], dtype=np.float64)
    return pixel_to_floor(undistort_points(pts, view.K, view.D), view.H)


def _cross_outline(patches: list[dict], base_id: str) -> dict:
    """The zone's cross-camera outline in ghost shape: the server-derived TWIN
    when present (it detects too), else the display-only ghost."""
    twin = next((p for p in patches if p.get("twin_of") == base_id), None)
    if twin is not None:
        return {"camera": twin["camera"], "polygon": twin["polygon"],
                "image_wh": twin["frame_wh"]}
    base = next(p for p in patches if p["id"] == base_id)
    assert base.get("ghost"), "zone has neither twin nor ghost"
    return base["ghost"]


def _authored_floor_boundary(clicks: list[list[float]], frame_wh, cam: dict) -> np.ndarray:
    """The floor boundary the operator's clicks actually enclose: densified in
    PIXEL space (straight lines on the image, as drawn), then authored to the
    floor — the reference the ghost must reproduce in the other camera."""
    pts = densify_polygon(np.asarray(clicks, dtype=np.float64), segments_per_edge=8)
    pts[:, 0] *= IMG_WH[0] / float(frame_wh[0])
    pts[:, 1] *= IMG_WH[1] / float(frame_wh[1])
    return pixel_to_floor(undistort_points(pts, K, D),
                          np.asarray(cam["H"], dtype=np.float64))


def _dist_to_polygon_boundary(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Min distance from each point to the polygon's boundary segments."""
    out = np.full(len(pts), np.inf)
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        ab = b - a
        tt = np.clip(((pts - a) @ ab) / (ab @ ab), 0.0, 1.0)
        proj = a + tt[:, None] * ab
        out = np.minimum(out, np.linalg.norm(pts - proj, axis=1))
    return out


def _synced_floor_polygon(tmp_path: Path, zone_id: str) -> np.ndarray:
    """The floor zone the save synced to zones.yaml — the system's single zone
    truth (isistream crops, zone state, cam-view clip all use it). The TWIN is
    now ITS projection into the other camera, so this densified polygon is the
    boundary the twin must reproduce (not the raw pixel-drawn boundary, which
    bows a few cm against it — straight image lines curve on the floor)."""
    doc = yaml.safe_load((tmp_path / "zones.yaml").read_text())
    z = next(z for z in doc["zones"] if str(z.get("id")) == zone_id)
    return densify_polygon(np.asarray(z["polygon"], dtype=np.float64),
                           segments_per_edge=8)


def _assert_aligned(ghost_floor: np.ndarray, authored_floor: np.ndarray) -> None:
    """The twin must reproduce the synced floor zone to millimetres (chain
    fidelity of project→store→unproject), and stay physically on the true
    zone boundary within a few cm."""
    assert ghost_floor.shape == authored_floor.shape
    err = np.linalg.norm(ghost_floor - authored_floor, axis=1)
    assert err.max() < 0.005, (
        f"cross-camera zone misaligned: twin deviates from the synced floor "
        f"zone by up to {err.max()*100:.2f} cm — must be < 0.5 cm"
    )
    d = _dist_to_polygon_boundary(ghost_floor, TRUTH)
    assert d.max() < 0.05, (
        f"ghost strays {d.max()*100:.1f} cm from the physical zone boundary"
    )


def test_zone_drawn_on_cam_b_aligns_in_cam_a(tmp_path: Path) -> None:
    app, cal = _build_app(tmp_path)
    with TestClient(app) as client:
        client.post("/api/zone-patches", json={"patches": [
            {"id": "z1", "name": "Z", "camera": "cam_b",
             "polygon": _operator_clicks(_cam_dict("cam_b", (2.0, 0.0)), TRUTH),
             "frame_wh": list(IMG_WH)},
        ]})
        got = client.get("/api/zone-patches").json()["patches"]
    outline = _cross_outline(got, "z1")
    assert outline["camera"] == "cam_a"
    _assert_aligned(_ghost_to_floor(outline, cal),
                    _synced_floor_polygon(tmp_path, "z1"))


def test_zone_drawn_on_cam_a_aligns_in_cam_b(tmp_path: Path) -> None:
    """Symmetry: authoring camera must not matter."""
    app, cal = _build_app(tmp_path)
    clicks = _operator_clicks(_cam_dict("cam_a", (0.0, 0.0)), TRUTH)
    with TestClient(app) as client:
        client.post("/api/zone-patches", json={"patches": [
            {"id": "z1", "name": "Z", "camera": "cam_a",
             "polygon": clicks, "frame_wh": list(IMG_WH)},
        ]})
        got = client.get("/api/zone-patches").json()["patches"]
    outline = _cross_outline(got, "z1")
    assert outline["camera"] == "cam_b"
    _assert_aligned(_ghost_to_floor(outline, cal),
                    _synced_floor_polygon(tmp_path, "z1"))


def test_zone_drawn_at_downscaled_stream_size_still_aligns(tmp_path: Path) -> None:
    """The dashboard stream is 1280x720 while the calibration is 1920x1080 —
    the stored frame_wh scaling must not break alignment (the real deployment
    draws at the downscaled size)."""
    app, cal = _build_app(tmp_path)
    clicks = np.asarray(_operator_clicks(_cam_dict("cam_b", (2.0, 0.0)), TRUTH))
    clicks *= [1280.0 / IMG_WH[0], 720.0 / IMG_WH[1]]
    with TestClient(app) as client:
        client.post("/api/zone-patches", json={"patches": [
            {"id": "z1", "name": "Z", "camera": "cam_b",
             "polygon": clicks.tolist(), "frame_wh": [1280, 720]},
        ]})
        got = client.get("/api/zone-patches").json()["patches"]
    outline = _cross_outline(got, "z1")
    _assert_aligned(_ghost_to_floor(outline, cal),
                    _synced_floor_polygon(tmp_path, "z1"))


def test_world_zone_round_trip_both_cameras(tmp_path: Path) -> None:
    """The FLOOR-zone path: clicks on either camera → pixel-to-floor must
    recover the true floor polygon; floor-to-pixel into EACH camera must land
    exactly where the polygon appears on that camera's raw stream."""
    app, _cal = _build_app(tmp_path)
    with TestClient(app) as client:
        for cam_id, center in (("cam_a", (0.0, 0.0)), ("cam_b", (2.0, 0.0))):
            clicks = _operator_clicks(_cam_dict(cam_id, center), TRUTH)
            r = client.post("/api/project/pixel-to-floor",
                            json={"camera_id": cam_id, "points": clicks})
            world = np.asarray(r.json()["points"])
            err = np.linalg.norm(world - TRUTH, axis=1)
            assert err.max() < 0.01, f"{cam_id} authoring error {err.max()*100:.1f} cm"

        for cam_id, center in (("cam_a", (0.0, 0.0)), ("cam_b", (2.0, 0.0))):
            r = client.post("/api/project/floor-to-pixel",
                            json={"camera_id": cam_id, "polygon": TRUTH.tolist()})
            got = np.asarray(r.json()["points"])
            t = np.array([center[0], center[1], 2.5])
            # The endpoint densifies (8 samples/edge) so straight floor edges
            # draw as curves on the distorted frame.
            expected = floor_to_pixel_distorted(densify_polygon(TRUTH, 8),
                                                K, D, R_LOOK_DOWN, t, IMG_WH)
            err = np.linalg.norm(got - expected, axis=1)
            assert err.max() < 0.5, (
                f"{cam_id} display error {err.max():.1f} px — overlay would not "
                f"hug the raw frame"
            )


# ---- auto-twin patches: the OCCLUSION guarantee -----------------------------


def test_save_derives_twin_on_other_camera(tmp_path: Path) -> None:
    """Saving a zone on cam_b creates a server-derived twin on cam_a covering
    the same floor region — both workers detect it."""
    app, cal = _build_app(tmp_path)
    clicks = _operator_clicks(_cam_dict("cam_b", (2.0, 0.0)), TRUTH)
    with TestClient(app) as client:
        client.post("/api/zone-patches", json={"patches": [
            {"id": "z1", "name": "Zone 1", "camera": "cam_b",
             "polygon": clicks, "frame_wh": list(IMG_WH),
             },
        ]})
        got = client.get("/api/zone-patches").json()["patches"]

    assert [p["id"] for p in got] == ["z1", "z1__twin"]
    twin = got[1]
    assert twin["camera"] == "cam_a"
    assert twin["twin_of"] == "z1"
    assert twin["name"] == "Zone 1"
    # No per-zone knobs survive to the twin (one global model, one threshold);
    # keys (model/infer_size/sahi/enhance) are gone — one perception, no local models.
    assert "confidence" not in twin
    assert "infer_size" not in twin and "model" not in twin
    # The twin's floor footprint matches the authored zone (via cam_a's math).
    from monitor_web.api.routes_projection import _load_rig_cached
    rig = _load_rig_cached(str(cal.resolve()), cal.stat().st_mtime_ns)
    tw = np.asarray(twin["polygon"], dtype=np.float64)
    floor = pixel_to_floor(undistort_points(tw, K, D), rig["cam_a"].H)
    d = _dist_to_polygon_boundary(floor, TRUTH)
    assert d.max() < 0.05
    # A twinned base patch needs no ghost (the twin IS the outline over there).
    assert got[0].get("ghost") is None


def test_twins_regenerate_and_never_duplicate(tmp_path: Path) -> None:
    """Round-tripping the GET response back into POST must not multiply twins,
    and deleting the base drops its twin."""
    app, _cal = _build_app(tmp_path)
    clicks = _operator_clicks(_cam_dict("cam_b", (2.0, 0.0)), TRUTH)
    with TestClient(app) as client:
        client.post("/api/zone-patches", json={"patches": [
            {"id": "z1", "name": "Zone 1", "camera": "cam_b",
             "polygon": clicks, "frame_wh": list(IMG_WH)},
        ]})
        got = client.get("/api/zone-patches").json()["patches"]
        # Naive client round-trips EVERYTHING (twins included).
        client.post("/api/zone-patches", json={"patches": got})
        again = client.get("/api/zone-patches").json()["patches"]
        assert [p["id"] for p in again] == ["z1", "z1__twin"]

        client.post("/api/zone-patches", json={"patches": []})
        assert client.get("/api/zone-patches").json()["patches"] == []


class _Det:
    def __init__(self, cls, conf, bbox):
        self.cls = cls
        self.confidence = conf
        self.bbox_xyxy = bbox


class _FakeManager:
    """zone_manager stub: per-zone detections + status ('' = no coverage)."""

    def __init__(self, dets_by_zone):
        self._dets = dets_by_zone

    def zone_status(self, zone_id):
        return "ok" if zone_id in self._dets else ""

    def zone_dets(self, zone_id):
        return list(self._dets.get(zone_id, []))


def test_zone_state_survives_occluded_camera(tmp_path: Path) -> None:
    """THE occlusion guarantee: the base camera is blind (occluded — no fresh
    coverage) but the twin on the other camera still sees the objects → the
    zone still reports them, under the BASE zone's id."""
    app, _cal = _build_app(tmp_path)
    clicks = _operator_clicks(_cam_dict("cam_b", (2.0, 0.0)), TRUTH)
    with TestClient(app) as client:
        client.post("/api/zone-patches", json={"patches": [
            {"id": "z1", "name": "Zone 1", "camera": "cam_b",
             "polygon": clicks, "frame_wh": list(IMG_WH)},
        ]})
        # cam_b occluded: ONLY the twin (cam_a) has detections.
        client.app.state.zone_manager = _FakeManager({
            "z1__twin": [
                _Det("palette", 0.9, (100.0, 200.0, 300.0, 300.0)),
                _Det("carton", 0.8, (150.0, 120.0, 250.0, 210.0)),
            ],
        })
        data = client.get("/api/zone-patches/state").json()

    assert set(data["states"]) == {"z1"}          # reported as THE zone, not the twin
    st = data["states"]["z1"]
    assert st["count"] == 2
    assert any(o["cls"] == "palette" and o.get("occupancy_state") == "full"
               for o in st["objects"])


def test_zone_state_merge_does_not_double_count(tmp_path: Path) -> None:
    """Both cameras seeing the same object must count it ONCE (per-class max),
    not sum across cameras."""
    app, _cal = _build_app(tmp_path)
    clicks = _operator_clicks(_cam_dict("cam_b", (2.0, 0.0)), TRUTH)
    with TestClient(app) as client:
        client.post("/api/zone-patches", json={"patches": [
            {"id": "z1", "name": "Zone 1", "camera": "cam_b",
             "polygon": clicks, "frame_wh": list(IMG_WH)},
        ]})
        one_palette = [_Det("palette", 0.9, (100.0, 200.0, 300.0, 300.0))]
        client.app.state.zone_manager = _FakeManager({
            "z1": list(one_palette),
            "z1__twin": list(one_palette),
        })
        data = client.get("/api/zone-patches/state").json()
    assert data["states"]["z1"]["count"] == 1


def test_twin_into_folding_lens_drops_folded_samples(tmp_path: Path) -> None:
    """The site cam_b lens (k3 = -0.069) FOLDS its forward projection at
    normalized r = 1.11, just outside its frame corners (r = 1.03): a zone
    authored on cam_a whose floor region spills past cam_b's field used to get
    twin samples projected back INSIDE cam_b's frame at folded positions —
    the 'Zone 2 distorted in cam2' bug. The ghost must instead be the zone
    CLIPPED to cam_b's reliable field: every point on the authored boundary or
    the field rim, covering the full visible overlap (not a dropped-sample
    sliver)."""
    from monitor_web.api.routes_zone_patches import _patch_ghost

    D_K3 = np.array([-0.36, 0.162, -0.003, -0.001, -0.069])
    K3 = np.array([[1077.0, 0.0, 961.0], [0.0, 1077.0, 550.0], [0.0, 0.0, 1.0]])

    class _View:
        def __init__(self, K_, D_, center):
            self.K, self.D = K_, D_
            self.R = R_LOOK_DOWN
            self.t = np.array([center[0], center[1], 2.5])
            self.H = floor_homography_from_K_R_t(K_, self.R, self.t)
            self.image_size_wh = IMG_WH

    class _Rig:
        camera_ids = ("cam_a", "cam_b")
        def __init__(self):
            self.views = {"cam_a": _View(K, D, (0.0, 0.0)),
                          "cam_b": _View(K3, D_K3, (2.0, 0.0))}
        def __getitem__(self, cid):
            return self.views[cid]

    rig = _Rig()
    # Floor region reaching 3.3 m from cam_b's nadir — its far edge sits in
    # cam_b's fold band (2.77-3.35 m) while fully visible on cam_a.
    zone_floor = np.array([[-1.3, -0.4], [0.5, -0.4], [0.5, 0.4], [-1.3, 0.4]])
    clicks = floor_to_pixel_distorted(zone_floor, K, D, R_LOOK_DOWN,
                                      rig["cam_a"].t, IMG_WH)
    ghost = _patch_ghost({"id": "z", "camera": "cam_a",
                          "polygon": clicks.tolist(),
                          "frame_wh": list(IMG_WH)}, rig)
    assert ghost is not None and ghost["camera"] == "cam_b"

    # The hazard is real: the raw forward model folds boundary samples back
    # inside cam_b's frame at positions whose floor mapping is >10 cm wrong.
    dense = densify_polygon(zone_floor, segments_per_edge=8)
    vb = rig["cam_b"]
    raw = floor_to_pixel_distorted(dense, vb.K, vb.D, vb.R, vb.t, IMG_WH)
    inside = ((raw[:, 0] >= 0) & (raw[:, 0] < IMG_WH[0])
              & (raw[:, 1] >= 0) & (raw[:, 1] < IMG_WH[1]))
    back = pixel_to_floor(undistort_points(raw[inside], vb.K, vb.D), vb.H)
    d_raw = _dist_to_polygon_boundary(back, zone_floor)
    assert d_raw.max() > 0.10, "expected folded samples in the raw projection"

    # The served ghost has none: every point maps back either onto the
    # authored zone boundary or onto the reliable-field RIM (the visible
    # overlap is bounded by the rim, not just the zone's own boundary).
    from backbone.shared.geometry import radial_fold_radius
    gpts = np.asarray(ghost["polygon"], dtype=np.float64)
    gback = pixel_to_floor(undistort_points(gpts, vb.K, vb.D), vb.H)
    d_boundary = _dist_to_polygon_boundary(gback, zone_floor)
    rim_m = radial_fold_radius(D_K3) * 0.9 * 2.5      # fold_safety * nadir 2.5 m
    d_rim = np.abs(np.linalg.norm(gback - np.array([2.0, 0.0]), axis=1) - rim_m)
    worst = np.minimum(d_boundary, d_rim).max()
    assert worst < 0.08, (
        f"ghost contains folded samples: strays {worst*100:.1f} cm from both "
        f"the authored boundary and the field rim"
    )
    # …and it is the real visible overlap, not a sliver: it reaches the
    # zone's near edge (x = 0.5) and the rim crossing (x ~ 2 - rim), spans y.
    assert gback[:, 0].max() > 0.45
    assert gback[:, 0].min() < (2.0 - rim_m) + 0.08
    assert gback[:, 1].min() < -0.35 and gback[:, 1].max() > 0.35
