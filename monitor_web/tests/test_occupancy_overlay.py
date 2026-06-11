"""Live-CAM pallet empty/full badge — the image-space (A) occupancy association.

Hermetic: synthetic ``Detection``s, no model/calibration.
"""

from __future__ import annotations

from backbone.core.types import Detection

from monitor_web.detection_overlay import image_occupancy


def _det(cls, bbox):
    x1, _y1, x2, y2 = bbox
    return Detection(camera_id="cam_a", capture_ts=0.0, cls=cls, confidence=0.9,
                     bbox_xyxy=bbox, foot_uv=((x1 + x2) / 2.0, y2))


def test_carton_on_pallet_is_palette_carton():
    pallet = _det("palette", (100, 300, 300, 360))
    carton = _det("carton", (150, 250, 250, 300))     # rests on top, aligned
    res = image_occupancy([pallet, carton])
    assert len(res) == 1
    _pal, label = res[0]
    assert label == "palette_carton"


def test_bare_pallet_is_palette_vide():
    res = image_occupancy([_det("palette", (100, 300, 300, 360))])
    assert res[0][1] == "palette_vide"


def test_object_beside_pallet_is_palette_vide():
    pallet = _det("palette", (100, 300, 300, 360))
    beside = _det("polybag", (600, 250, 700, 300))    # no horizontal overlap
    res = image_occupancy([pallet, beside])
    assert res[0][1] == "palette_vide"


def test_both_loads_listed_in_canonical_order():
    pallet = _det("palette", (100, 300, 300, 360))
    carton = _det("carton", (150, 280, 190, 300))
    polybag = _det("polybag", (200, 240, 260, 300))
    _pal, label = image_occupancy([pallet, polybag, carton])[0]   # added polybag-first
    assert label == "palette_carton_polybag"                      # …still carton first


def test_polybag_only_is_palette_polybag():
    pallet = _det("palette", (100, 300, 300, 360))
    polybag = _det("polybag", (150, 250, 250, 300))
    assert image_occupancy([pallet, polybag])[0][1] == "palette_polybag"
