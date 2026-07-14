"""Masks never render outside the floor zone (the mask_clip stencil).

A detection whose foot is inside a zone is kept, but its segmentation mask can
hug/cross the zone boundary — the stencil bounds the FILL so no mask pixel
lands outside the dashed outline, on the cam views and the zone panels alike.
"""

from types import SimpleNamespace

import numpy as np

from monitor_web.overlay import draw
from monitor_web.zone_projection import zone_stencil

_GRAY = 100


def _frame(w=200, h=200):
    return np.full((h, w, 3), _GRAY, dtype=np.uint8)


def _det(mask_poly=None, mask=None, bbox=(20.0, 20.0, 180.0, 180.0)):
    return SimpleNamespace(cls="palette", confidence=0.9, bbox_xyxy=bbox,
                           foot_uv=(100.0, 180.0), mask=mask, mask_poly=mask_poly,
                           keypoints_uv=None, mask_offset_xy=None)


# one square zone occupying the LEFT half of the frame
_ZONE = [("z1", "Zone 1", np.array([[0, 0], [100, 0], [100, 200], [0, 200]], float))]


def test_polygon_mask_is_clipped_to_the_zone():
    """A mask polygon spanning both halves colours pixels ONLY inside the zone."""
    img = _frame()
    stencil = zone_stencil((200, 200), _ZONE)
    # mask covers x 40..160 — half inside the zone, half outside
    d = _det(mask_poly=[[40, 80], [160, 80], [160, 120], [40, 120]])
    draw(img, d, show_boxes=False, mask_clip=stencil)
    inside = img[100, 60]     # x=60 → inside zone AND mask → tinted
    outside = img[100, 140]   # x=140 → inside mask but OUTSIDE zone → untouched
    assert not np.array_equal(inside, (_GRAY, _GRAY, _GRAY))
    assert np.array_equal(outside, (_GRAY, _GRAY, _GRAY))


def test_bitmap_mask_is_clipped_to_the_zone():
    img = _frame()
    stencil = zone_stencil((200, 200), _ZONE)
    m = np.zeros((200, 200), dtype=bool)
    m[80:120, 40:160] = True
    draw(img, _det(mask=m), show_boxes=False, mask_clip=stencil)
    assert not np.array_equal(img[100, 60], (_GRAY, _GRAY, _GRAY))
    assert np.array_equal(img[100, 140], (_GRAY, _GRAY, _GRAY))


def test_no_stencil_keeps_the_old_behaviour():
    """Without a stencil (frames mode, MP4 viewer) the whole mask still fills."""
    img = _frame()
    d = _det(mask_poly=[[40, 80], [160, 80], [160, 120], [40, 120]])
    draw(img, d, show_boxes=False)
    assert not np.array_equal(img[100, 60], (_GRAY, _GRAY, _GRAY))
    assert not np.array_equal(img[100, 140], (_GRAY, _GRAY, _GRAY))


def test_stencil_rasterizes_all_zones():
    two = [*_ZONE, ("z2", "Zone 2", np.array([[150, 150], [199, 150], [199, 199], [150, 199]], float))]
    st = zone_stencil((200, 200), two)
    assert st[10, 10] == 255      # zone 1
    assert st[180, 180] == 255    # zone 2
    assert st[10, 180] == 0       # neither
