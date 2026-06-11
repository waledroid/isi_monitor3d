import numpy as np
from src.stages.control_maps.canny import CannyExtractor


def _square_image():
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    img[50:150, 100:200] = 255
    return img


def test_edges_found_shape_dtype():
    out = CannyExtractor().extract(_square_image())
    assert out.shape == (200, 300) and out.dtype == np.uint8
    assert out.any()                                          # edges exist


def test_thresholds_change_edge_count():
    img = _square_image()
    loose = CannyExtractor(low=10, high=50).extract(img)
    tight = CannyExtractor(low=200, high=250).extract(img)
    assert int(loose.sum()) >= int(tight.sum())
