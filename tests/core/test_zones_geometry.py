import numpy as np
import pytest

from msq.core.geometry import feret_px, min_feret_px
from msq.core.zones import zones


def test_zones_of_a_square_and_of_two_touching_squares():
    labels = np.zeros((20, 20), np.uint32)
    labels[5:15, 5:15] = 1
    core, band, ring = zones(labels, k=1)
    assert (core == 1).sum() == 64 and (band == 1).sum() == 36 and (ring == 1).sum() == 40
    assert not ((core > 0) & (band > 0)).any() and not ((ring > 0) & (labels > 0)).any()
    labels = np.zeros((12, 16), np.uint32)
    labels[2:8, 2:8] = 1
    labels[2:8, 8:14] = 2
    core, band, ring = zones(labels, 1)
    assert (core == 1).sum() == 16 and (core == 2).sum() == 16  # kept apart where they touch
    assert (ring > 0).sum() == 2 * 12 + 2 * 6 and not ring[labels > 0].any()


def test_ring_between_two_fibers_goes_to_the_nearer_one():
    labels = np.zeros((10, 13), np.uint32)
    labels[2:8, 1:6] = 1
    labels[2:8, 7:12] = 2
    _, _, ring = zones(labels, k=1)
    assert set(np.unique(ring[2:8, 6])) <= {1, 2} and (ring[2:8, 6] > 0).all()


def test_feret_of_rectangle_disk_and_degenerate_masks():
    m = np.zeros((50, 80), bool)
    m[10:30, 5:65] = True  # 20 x 60 px
    fmin, fmax = feret_px(m)
    assert fmin == pytest.approx(20, abs=0.01) and fmax == pytest.approx(np.hypot(20, 60), abs=0.01)
    assert min_feret_px(m) == fmin
    yy, xx = np.mgrid[:200, :200]
    assert min_feret_px((yy - 100) ** 2 + (xx - 100) ** 2 <= 50**2) == pytest.approx(101, abs=2)
    line = np.zeros((10, 10), bool)
    line[3, 2:7] = True
    assert feret_px(line) == (1.0, pytest.approx(np.hypot(1, 5)))
    assert feret_px(np.zeros((5, 5), bool)) == (0.0, 0.0)
