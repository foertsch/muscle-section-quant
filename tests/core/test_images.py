import numpy as np
import pytest
from skimage.measure import block_reduce

from msq.core.images import block_mean, block_mean_planes, block_mean_rgb, iter_strips, trim_to_multiple


def test_block_mean_keeps_grids_aligned():
    img = np.arange(10 * 9, dtype=np.uint16).reshape(10, 9)
    assert trim_to_multiple(img, 4).shape == (8, 8)
    ds = block_mean(img, 4)
    assert ds.shape == (2, 2) and ds.dtype == np.float32
    assert ds[0, 0] == pytest.approx(img[:4, :4].mean())


def test_strips_cover_the_trimmed_image_once_and_flip_on_request():
    frame = np.random.default_rng(1).integers(0, 255, (103, 50, 3), dtype=np.uint8)
    rows = [(y0, y1) for y0, y1, _ in iter_strips(frame, 6, rows=12)]
    assert rows[0][0] == 0 and rows[-1][1] == 102 and all(a[1] == b[0] for a, b in zip(rows, rows[1:]))
    _, _, first = next(iter_strips(frame, 6, rows=12, flip=True))
    assert first.shape == (12, 48, 3) and (first == frame[:12, :48, ::-1]).all()
    _, _, plain = next(iter_strips(frame, 6, rows=12))
    assert (plain == frame[:12, :48]).all()


def test_block_mean_rgb_matches_a_whole_image_block_reduce():
    frame = np.random.default_rng(2).integers(0, 255, (60, 45, 3), dtype=np.uint8)
    small = block_mean_rgb(frame, 3, rows=9, flip=True)
    expected = block_reduce(frame[..., ::-1].astype(np.float32), (3, 3, 1), np.mean)
    assert small.shape == (20, 15, 3) and np.allclose(small, expected)


def test_block_mean_planes_in_strips_equals_whole_image():
    frame = np.random.default_rng(0).integers(0, 16383, (3, 103, 58), dtype=np.uint16)
    got = block_mean_planes(frame, [2, 0], 4, rows=12)
    for g, c in zip(got, [2, 0]):
        assert g.shape == (25, 14)
        np.testing.assert_allclose(g, block_reduce(frame[c, :100, :56].astype(np.float32), (4, 4), np.mean))
