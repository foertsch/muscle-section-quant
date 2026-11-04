import numpy as np
import pytest

from msq.core.tissue import background_level, keep_large_pieces, main_section, off_tissue, otsu_tissue


def disk(shape, cy, cx, r):
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r**2


def test_main_section_keeps_largest_piece_and_bridges_small_gaps():
    t = disk((300, 300), 120, 120, 80)  # main section
    t |= disk((300, 300), 120, 255, 30)  # neighbouring section ~25 px (> 10 um) away
    t[90:110, 90:110] = False  # a hole inside the main section is filled
    piece = disk((300, 300), 235, 120, 34)  # fragment 1 px (< 10 um) below the main section
    m = main_section(t | piece, px_um=1.0, close_um=10)
    assert m[120, 120] and m[100, 100] and m[235, 120]
    assert not m[120, 255]
    assert not main_section(np.zeros((20, 20), bool), 1.0).any()
    assert main_section(t, px_um=1.0).shape == t.shape  # the 50 um default


def test_keep_large_pieces_drops_specks():
    m = disk((200, 200), 60, 60, 40) | disk((200, 200), 150, 150, 20) | disk((200, 200), 20, 180, 2)
    kept = keep_large_pieces(m)
    assert kept[60, 60] and kept[150, 150] and not kept[20, 180]
    assert kept.sum() == m.sum() - disk((200, 200), 20, 180, 2).sum()
    one = disk((50, 50), 25, 25, 10)
    assert keep_large_pieces(one) is one  # a single piece is returned as it is


def test_background_level_ignores_zero_padding_and_needs_pixels():
    img = np.full((100, 100), 480.0, np.float32)
    img[:10] = 0
    assert background_level(img, np.ones_like(img, bool)) == 480.0
    with pytest.raises(ValueError, match="too few"):
        background_level(np.ones((5, 5)), np.ones((5, 5), bool))


def test_otsu_tissue_separates_glass_from_dim_tissue_on_a_log_scale():
    img = np.full((120, 120), 300.0)
    img[30:90, 30:90] = 1200.0
    img[55:60, 55:60] = 300.0  # a hole in the tissue is filled
    img[:8] = 0  # unscanned rows are not tissue
    img[0, 100:103] = 5000.0  # a speck on an unscanned row: excluded by valid
    valid = img > 0
    for f in (lambda a: a, np.log1p):
        t = otsu_tissue(img, valid, f)
        assert t[60, 60] and t[57, 57] and not t[10, 10] and not t[0, 101] and t.sum() == 60 * 60
    speck = img.copy()
    speck[100:102, 5:7] = 1200.0
    assert not otsu_tissue(speck, valid)[100, 5]  # a piece under 1% of the largest is dropped


def test_off_tissue_diamond_or_euclidean_distance():
    t = np.zeros((101, 101), bool)
    t[50, 50] = True
    valid = np.ones_like(t)
    valid[:, :5] = False
    diamond = off_tissue(t, valid, margin_um=10.0, px_um=1.0)
    assert not diamond[50, 60] and diamond[50, 61]  # 10 along the axes ...
    assert not diamond[55, 55] and diamond[56, 55]  # ... and 0.7x that diagonally
    euclid = off_tissue(t, valid, 10.0, 1.0, euclidean=True)
    assert euclid[50, 61] and not euclid[50, 60] and euclid[57, 58] and not euclid[57, 57] and not euclid[56, 55]
    assert not diamond[:, :5].any() and not euclid[:, :5].any()
