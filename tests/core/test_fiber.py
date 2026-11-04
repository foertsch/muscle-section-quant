import numpy as np

from msq.core.fiber import edge_zones, full_res_mask, sector_index


def disc(r, size=61):
    yy, xx = np.mgrid[:size, :size]
    return (yy - size // 2) ** 2 + (xx - size // 2) ** 2 <= r**2


def test_full_res_mask_interpolates_the_coarse_mask_and_cuts_at_half():
    m_ds = np.zeros((5, 5), bool)
    m_ds[1:4, 1:4] = True
    m = full_res_mask(m_ds, 4)
    assert m.shape == (20, 20) and m.dtype == bool
    assert m[10, 10] and not m[0, 0] and 130 <= m.sum() <= 150  # about the 3 x 3 blocks of 4 x 4
    assert m[:, :10].sum() == m[:, 10:].sum() == m[:10].sum() == m[10:].sum()  # centred
    assert full_res_mask(np.ones((3, 3), bool), 2).all()


def test_edge_zones_band_on_both_sides_and_the_core_inside():
    m = disc(20)
    band, edge_in, core = edge_zones(m, px_um=1.0, band_um=1.5, core_um=3.0)
    assert band[30, 50] and band[30, 51] and not band[30, 52] and not band[30, 30]  # on the edge, outside, beyond, centre
    assert (edge_in <= m).all() and (band & m).sum() == edge_in.sum()  # edge_in is the inner half of the band
    assert core[30, 30] and not core[30, 30 + 18] and (core <= m).all() and not (core & band).any()
    assert (core & ~m).sum() == 0 and core.sum() < m.sum()


def test_edge_zones_valid_pixels_only_and_the_core_cap_of_small_fibers():
    m = disc(20)
    valid = np.ones_like(m)
    valid[:, 40:] = False
    band, edge_in, _ = edge_zones(m, 1.0, 1.5, 3.0, valid)
    assert not band[:, 40:].any() and band[:, :40].any() and edge_in[:, 40:].any()  # edge_in ignores validity
    _, _, core = edge_zones(disc(2, 11), 1.0, 1.5, 30.0)  # core_um far beyond the fiber: half its depth instead
    assert core.any() and core[5, 5]


def test_sector_index_covers_the_circle_once():
    yy, xx = np.mgrid[-20:21, -20:21]
    idx = sector_index(yy, xx, 0.0, 0.0, 8)
    assert idx.min() == 0 and idx.max() == 7 and set(np.unique(idx)) == set(range(8))
    assert idx[20, 30] == 4 and idx[20, 10] == 7  # angle 0 -> n / 2; angle pi (just below) -> the last sector
    assert sector_index(np.array([0]), np.array([-5]), 0.0, 0.0, 8)[0] == 7  # exactly pi is clamped to n - 1
    assert np.bincount(idx.ravel()).min() > 150  # no sector is starved (1681 pixels over 8 sectors)
