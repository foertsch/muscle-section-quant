
import numpy as np

from msq.psr import click_windows as cw
from msq.psr import section as sec


def test_window_lies_inside_the_section_and_on_tissue():
    sect = np.zeros((400, 400), bool)
    sect[50:350, 50:350] = True
    classes = np.zeros((400, 400, 4), np.uint8)
    classes[..., sec.YELLOW] = 100
    classes[50:200, 50:350, sec.YELLOW] = 0   # top half of the section unstained
    corner = cw.pick_window(sect, classes, grid_um=1.0, rng=np.random.default_rng(0), size_um=60, edge_um=10)
    y0, x0 = corner
    assert 60 <= y0 and y0 + 60 <= 340 and 60 <= x0 and x0 + 60 <= 340   # 10 um from the section edge
    assert y0 >= 200                                                      # on stained tissue only


def test_no_window_when_the_section_is_too_small():
    sect = np.zeros((100, 100), bool)
    sect[40:60, 40:60] = True
    classes = np.full((100, 100, 4), 100, np.uint8)
    assert cw.pick_window(sect, classes, 1.0, np.random.default_rng(0), size_um=50, edge_um=5) is None


def test_overlay_is_transparent_except_outlines():
    lab = np.zeros((20, 20), np.uint32)
    lab[5:15, 5:15] = 3
    ov = cw.outline_overlay(lab)
    assert ov[10, 10, 3] == 0 and ov[5, 10, 3] == 255 and ov[0, 0, 3] == 0
