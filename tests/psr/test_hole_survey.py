
import numpy as np

from msq.psr import hole_survey as hs


def test_holes_inside_fibers_are_found_and_glass_is_not():
    t = np.tile(np.array([0.95, 0.92, 0.19], np.float32), (60, 60, 1))   # yellow fiber
    yellow = np.ones((60, 60), bool)
    t[:, :15] = 1.0                                                      # glass strip on the left (large white)
    yellow[:, :15] = False
    t[30:33, 40:43] = 1.0                                                # a 9 px hole inside the fiber
    yellow[30:33, 40:43] = False
    holes = hs.hole_masks(t, yellow, grid_um=1.0)
    assert holes[31, 41] and not holes[:, :15].any() and holes.sum() == 9


def test_tile_share_counts_only_holey_tiles():
    yellow = np.ones((400, 400), bool)
    holes = np.zeros_like(yellow)
    holes[:200, :200][::5, ::5] = True                                   # 4% holes in the top-left tile only
    yellow[holes] = False
    assert abs(hs.tile_share(holes, yellow, grid_um=1.0) - 0.25) < 0.01
