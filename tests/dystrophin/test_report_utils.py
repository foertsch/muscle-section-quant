
import numpy as np
import pandas as pd
import pytest

from msq.core.report import outlines
from msq.core.report.images import jpeg_uri
from msq.dystrophin import report_utils as ru  # noqa: E402


def test_fiber_boxes_scale_to_full_resolution():
    lab = np.zeros((10, 10), np.uint32)
    lab[2:4, 5:9] = 1
    lab[6:9, 1:3] = 3
    b = ru.fiber_boxes(lab, 4).set_index("label")
    assert list(b.index) == [1, 3]
    assert tuple(b.loc[1]) == (8, 16, 20, 36)
    assert tuple(b.loc[3]) == (24, 36, 4, 12)


def test_outline_path_traces_square():
    m = np.zeros((20, 20), bool)
    m[5:15, 5:15] = True
    p = ru.outline_path(m, oy=100, ox=200)
    assert p.startswith("M") and p.endswith("Z")
    xs, ys = zip(*[map(float, pt.split(",")) for pt in p[1:-1].split("L")])
    assert min(xs) == pytest.approx(204.5, abs=0.6) and max(xs) == pytest.approx(214.5, abs=0.6)
    assert min(ys) == pytest.approx(104.5, abs=0.6) and max(ys) == pytest.approx(114.5, abs=0.6)


def test_the_dystrophin_outline_and_jpeg_defaults_are_the_core_ones_at_1_2_px_and_quality_88():
    yy, xx = np.mgrid[:40, :40]
    disc = (yy - 20) ** 2 + (xx - 20) ** 2 <= 15**2  # a round outline: the tolerance decides how many points it keeps
    assert ru.outline_path(disc, 5, 6) == outlines.outline_path(disc, 5, 6, tol=1.2) != outlines.outline_path(disc, 5, 6, tol=0.2)
    img = np.random.default_rng(0).integers(0, 255, (16, 16), dtype=np.uint8)
    assert ru.jpeg_b64(img) == jpeg_uri(img, 88, True)


def test_candidate_windows_skip_excluded_objects():
    n = 40
    fib = pd.DataFrame({
        "label": np.arange(1, n + 1),
        "by0": np.repeat(np.arange(0, 800, 100), 5)[:n], "by1": np.repeat(np.arange(90, 890, 100), 5)[:n],
        "bx0": np.tile(np.arange(0, 500, 100), 8)[:n], "bx1": np.tile(np.arange(90, 590, 100), 8)[:n],
        "area_um2": 2500.0,  # 40 x 2500 um2 covers 77% of a 360 um window
        "include": True, "flag_bright_cytoplasm": False,
        "actin_edge_sharpness": 0.03, "dys_membrane_specific": 500.0,
    })
    ok = ru.candidate_windows(fib, (1000, 1000), px_um=0.36, size=1000, stride=200, min_n=25)
    assert len(ok) == 1
    fib.loc[3, "include"] = False  # an excluded object inside the only window
    assert ru.candidate_windows(fib, (1000, 1000), px_um=0.36, size=1000, stride=200, min_n=25).empty
