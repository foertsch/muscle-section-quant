import re
from pathlib import Path

import numpy as np
import pandas as pd

from msq.dystrophin import decision_report as dr  # noqa: E402


def decode_sorted(deltas) -> np.ndarray:
    """What the page does (decode() in decision_template.html): a running sum, back to the 0.1 a.u. grid."""
    return np.cumsum(np.asarray(deltas, np.int64)) / 10


def test_encoding_round_trips_scores_exactly():
    v = np.array([470.7, -12.3, 0.0, 1203.4, np.nan, 470.7, 99.9])
    d = dr.encode_sorted(v)
    assert all(isinstance(x, int) for x in d)
    out = decode_sorted(d)
    assert np.array_equal(out, np.sort(v[np.isfinite(v)]))
    # same doubles as the CSV values, so comparisons against the threshold behave as in full_summary.py
    assert out[-2] == 470.7 and (out >= 470.7).sum() == 3


def test_encoding_keeps_the_page_small():
    rng = np.random.default_rng(0)
    v = np.round(rng.normal(200, 150, 20000), 1)
    d = dr.encode_sorted(v)
    assert max(d[1:]) < 1000 and len(",".join(map(str, d))) < 3 * len(v)


def test_threshold_range_spans_the_slider():
    ctrl = np.arange(1001, dtype=float)
    lo, hi = dr.threshold_range(ctrl, (90, 99.5))
    assert (lo, hi) == (900.0, 995.0)


def test_borderline_pool_covers_the_threshold_range_without_duplicates():
    labels = np.arange(1, 401)
    s50 = np.linspace(0, 1200, 400)
    scores = pd.DataFrame({"score_50": s50, "score_60": s50 - 80, "score_70": s50 - 160}, index=labels)
    pool = dr.borderline_pool(scores, lo=400, hi=800, n_targets=5)
    assert len(pool) == len(set(pool))
    spacing = (800 - 400) / 4
    targets = np.linspace(400 - spacing, 800 + spacing, 7)
    for k in dr.LEVELS:
        picked = scores.loc[pool, f"score_{k}"].to_numpy()
        for t in targets:   # every target has a pooled fiber within one bin width (3 a.u.) at every level
            assert np.abs(picked - t).min() <= 1200 / 399 + 1e-9


def test_borderline_pool_skips_targets_without_nearby_fibers():
    scores = pd.DataFrame({k: [10.0, 20.0, 900.0] for k in ("score_50", "score_60", "score_70")}, index=[1, 2, 3])
    pool = dr.borderline_pool(scores, lo=400, hi=600, n_targets=3)
    assert pool == []   # nothing within 100 a.u. of the targets 300 to 700


def test_fiber_record_keeps_scores_and_status():
    r = pd.Series({"label": 7, "include": True, "flag_bright_cytoplasm": False, "dys_membrane_specific": 312.34,
                   "area_um2": 1500.4})
    calls = pd.DataFrame({"score_50": [480.0], "score_60": [400.1], "score_70": [np.nan]}, index=[7])
    rec = dr.fiber_record(next(pd.DataFrame([r]).itertuples()), calls)
    assert rec == {"label": 7, "status": "included", "dys": 312.3, "area": 1500, "s50": 480.0, "s60": 400.1, "s70": None}
    missing = dr.fiber_record(next(pd.DataFrame([{**r, "label": 8, "include": False}]).itertuples()), calls)
    assert missing["status"] == "excluded" and missing["s50"] is None


def test_window_by_share_matches_the_section_share():
    # three windows of 10 fibers each: 0, 3 and 8 positive
    fib = pd.DataFrame({"by0": 0, "by1": 10, "bx0": np.repeat([0, 100, 200], 10) + 1, "bx1": 0},
                       index=pd.Index(range(1, 31), name="label"))
    fib["bx1"] = fib["bx0"] + 10
    positive = pd.Series(0.0, index=fib.index)
    positive[[11, 12, 13] + list(range(21, 29))] = 1.0
    cands = pd.DataFrame({"y0": 0, "x0": [0, 100, 200], "local_median": [100.0, 200.0, 300.0]})
    w = dr.window_by_share(cands, fib, positive, target=25.0, section_median=150.0, size=50)
    assert (w["x0"], w["share"]) == (100, 30.0)
    # equal shares: the window nearer the section median wins
    cands2 = pd.DataFrame({"y0": 0, "x0": [100, 100], "local_median": [400.0, 160.0]})
    assert dr.window_by_share(cands2, fib, positive, 30.0, 150.0, size=50)["local_median"] == 160.0


def test_overview_factor_keeps_the_longest_side_within_the_limit():
    assert dr.overview_factor((16000, 9000), f=4, max_px=1000) == 4      # 4000 ds px -> 1000
    assert dr.overview_factor((16004, 9000), f=4, max_px=1000) == 5      # just over -> next block size
    assert dr.overview_factor((2000, 1000), f=4, max_px=1000) == 1


def test_fiber_dots_place_called_fibers_and_keep_scores_exact():
    cents = pd.DataFrame({"centroid_y_px": [100.0, 400.0, 999.0], "centroid_x_px": [62.0, 810.0, 0.0]},
                         index=pd.Index([1, 2, 3], name="label"))
    calls = pd.DataFrame({"score_50": [470.7, 12.3], "score_60": [400.0, np.nan], "score_70": [-5.4, 0.0]}, index=[2, 1])
    d = dr.fiber_dots(cents, calls, scale=20)
    assert d["fx"] == [3, 40] and d["fy"] == [5, 20]                     # label 3 is not called, so not drawn
    assert d["f50"] == [123, 4707] and d["f60"] == [None, 4000] and d["f70"] == [0, -54]
    assert d["f50"][1] / 10 == 470.7                                    # same double the page compares to the threshold


def test_the_report_templates_carry_no_facts_of_one_dataset():
    """The templates serve every run; facts about one dataset (its animals, notes about single animals) belong in
    that dataset's documentation. Structural check: no animal number (1000 to 1999) in a template; everything about a
    run comes from its data."""
    package = Path(dr.__file__).parent
    for name in ("decision_template.html", "compare_template.html"):
        text = (package / name).read_text(encoding="utf-8")
        assert "/*DATA*/null" in text, name  # the page's data is filled in per run
        numbers = re.findall(r"(?<![\d.#])(1\d{3})(?![\d.]|px)", text)  # layout sizes (1400px) are not animals
        assert set(numbers) <= {"1000"}, name  # 1000 converts mm to um
