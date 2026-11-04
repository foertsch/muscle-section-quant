
import numpy as np
import pandas as pd
import pytest
import tifffile

from msq.core.synthetic.dystrophin import synthetic_section
from msq.core.tissue import off_tissue
from msq.dystrophin import compare_report as cr  # noqa: E402
from msq.dystrophin import measure as dq
from msq.dystrophin import report_utils as ru


def test_typical_section_is_the_middle_median_with_controls_first():
    sec = pd.DataFrame({
        "sample": ["Q1_1104", "Q2_1104", "Q3_1104", "Q1_1106", "Q2_1106", "Q3_1106", "Q1_1108", "T1_1104"],
        "muscle": ["QUAD"] * 7 + ["TA"],
        "animal": ["1104"] * 3 + ["1106"] * 3 + ["1108", "1104"],
        "group": ["treated"] * 3 + ["ctrl"] * 4 + ["treated"],
        "section": [1, 2, 3, 1, 2, 3, 1, 1],
        "median_dys_membrane_specific": [203, 242, 253, 247, 160, 157, 40, 999],
    })
    out = cr.typical_sections(sec, "QUAD")
    # 1106: 157 < 160 < 247 -> sec2; 1108 has one section; 1104: 203 < 242 < 253 -> sec2; TA ignored
    assert list(out["sample"]) == ["Q2_1106", "Q1_1108", "Q2_1104"]


def fibers(n=101, px_um=0.36):
    fib = pd.DataFrame({"label": np.arange(1, n + 1), "include": True, "area_um2": 1500.0,
                        "dys_membrane_specific": np.arange(n, dtype=float),
                        "centroid_y_px": 1000.0, "centroid_x_px": 1000.0})
    return fib.set_index("label", drop=False)


def test_matched_fibers_take_closest_mid_sized_fiber_that_fits():
    fib = fibers()
    fib.loc[51, "area_um2"] = 100.0       # the exact median (value 50) is too small
    fib.loc[50, "centroid_x_px"] = 20.0   # value 49 sits at the image edge, its close-up would not fit
    fib.loc[2, "include"] = False         # excluded fibers neither set the percentile nor get picked
    picks = cr.matched_fibers(fib, shape=(2000, 2000), px_um=0.36)
    assert [p["q"] for p in picks] == list(cr.QUANTILES)
    inc = fib[fib["include"]]["dys_membrane_specific"]
    for p in picks:
        assert p["target"] == pytest.approx(inc.quantile(p["q"]))
    assert picks[1]["label"] == 52 and picks[1]["value"] == 51.0   # nearest eligible to the median 50
    assert all(p["label"] not in (2, 50, 51) for p in picks)


def test_shared_edges_follow_the_brightest_sample():
    rng = np.random.default_rng(0)
    dim, bright = rng.normal(100, 50, 5000), rng.normal(600, 200, 5000)
    e = cr.shared_edges([dim, dim, bright])
    for v in (dim, bright):
        assert (v >= e[-1]).mean() <= 0.005 and (v < e[0]).mean() <= 0.005


def test_display_ranges_pool_all_sections():
    a = pd.DataFrame({"include": [True, True, False], "dys_membrane_peak": [100.0, 300, 9999],
                      "dys_cytoplasm": [50.0, 70, 9999]})
    b = pd.DataFrame({"include": [True, True], "dys_membrane_peak": [500.0, 700], "dys_cytoplasm": [90.0, 110]})
    r = cr.display_ranges([a, b])
    peak = [100, 300, 500, 700]
    assert r["full"] == (0.0, round(np.percentile(peak, 99), -1))
    assert r["contrast"] == (round(np.percentile([50, 70, 90, 110], 25), -1), round(np.percentile(peak, 95), -1))


def test_pixel_hists_subtract_background_and_split_tissue_from_slide():
    f, bg = 2, 500.0
    gfp = np.full((8, 8), bg, dtype=np.uint16)
    gfp[:4, :] = bg + 110           # tissue rows
    gfp[6, 6] = 0                   # blank padding, skipped
    tissue = np.zeros((4, 4), bool)
    tissue[:2] = True
    off = np.zeros((4, 4), bool)
    off[3] = True
    e = np.arange(-200, 301, 25.0)
    res = cr.pixel_hists(gfp, bg, tissue, off, f, e)
    assert res["median_tissue"] == pytest.approx(110)
    assert res["tissue"][np.searchsorted(e, 100, side="right") - 1] == pytest.approx(100)
    assert res["off"][np.searchsorted(e, 0, side="right") - 1] == pytest.approx(100)  # zero pixel not counted
    assert res["sd_off"] == pytest.approx(0)


def test_sample_stats_positive_share_matches_the_threshold():
    calls = pd.DataFrame({"call_50": ["positive", "negative", "negative", "not called"],
                          "call_60": ["positive", "negative", "negative", "not called"],
                          "call_70": ["negative", "negative", "negative", "not called"],
                          "score_50": [700.0, 600, 100, np.nan], "score_60": [650.0, 300, 50, np.nan],
                          "score_70": [500.0, 200, 10, np.nan]})
    inc = pd.DataFrame({"dys_membrane_specific": [10.0, 20, 30], "dys_cytoplasm": 1.0, "dys_membrane_peak": 2.0})
    row = pd.Series({"sample": "s", "animal": "1104", "group": "treated", "section": 2})
    st = cr.sample_stats(row, inc, calls)
    assert st["n_called"] == 3 and st["n_included"] == 3
    assert st["pct_positive_50"] == pytest.approx(100 / 3)
    assert st["pct_positive_60"] == pytest.approx(100 / 3)
    assert st["pct_positive_70"] == 0
    assert st["dys_median"] == 20


def test_sample_stats_count_the_calls_so_controls_show_their_held_out_calls():
    """A control section is called against the threshold from the other control animals (full_summary); the page shows
    those calls, not the scores against the threshold from all controls."""
    calls = pd.DataFrame({k: v for k, v in {
        "call_50": ["positive", "negative"], "call_60": ["negative", "negative"], "call_70": ["negative", "negative"],
        "score_50": [600.0, 500], "score_60": [400.0, 300], "score_70": [200.0, 100]}.items()})
    inc = pd.DataFrame({"dys_membrane_specific": [10.0, 20], "dys_cytoplasm": 1.0, "dys_membrane_peak": 2.0})
    row = pd.Series({"sample": "s", "animal": "0001", "group": "ctrl", "section": 1})
    st = cr.sample_stats(row, inc, calls)  # the held-out call is positive whatever the muscle's threshold is
    assert st["pct_positive_50"] == 50.0


def test_write_html_embeds_payload(tmp_path):
    out = tmp_path / "r.html"
    cr.write_html({"muscle": "QUAD", "threshold": 606}, out)
    html = out.read_text()
    assert 'const DATA = {"muscle":"QUAD","threshold":606};\n' in html
    assert "/*DATA*/" not in html and "<!--ASSET" not in html


def test_typical_window_matches_section_median():
    cands = pd.DataFrame({"y0": [0, 200, 400], "x0": [0, 0, 0], "local_median": [120.0, 250.0, 346.0]})
    w = cr.typical_window(cands, section_median=242.0)
    assert (w["y0"], w["local_median"]) == (200, 250.0)


def test_robust_sd_ignores_debris():
    rng = np.random.default_rng(0)
    v = np.r_[rng.normal(0, 10, 10000), np.full(300, 3000.0)]   # 3% bright debris
    assert cr.robust_sd(v) == pytest.approx(10, rel=0.08)
    assert np.std(v) > 100


def test_a_fiber_outline_is_drawn_at_full_resolution_relative_to_the_window():
    labels = np.zeros((12, 14), np.uint32)
    labels[3:7, 4:9] = 1
    labels[0:2, 10:14] = 2  # at the image corner: the crop is clipped, not padded
    boxes = ru.fiber_boxes(labels, 4).itertuples()
    paths = {r.label: cr.outline(labels, 4, r, 8, 12) for r in boxes}
    assert paths[1] == "M22.0,19.5L5.0,19.5L3.5,18.0L3.5,5.0L5.0,3.5L22.0,3.5L23.5,5.0L23.5,18.0L22.0,19.5Z"
    assert paths[2] == "M43.0,-0.5L29.0,-0.5L27.5,-8.0L43.0,-8.5L43.0,-0.5Z"


def test_pixel_masks_come_from_the_run_files_or_are_recomputed_from_the_actin(tmp_path):
    s = synthetic_section(11, "ctrl", size=512)
    f, px_um = 4, s["px_um"]
    p = dq.params_for(px_um)
    act_ds = dq.downsample(s["act"], f)
    tissue, off = dq.tissue_mask(act_ds, px_um * f, p.tissue_margin_um)
    section, got_off = cr.pixel_masks(tmp_path, s["act"], f, px_um, p)  # a run made before the files existed
    assert np.array_equal(section, dq.main_section(tissue, px_um * f)) and np.array_equal(got_off, off)
    assert section.dtype == got_off.dtype == bool

    stored = np.zeros_like(tissue)
    stored[10:20, 10:20] = True  # not what the actin gives: proves the file is what is read
    for kind, mask in (("tissue", tissue), ("section", stored)):
        tifffile.imwrite(tmp_path / f"{kind}_ds{f}.tif", mask.astype(np.uint8), compression="zlib")
    section, got_off = cr.pixel_masks(tmp_path, s["act"], f, px_um, p)
    assert np.array_equal(section, stored) and section.dtype == got_off.dtype == bool
    assert np.array_equal(got_off, off_tissue(tissue, act_ds > 0, p.tissue_margin_um, px_um * f))
    assert np.array_equal(got_off, off)  # grown from the stored tissue, it is the mask the recomputation gives
