import json

import numpy as np
import pandas as pd
import pytest
import tifffile

from msq.dystrophin import measure as dq  # noqa: E402

PX = 0.36  # um/px, as in the 20x scans
BG = 500.0


def disk(shape, cy, cx, r):
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r**2


def synthetic_fiber(r=60, size=200, cyto=1500, membrane=2500, actin=6000, hollow=False):
    """Disk fiber with a 3-px dystrophin ring on its edge and uniform actin inside."""
    c = size / 2
    mask = disk((size, size), c, c, r)
    ring = mask & ~disk((size, size), c, c, r - 3)
    gfp = np.full((size, size), BG, np.float32)
    gfp[mask] = cyto
    gfp[ring] = membrane
    act = np.full((size, size), BG, np.float32)
    act[mask] = actin
    if hollow:
        act[disk((size, size), c, c, r - 15)] = BG
    return mask, gfp.astype(np.uint16), act.astype(np.uint16)


def test_measure_fiber_recovers_membrane_and_cytoplasm():
    mask, gfp, act = synthetic_fiber()
    r = dq.measure_fiber(mask, gfp, act, PX, dq.Params(), BG, BG, sat=16383)
    assert r["dys_cytoplasm"] == pytest.approx(1500 - BG, rel=0.01)
    # the ridge estimate sits near the ring intensity (smoothing pulls it down slightly)
    assert 2200 - BG < r["dys_membrane_peak"] <= 2500 - BG
    assert r["dys_membrane_specific"] == pytest.approx(r["dys_membrane_peak"] + BG - 1500)
    assert r["dys_sector_cv"] < 0.05
    assert r["n_sectors_measured"] == 36
    assert r["actin_core_to_edge"] == pytest.approx(1.0, abs=0.01)
    assert r["area_um2"] == pytest.approx(mask.sum() * PX**2)
    assert not r["touches_void"]


def test_saved_sectors_are_the_values_behind_the_median():
    mask, gfp, act = synthetic_fiber()
    half = np.zeros_like(mask)
    half[:, 100:] = True
    gfp[mask & half & (gfp == 2500)] = 1500  # dystrophin on one half of the ring only
    r = dq.measure_fiber(mask, gfp, act, PX, dq.Params(), BG, BG, sat=16383)
    sec = r["_sectors"]
    assert sec.shape == (36, 2)
    assert np.nanmedian(sec[:, 1]) == pytest.approx(r["dys_membrane_specific"])
    assert np.nanmedian(sec[:, 0]) == pytest.approx(r["dys_membrane_peak"])
    # about half of the 36 wedges carry the ring
    assert np.nanmean(sec[:, 1] > 300) == pytest.approx(0.5, abs=0.1)


def test_measure_all_collects_sectors():
    mask, gfp, act = synthetic_fiber(size=200)
    labels_ds = (dq.downsample(mask, 4) > 0.5).astype(np.uint32)
    sectors = []
    df = dq.measure_all(labels_ds, gfp, act, PX, dq.Params(), BG, BG, sat=16383, progress=False,
                        sectors_out=sectors)
    assert len(sectors) == 1 and sectors[0].shape == (36, 4)
    assert (sectors[0][:, 0] == 1).all() and list(sectors[0][:, 1]) == list(range(36))
    assert np.nanmedian(sectors[0][:, 3]) == pytest.approx(df.loc[0, "dys_membrane_specific"])
    assert "_sectors" not in df.columns


def test_measure_fiber_patchy_membrane_raises_sector_cv():
    mask, gfp, act = synthetic_fiber()
    intact = dq.measure_fiber(mask, gfp, act, PX, dq.Params(), BG, BG, sat=16383)
    half = np.zeros_like(mask)
    half[:, 100:] = True
    gfp[mask & half & (gfp == 2500)] = 1500  # dystrophin missing on one side
    patchy = dq.measure_fiber(mask, gfp, act, PX, dq.Params(), BG, BG, sat=16383)
    assert patchy["dys_sector_cv"] > 0.2
    assert patchy["dys_sector_cv"] > 5 * intact["dys_sector_cv"]


def test_hollow_object_has_low_core_to_edge_actin():
    mask, gfp, act = synthetic_fiber(hollow=True)
    r = dq.measure_fiber(mask, gfp, act, PX, dq.Params(), BG, BG, sat=16383)
    assert r["actin_core_to_edge"] < dq.Params().hollow_ratio


def test_zero_padding_is_flagged():
    mask, gfp, act = synthetic_fiber()
    gfp[:, :45] = 0
    r = dq.measure_fiber(mask, gfp, act, PX, dq.Params(), BG, BG, sat=16383)
    assert r["touches_void"]


def test_padding_in_the_outer_band_alone_is_not_flagged():
    """Pins today's rule (see the comment in measure_fiber): only the outline counts, so changing it is deliberate."""
    mask, gfp, act = synthetic_fiber()  # the outline starts at x = 40
    gfp[:, :38] = 0  # padding 2 px outside the outline, inside the 1.5 um outer band
    r = dq.measure_fiber(mask, gfp, act, PX, dq.Params(), BG, BG, sat=16383)
    assert not r["touches_void"] and r["n_sectors_measured"] == 36


def test_measure_all_maps_labels_to_full_resolution():
    f = 4
    size = 400
    mask_a = disk((size, size), 100, 100, 60)
    mask_b = disk((size, size), 260, 280, 70)
    gfp = np.full((size, size), BG, np.float32)
    act = np.full((size, size), BG, np.float32)
    for m in (mask_a, mask_b):
        gfp[m] = 1500
        act[m] = 6000
    labels_ds = np.zeros((size // f, size // f), np.uint32)
    labels_ds[dq.downsample(mask_a, f) > 0.5] = 1
    labels_ds[dq.downsample(mask_b, f) > 0.5] = 2
    df = dq.measure_all(labels_ds, gfp.astype(np.uint16), act.astype(np.uint16), PX, dq.Params(),
                        BG, BG, sat=16383, progress=False)
    assert list(df["label"]) == [1, 2]
    assert df.loc[0, "area_um2"] == pytest.approx(mask_a.sum() * PX**2, rel=0.05)
    assert df.loc[1, "centroid_y_px"] == pytest.approx(260, abs=2)
    assert df.loc[1, "centroid_x_px"] == pytest.approx(280, abs=2)


def test_stretch_scales_between_the_percentiles():
    from msq.dystrophin import qc

    a = np.arange(1001, dtype=np.uint16)
    s = qc.stretch(a, 10, 90)
    assert s.min() == 0 and s.max() == 1  # clipped outside the percentiles
    assert s[500] == pytest.approx(0.5)
    assert qc.stretch(np.full(5, 7, np.uint16)).max() == 0  # a flat crop stays dark, no division by zero


def test_save_writes_the_figure_with_the_given_options_and_closes_it(tmp_path):
    import matplotlib.pyplot as plt

    from msq.dystrophin import qc

    fig, ax = plt.subplots(figsize=(2, 1))
    ax.plot([0, 1])
    qc.save(fig, tmp_path / "f.png", dpi=50)
    assert plt.imread(tmp_path / "f.png").shape[:2] == (50, 100)  # figsize x dpi
    assert not plt.fignum_exists(fig.number)


def test_fiber_geometry_shows_the_sector_values_the_measurement_uses():
    """What the explainer scripts draw is what measure_all saved: the same ring pieces, minus the same cytoplasm."""
    p, size = dq.Params(), 200
    mask, gfp, act = synthetic_fiber(size=size)
    labels = np.zeros((size // p.downsample,) * 2, np.uint32)
    labels[dq.downsample(mask, p.downsample) > 0.5] = 1
    saved = []
    dq.measure_all(labels, gfp, act, PX, p, BG, BG, sat=16383, progress=False, sectors_out=saved)
    box = pd.Series({"label": 1, "by0": 0, "bx0": 0, "by1": size, "bx1": size})
    geo = dq.fiber_geometry(labels, box, gfp, p.downsample, PX, p)
    # saved columns: label, sector, ridge_bgsub, ridge_minus_cyto
    assert np.allclose(geo["sec"], saved[0][:, 3], equal_nan=True)
    assert geo["mask"].shape == geo["g"].shape == geo["sec_img"].shape and geo["core"].any()
    assert np.isfinite(geo["sec_img"]).sum() > 0 and geo["outline"].shape[1] == 2


def test_a_section_run_keeps_its_tissue_and_main_section_masks(tmp_path):
    """The reports and the explainers read these instead of finding the tissue again (not compared by the goldens)."""
    from msq.core.segment import FixedLabels
    from msq.core.synthetic.dystrophin import synthetic_section

    s = synthetic_section(5, "treated", size=512)
    scan = tmp_path / "scan.nd2"
    scan.write_bytes(b"")
    dq.run(scan, tmp_path / "out", name="S1", segment=FixedLabels(s["labels_ds"]),
           read=lambda path: {k: s[k] for k in ("gfp", "act", "px_um", "sat", "channels")})
    d = tmp_path / "out" / "S1"
    tissue, section = (tifffile.imread(d / f"{n}_ds4.tif") for n in ("tissue", "section"))
    act_ds = dq.downsample(s["act"], 4)
    want, _ = dq.tissue_mask(act_ds, s["px_um"] * 4, dq.Params().tissue_margin_um)
    assert tissue.dtype == section.dtype == np.uint8 and tissue.shape == section.shape == s["labels_ds"].shape
    assert np.array_equal(tissue, want) and np.array_equal(section, dq.main_section(want, s["px_um"] * 4))
    assert (d / "run_info.json").exists() and tissue.any() and section.sum() >= tissue.sum() * 0.5


def test_is_done_needs_all_outputs(tmp_path):
    for f in ("fibers.csv", "sectors.csv.gz"):
        (tmp_path / f).write_text("x")
    assert not dq.is_done(tmp_path)  # interrupted before run_info.json
    (tmp_path / "run_info.json").write_text("{}")
    assert dq.is_done(tmp_path)


def test_is_done_with_a_scan_checks_that_the_results_come_from_it(tmp_path):
    scan = tmp_path / "scans" / "TA sec2 1104 20x.nd2"
    scan.parent.mkdir()
    scan.write_bytes(b"x" * 10)
    d = tmp_path / "TA_sec2_1104_20x_treated"
    d.mkdir()
    for f in ("fibers.csv", "sectors.csv.gz"):
        (d / f).write_text("x")
    (d / "run_info.json").write_text(json.dumps({"source": "Z:\\scans\\TA sec2 1104 20x.nd2", "source_size": 10}))
    assert dq.is_done(d, scan)  # same file name and size; the folder and the separator do not matter
    assert not dq.is_done(d, scan.with_name("TA sec2 1104 20x rescan.nd2"))  # the table points to another scan
    scan.write_bytes(b"x" * 11)
    assert not dq.is_done(d, scan) and dq.is_done(d)  # the scan was replaced
    (d / "run_info.json").write_text(json.dumps({"source": "/old/TA sec2 1104 20x.nd2"}))  # results from before sizes
    assert dq.is_done(d, scan)


def test_params_follow_pixel_size():
    p20, p4 = dq.params_for(0.36), dq.params_for(1.83)
    assert p20 == dq.Params()
    assert p4.downsample == 1 and p4.n_sectors < p20.n_sectors
    # the 4x band must still contain the pixels one step either side of the mask edge
    assert 1.83 <= p4.band_um < 2 * 1.83
    assert p4.inverted_threshold < p20.inverted_threshold


def test_measure_fiber_at_4x_resolution():
    px = 1.83
    mask = disk((60, 60), 30, 30, 13)  # ~48 um fiber
    ring = mask & ~disk((60, 60), 30, 30, 12)
    gfp = np.full((60, 60), BG, np.float32)
    gfp[mask] = 700
    gfp[ring] = 900
    act = np.where(mask, 3000, BG).astype(np.uint16)
    r = dq.measure_fiber(mask, gfp.astype(np.uint16), act, px, dq.params_for(px), BG, BG, sat=16383)
    assert r["n_sectors_measured"] == 16
    assert r["dys_cytoplasm"] == pytest.approx(700 - BG, rel=0.02)
    assert 0 < r["dys_membrane_specific"] <= 200


def test_add_flags_excludes_and_flags():
    n = 50
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "area_um2": rng.uniform(500, 3000, n),
        "touches_border": False,
        "touches_void": False,
        "actin_core_to_edge": 1.1,
        "actin_core": 6000.0,
        "dys_cytoplasm": rng.normal(1000, 50, n),
        "dys_membrane_specific": rng.normal(400, 100, n),
    })
    df.loc[0, "area_um2"] = 10  # debris
    df.loc[1, "actin_core_to_edge"] = 0.4  # vessel
    df.loc[2, "actin_core"] = 500  # dark object
    df.loc[3, "dys_cytoplasm"] = 8000  # necrotic / IgG+ fiber
    df.loc[4, "dys_membrane_specific"] = -300  # interior far brighter than ring
    df.loc[5, "dys_membrane_specific"] = -40  # ring-less fiber near zero: kept
    df["in_main_section"] = True
    df.loc[6, "in_main_section"] = False  # fiber on a neighbouring section
    out = dq.add_flags(df, dq.Params())
    assert out.loc[0, "excl_size"] and out.loc[1, "excl_hollow"] and out.loc[2, "excl_low_actin"]
    assert out.loc[3, "flag_bright_cytoplasm"]
    assert out.loc[4, "excl_inverted"] and not out.loc[5, "excl_inverted"]
    assert out.loc[6, "excl_off_section"]
    assert not out.loc[[0, 1, 2, 3, 4, 6], "include"].any()
    assert out.loc[5, "include"] and out.loc[7:, "include"].all()
