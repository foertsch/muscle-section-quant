
import numpy as np
import pandas as pd
import pytest

from msq.dystrophin import full_summary as fs  # noqa: E402


def fake_section(rng, n_fibers, positive_frac, start_label=1):
    """Fibers with 36 sectors each: negatives ~N(100, 120), positives ~N(900, 150)."""
    labels = np.arange(start_label, start_label + n_fibers)
    pos = rng.random(n_fibers) < positive_frac
    rows = []
    for lab, p in zip(labels, pos):
        v = rng.normal(900, 150, 36) if p else rng.normal(100, 120, 36)
        rows += [(lab, i, x) for i, x in enumerate(v)]
    sectors = pd.DataFrame(rows, columns=["label", "sector", "ridge_minus_cyto"])
    fibers = pd.DataFrame({"sector_measured_frac": 1.0, "dys_membrane_specific": np.where(pos, 900.0, 100.0),
                           "area_um2": 1500.0}, index=pd.Index(labels, name="label"))
    return {"sectors": sectors, "fibers": fibers}, pos


@pytest.fixture
def dataset():
    rng = np.random.default_rng(1)
    samples, meta, truth = {}, [], {}
    for a in ("1106", "1107", "1108"):
        s = f"TA_sec1_{a}_20x_ctrl"
        samples[s], truth[s] = fake_section(rng, 400, 0.01)  # ~1% revertant-like positives
        meta.append({"sample": s, "group": "ctrl", "muscle": "TA", "animal": a, "section": 1})
    for a in ("1104", "1105"):
        for sec in (1, 2):
            s = f"TA_sec{sec}_{a}_20x_treated"
            samples[s], truth[s] = fake_section(rng, 400, 0.4)
            meta.append({"sample": s, "group": "treated", "muscle": "TA", "animal": a, "section": sec})
    meta = pd.DataFrame(meta).set_index("sample")
    return samples, meta, truth, fs.fiber_scores(samples)


def test_one_threshold_with_falling_control_false_positives(dataset):
    samples, meta, _, scores = dataset
    ctrl = list(meta.index[meta["group"] == "ctrl"])
    th = fs.calibrate_muscle(scores, ctrl)
    thresholds = {th[c]["sector_threshold"] for c in fs.COVERAGES}
    assert len(thresholds) == 1  # the same threshold at every stringency level
    pooled = pd.concat([scores[s]["score_50"] for s in ctrl])
    assert fs.FPR == 0.02 and thresholds.pop() == pytest.approx(np.percentile(pooled, 98))
    assert th[0.5]["ctrl_fpr_pooled"] == pytest.approx(fs.FPR, abs=0.003)  # calibrated at 50%
    assert th[0.5]["ctrl_fpr_pooled"] >= th[0.6]["ctrl_fpr_pooled"] >= th[0.7]["ctrl_fpr_pooled"]
    for c in fs.COVERAGES:
        assert th[c]["n_ctrl_fibers"] == 1200


def test_stricter_levels_are_nested(dataset):
    samples, meta, _, scores = dataset
    ctrl = list(meta.index[meta["group"] == "ctrl"])
    th = fs.calibrate_muscle(scores, ctrl)
    for s in meta.index:
        c = fs.call_fibers(samples[s], scores[s], th)
        p50, p60, p70 = (c[f"call_{k}"] == "positive" for k in ("50", "60", "70"))
        assert not (p70 & ~p60).any() and not (p60 & ~p50).any()


def test_leave_one_animal_out_scores_each_control(dataset):
    _, meta, _, scores = dataset
    ctrl = list(meta.index[meta["group"] == "ctrl"])
    lo = fs.leave_one_animal_out(scores, ctrl, meta["animal"].to_dict())
    assert sorted(lo["held_out_animal"].unique()) == ["1106", "1107", "1108"]
    assert len(lo) == 3 * len(fs.COVERAGES)
    assert (lo["fpr"] < fs.FPR + 0.04).all()


def test_calls_and_tables(dataset):
    samples, meta, truth, scores = dataset
    ctrl = list(meta.index[meta["group"] == "ctrl"])
    th = fs.calibrate_muscle(scores, ctrl)
    calls = {s: fs.call_fibers(samples[s], scores[s], th) for s in meta.index}
    c0 = calls["TA_sec1_1104_20x_treated"]
    assert set(c0["call_50"]) <= {"positive", "negative", "not called"}
    assert ((c0["score_50"] >= th[0.5]["sector_threshold"]) == (c0["call_50"] == "positive")).all()
    sec = fs.section_table(calls, meta)
    for _, r in sec[sec["group"] == "treated"].iterrows():
        true_pos = 100 * truth[r["sample"]].mean()
        # every true positive is found; on top, about FPR of the negatives pass, as in the controls
        expected = true_pos + fs.FPR * (100 - true_pos)
        assert r["pct_positive_50"] == pytest.approx(expected, abs=1.5)
    animals = fs.animal_table(sec)
    t = animals[animals["animal"] == "1104"].iloc[0]
    assert t["n_sections"] == 2
    assert t["pct_positive_50"] == pytest.approx(sec.loc[sec["animal"] == "1104", "pct_positive_50"].mean())
    assert t["pct_positive_50_pooled"] == pytest.approx(100 * t["n_positive_50"] / t["n_called"])


def test_summarize_with_a_borrowed_threshold(tmp_path):
    from msq.core.synthetic.dystrophin import write_section

    names = ["TA_sec1_1104_20x_treated", "TA_sec2_1104_20x_treated"]
    for s in names:
        write_section(tmp_path / "sections", s)
    meta = pd.DataFrame({"sample": names, "group": "treated", "muscle": "TA", "animal": "1104",
                         "section": [1, 2]}).set_index("sample")
    with pytest.raises(ValueError, match="no control sections"):
        fs.summarize(tmp_path / "sections", meta, tmp_path / "t0", tmp_path / "f0")
    out = fs.summarize(tmp_path / "sections", meta, tmp_path / "tables", tmp_path / "figures",
                       borrowed={"TA": {"threshold": 449.0, "source": "run A"}})
    th = out["thresholds"]["TA"]
    assert th[0.5]["sector_threshold"] == 449.0 and th[0.5]["ctrl_fpr_pooled"] is None
    assert out["sources"]["TA"] == "borrowed from run A"
    assert pd.read_csv(tmp_path / "tables" / "leave_one_animal_out.csv").empty
    assert (tmp_path / "figures" / "score_distributions.png").exists()
    assert fs.ctrl_passing(th) == "threshold borrowed, no controls in this run"


def test_fibers_with_too_few_sectors_are_not_called(tmp_path):
    """Fewer than 24 of 36 sectors measured: no score, so the fiber neither moves the threshold nor counts in the
    percentages, even when the sectors it has are very bright."""
    from msq.core.synthetic.dystrophin import write_section

    animals = {"TA_sec1_1106_20x_ctrl": "1106", "TA_sec1_1107_20x_ctrl": "1107", "TA_sec1_1104_20x_treated": "1104"}
    for s in animals:
        write_section(tmp_path / "sections", s, n_fibers=300, few_sectors=10)
    meta = pd.DataFrame({"sample": list(animals), "group": [s.rsplit("_", 1)[1] for s in animals], "muscle": "TA",
                         "animal": list(animals.values()), "section": 1}).set_index("sample")
    out = fs.summarize(tmp_path / "sections", meta, tmp_path / "tables", tmp_path / "figures")

    calls = pd.read_csv(tmp_path / "tables" / "calls.csv.gz")
    short = calls["label"] > 290
    assert (calls.loc[short, ["call_50", "call_60", "call_70"]] == "not called").all().all()
    assert calls.loc[short, "score_50"].isna().all() and calls.loc[~short, "score_50"].notna().all()
    ctrl50 = calls.loc[calls["sample"].str.endswith("_ctrl") & ~short, "score_50"]
    assert out["thresholds"]["TA"][0.5]["sector_threshold"] == pytest.approx(np.percentile(ctrl50, 98))
    sec = out["sections"].set_index("sample")
    assert (sec["n_fibers"] == 300).all() and (sec["n_called"] == 290).all()
    assert (sec["pct_positive_50"] == 100 * sec["n_positive_50"] / 290).all()


# ---------------------------------------------------------------- control animals by their held-out calls (review C3)


def summary_of(tmp_path, sections: list[tuple[str, str, str, int]], borrowed=None):
    """summarize on fake sections (muscle, group, animal, section), as the runner calls it."""
    from msq.core.synthetic.dystrophin import write_section

    rows = []
    for m, g, a, sec in sections:
        s = f"{m}_sec{sec}_{a}_20x_{g}"
        write_section(tmp_path / "results", s, n_fibers=250)
        rows.append({"sample": s, "group": g, "muscle": m, "animal": a, "section": sec})
    meta = pd.DataFrame(rows).set_index("sample")
    out = fs.summarize(tmp_path / "results", meta, tmp_path / "tables", tmp_path / "figs", borrowed=borrowed)
    return out, meta


DESIGN = [("TA", "ctrl", a, s) for a in ("0001", "0002", "0003") for s in (1, 2)] + \
         [("TA", "treated", a, s) for a in ("0101", "0102") for s in (1, 2)]


def test_control_animals_are_called_against_the_threshold_from_the_other_controls(tmp_path):
    out, meta = summary_of(tmp_path, DESIGN)
    tables = tmp_path / "tables"
    loao = pd.read_csv(tables / "leave_one_animal_out.csv", dtype={"held_out_animal": str})
    calls = pd.read_csv(tables / "calls.csv.gz")
    sections = pd.read_csv(tables / "sections.csv", dtype={"animal": str})
    animals = pd.read_csv(tables / "animals.csv", dtype={"animal": str})
    threshold = out["thresholds"]["TA"][fs.THRESHOLD_COVERAGE]["sector_threshold"]
    for a in ("0001", "0002", "0003"):
        held = loao[(loao["held_out_animal"] == a) & (loao["coverage"] == 0.5)].iloc[0]
        own = sections[sections["animal"] == a]
        assert set(own["threshold"]) == {held["sector_threshold"]}
        assert set(own["threshold_from"]) == {"other control animals"}
        assert set(calls.loc[calls["sample"].isin(own["sample"]), "threshold"]) == {held["sector_threshold"]}
        row = animals[animals["animal"] == a].iloc[0]
        assert row["threshold_from"] == "other control animals"
        for c in fs.COVERAGES:  # the pooled share is exactly the held-out false-positive rate
            fpr = loao[(loao["held_out_animal"] == a) & (loao["coverage"] == c)]["fpr"].iloc[0]
            assert row[f"pct_positive_{fs.pct(c)}_pooled"] == pytest.approx(100 * fpr)
    treated = sections[sections["group"] == "treated"]
    assert set(treated["threshold"]) == {threshold} and set(treated["threshold_from"]) == {"all controls"}
    # treated numbers are those of the threshold from all controls, as before
    samples = fs.cal.load_results(tmp_path / "results")
    scores = fs.fiber_scores(samples)
    for s in treated["sample"]:
        old = fs.call_fibers(samples[s], scores[s], out["thresholds"]["TA"])
        new = calls[calls["sample"] == s].reset_index(drop=True)
        assert (old["call_50"] == new["call_50"]).all() and (old["call_70"] == new["call_70"]).all()
    # the figure titles report the held-out rate, not the in-sample one
    assert fs.ctrl_held_out(out["thresholds"]["TA"]).startswith("controls held out at 50/60/70%: ")


def test_control_values_move_from_the_in_sample_two_percent(tmp_path):
    """In-sample, about FPR of the pooled control fibers pass by construction; held out, each animal is scored as
    an unseen one, so the controls' mean differs from the in-sample rate."""
    out, _ = summary_of(tmp_path, DESIGN)
    animals = out["animals"]
    ctrl = animals[animals["group"] == "ctrl"]
    in_sample = out["thresholds"]["TA"][0.5]["ctrl_fpr_pooled"]
    held_mean = out["thresholds"]["TA"][0.5]["loao_fpr_mean"]
    assert ctrl["pct_positive_50_pooled"].mean() == pytest.approx(100 * held_mean)
    assert in_sample == pytest.approx(fs.FPR, abs=0.003) and held_mean != pytest.approx(in_sample)


def test_one_control_animal_is_called_against_its_own_threshold_and_marked(tmp_path):
    out, _ = summary_of(tmp_path, [("TA", "ctrl", "0001", 1), ("TA", "ctrl", "0001", 2), ("TA", "treated", "0101", 1)])
    sections = out["sections"]
    threshold = out["thresholds"]["TA"][fs.THRESHOLD_COVERAGE]["sector_threshold"]
    assert set(sections["threshold"]) == {threshold}
    assert sections.set_index("group")["threshold_from"].to_dict() == {"ctrl": "this animal alone",
                                                                  "treated": "all controls"}
    assert fs.ctrl_held_out(out["thresholds"]["TA"]).startswith("controls passing at 50/60/70%")


def test_a_borrowed_threshold_is_named_as_such(tmp_path):
    out, _ = summary_of(tmp_path, [("TA", "treated", "0101", 1)], borrowed={"TA": {"threshold": 400.0, "source": "x"}})
    assert out["sections"]["threshold"].tolist() == [400.0]
    assert out["sections"]["threshold_from"].tolist() == ["borrowed threshold"]
    assert out["animals"]["threshold_from"].tolist() == ["borrowed threshold"]
