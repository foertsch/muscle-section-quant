import numpy as np
import pandas as pd
import pytest

from msq.core.calls import animal_means, group_stats, rate_at_or_above, threshold_from_controls

VALUES = pd.Series([3.0, np.nan, 1.0, 7.5, 2.25, np.nan, 9.0, 4.0])


@pytest.mark.parametrize("pct", [50.0, 90.0, 98.0, 100 * (1 - 0.02), 99.5])
def test_the_threshold_equals_both_expressions_it_replaces(pct):
    dystrophin = float(np.nanpercentile(np.asarray(VALUES, dtype=float), pct))
    necrosis = float(np.percentile(VALUES.dropna(), pct))
    assert threshold_from_controls(VALUES, pct) == dystrophin == necrosis
    assert threshold_from_controls(VALUES.to_numpy(), pct) == dystrophin
    assert threshold_from_controls(list(VALUES), pct) == dystrophin


def test_no_control_value_raises():
    for values in (pd.Series([], dtype=float), pd.Series([np.nan, np.nan])):
        with pytest.raises(ValueError, match="no control"):
            threshold_from_controls(values, 98.0)


def test_rate_counts_nan_as_below_and_is_nan_without_values():
    assert rate_at_or_above(VALUES, 4.0) == float((VALUES >= 4.0).mean()) == 3 / 8
    assert rate_at_or_above(VALUES.to_numpy(), 4.0) == 3 / 8
    assert np.isnan(rate_at_or_above(pd.Series([], dtype=float), 1.0))


SECTIONS = pd.DataFrame({
    "muscle": ["TA", "TA", "TA", "QUAD", "QUAD", "TA"],
    "group": ["treated", "ctrl", "ctrl", "ctrl", "treated", "treated"],
    "animal": ["1002", "1001", "1001", "1001", "1002", "1003"],
    "sample": ["s1", "s2", "s3", "s4", "s5", "s6"],
    "n_fibers": [10, 20, 30, 40, 50, 60],
    "share": [0.1, 0.2, 0.4, 0.5, 0.6, 0.3],
})
KEYS = ["muscle", "group", "animal"]


def test_animal_means_in_sorted_key_order():
    t = animal_means(SECTIONS, KEYS, ["share"], sums=["n_fibers"])
    old = SECTIONS.groupby(KEYS).agg(n_sections=("sample", "size"), n_fibers=("n_fibers", "sum"),
                                     share=("share", "mean")).reset_index()
    pd.testing.assert_frame_equal(t, old)
    assert list(t.columns) == [*KEYS, "n_sections", "n_fibers", "share"]


def test_animal_means_with_an_order_sorts_stably_by_it():
    rank = {"ctrl": 0, "treated": 1}
    order = lambda s: s.map(rank) if s.name == "group" else s  # noqa: E731
    t = animal_means(SECTIONS, KEYS, ["share"], sums=["n_fibers"], order=order)
    old = (SECTIONS.groupby(KEYS, sort=False)
           .agg(n_sections=("sample", "size"), n_fibers=("n_fibers", "sum"), share=("share", "mean")).reset_index()
           .sort_values(KEYS, key=order, kind="stable").reset_index(drop=True))
    pd.testing.assert_frame_equal(t, old)


def test_group_stats_as_necrosis_writes_them():
    animals = animal_means(SECTIONS, KEYS, ["share"], sums=["n_fibers"])
    t = group_stats(animals, ["muscle", "group"], ["share"], sums=["n_fibers"])
    g = animals.groupby(["muscle", "group"], sort=False)
    old = g.agg(n_animals=("animal", "size"), n_fibers=("n_fibers", "sum"))
    old["share_mean"] = g["share"].mean()
    old["share_sd"] = g["share"].std()
    pd.testing.assert_frame_equal(t, old.reset_index())


def test_group_stats_as_psr_writes_them():
    animals = animal_means(SECTIONS, ["group", "animal"], ["share", "n_fibers"])
    t = group_stats(animals, ["group"], ["share", "n_fibers"], sd="std", sort=True)
    old = animals.groupby("group")[["share", "n_fibers"]].agg(["mean", "std"])
    old.columns = [f"{c}_{s}" for c, s in old.columns]
    old.insert(0, "n_animals", animals.groupby("group").size())
    pd.testing.assert_frame_equal(t, old.reset_index(), check_names=False)
