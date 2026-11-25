from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from msq.psr import fiber_size as fs


def write_section(run: Path, sample: str, feret: list[float], red: list[int], border=None, in_section=None):
    d = run / "sections" / "base" / sample
    d.mkdir(parents=True)
    n = len(feret)
    o = pd.DataFrame({"label": np.arange(1, n + 1), "min_feret_um": feret, "max_feret_um": [1.5 * f for f in feret],
                      "area_um2": [np.pi * (f / 2) ** 2 for f in feret],
                      "in_section": in_section or [True] * n, "touches_border": border or [False] * n,
                      "core_bg": 0, "core_red": red, "core_yellow": [100 - r for r in red], "core_other": 0,
                      "band_bg": 0, "band_red": 0, "band_yellow": 0, "band_other": 0})
    o.to_csv(d / "objects.csv", index=False)


def test_fibers_apply_the_provisional_rule(tmp_path):
    pd.DataFrame({"sample": ["A_control", "B_group_a"], "group": ["control", "group_a"], "animal": ["1", "2"], "muscle": "QUAD",
                  "section": 1, "path": ""}).to_csv(tmp_path / "samples.csv", index=False)
    write_section(tmp_path, "A_control", [40, 45, 50, 2], [0, 10, 80, 0])          # 50% red -> out; 2 um -> too small
    write_section(tmp_path, "B_group_a", [20, 30, 35], [0, 0, 0], border=[False, True, False])
    df = fs.fibers(tmp_path)
    assert sorted(df.loc[df["sample"] == "A_control", "min_feret_um"]) == [40, 45]
    assert sorted(df.loc[df["sample"] == "B_group_a", "min_feret_um"]) == [20, 35]
    assert set(df["group"]) == {"control", "group_a"}


def test_describe_per_group(example):
    rng = np.random.default_rng(0)
    df = pd.concat([pd.DataFrame({"min_feret_um": rng.normal(m, 8, 300), "group": g, "animal": g, "sample": g})
                    for g, m in [("control", 47), ("group_a", 33), ("group_b", 35)]], ignore_index=True)
    df["area_um2"] = np.pi * (df["min_feret_um"] / 2) ** 2
    df["max_feret_um"] = 1.6 * df["min_feret_um"]
    groups = fs.describe(df, ["group"], example)
    assert list(groups["group"]) == ["control", "group_a", "group_b"]
    assert abs(groups.loc[0, "min_feret_um_median"] - 47) < 2
    assert 100 < groups.loc[0, "min_feret_variance_coefficient"] < 250


def test_describe_sorts_by_the_cohorts_group_order_whatever_the_order_of_the_rows(example):
    rng = np.random.default_rng(1)
    df = pd.concat([pd.DataFrame({"min_feret_um": rng.normal(40, 5, 20), "group": g, "animal": a, "sample": f"{a}_s"})
                    for g, a in [("group_b", "3"), ("control", "2"), ("group_a", "1"), ("control", "1")]], ignore_index=True)
    df["area_um2"] = np.pi * (df["min_feret_um"] / 2) ** 2
    df["max_feret_um"] = 1.6 * df["min_feret_um"]
    assert list(fs.describe(df, ["group"], example)["group"]) == ["control", "group_a", "group_b"]
    by_animal = fs.describe(df, ["group", "animal", "sample"], example)
    assert by_animal[["group", "animal"]].values.tolist() == [["control", "1"], ["control", "2"], ["group_a", "1"],
                                                               ["group_b", "3"]]


def test_max_feret_is_filled_in_from_the_outlines(tmp_path):
    import json

    import tifffile
    d = tmp_path / "s"
    d.mkdir()
    labels = np.zeros((20, 30), np.uint32)
    labels[2:6, 3:13] = 1  # 4 x 10 grid pixels
    tifffile.imwrite(d / "labels_ds6.tif", labels)
    (d / "section.json").write_text(json.dumps({"params": {"downsample": 6}, "grid_um": 2.0}))
    got = fs.add_max_feret(pd.DataFrame({"label": [1]}), d)
    assert got.loc[0, "max_feret_um"] == pytest.approx(2.0 * np.hypot(4, 10))
