import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from msq.dystrophin import plot_fiber_size as pf  # noqa: E402
from msq.core.synthetic.dystrophin import write_section  # noqa: E402


def sections(root: Path, rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    """Fake sections for (muscle, animal, group) rows; returns their meta table, indexed by sample."""
    meta = []
    for m, a, g in rows:
        s = f"{m}_sec1_{a}_20x_{g}"
        write_section(root, s)
        meta.append({"sample": s, "muscle": m, "animal": a, "group": g, "section": 1})
    return pd.DataFrame(meta).set_index("sample")


def panels(monkeypatch, root: Path, meta: pd.DataFrame) -> dict[tuple[str, str, str], str]:
    """Draw the figure and say, per (muscle, group, animal) panel, whether it shows data or 'no sections'."""
    figs = []
    monkeypatch.setattr(pf.plt, "close", figs.append)
    pf.make(root, meta, root / "figures", root / "tables")
    axs = figs[0].axes
    order = pf.animal_order(pf.load_sizes(root, meta))
    muscles = [m for m in pf.settings.MUSCLES if m in set(meta["muscle"])]
    out = {}
    for i, (g, a) in enumerate(order):
        for j, m in enumerate(muscles):
            ax = axs[i * len(muscles) + j]
            empty = any(t.get_text() == "no sections" for t in ax.texts)
            out[(m, g, a)] = "empty" if empty else ("data" if ax.patches else "blank")
    return out


def test_size_stats():
    v = pd.Series([10.0, 20.0, 30.0, 40.0, np.nan])
    st = pf.size_stats(v)
    assert st["n"] == 4 and st["median"] == 25.0 and st["mean"] == 25.0
    assert st["cv_x1000"] == pytest.approx(1000 * np.std([10, 20, 30, 40], ddof=1) / 25)
    assert st["pct_below_20um"] == 25.0


def test_load_sizes_keeps_included_fibers_and_labels_them(tmp_path):
    for s in ("TA_sec1_1106_20x_ctrl", "TA_sec1_1104_20x_treated"):
        (tmp_path / s).mkdir()
        pd.DataFrame({"include": [True, False, True], "min_feret_um": [30.0, 5.0, 40.0],
                      "area_um2": [900.0, 20.0, 1600.0], "other": 1}).to_csv(tmp_path / s / "fibers.csv", index=False)
    meta = pd.DataFrame({"muscle": "TA", "group": ["ctrl", "treated"], "animal": ["1106", "1104"]},
                        index=["TA_sec1_1106_20x_ctrl", "TA_sec1_1104_20x_treated"])
    sizes = pf.load_sizes(tmp_path, meta)
    assert len(sizes) == 4 and set(sizes["min_feret_um"]) == {30.0, 40.0}
    assert list(sizes.columns) == ["min_feret_um", "area_um2", "sample", "muscle", "group", "animal"]
    assert pf.animal_order(sizes) == [("ctrl", "1106"), ("treated", "1104")]
    t = pf.stats_table(sizes)
    assert t.set_index("animal").loc["1104", "median"] == 35.0


def test_muscles_with_different_animals(tmp_path, monkeypatch):
    meta = sections(tmp_path, [("QUAD", "1104", "treated"), ("QUAD", "1106", "ctrl"),
                               ("TA", "1105", "treated"), ("TA", "1106", "ctrl")])
    shown = panels(monkeypatch, tmp_path, meta)
    assert shown[("QUAD", "treated", "1104")] == "data" and shown[("TA", "treated", "1104")] == "empty"
    assert shown[("TA", "treated", "1105")] == "data" and shown[("QUAD", "treated", "1105")] == "empty"
    assert shown[("QUAD", "ctrl", "1106")] == shown[("TA", "ctrl", "1106")] == "data"
    assert len(pd.read_csv(tmp_path / "tables" / "fiber_size_by_animal.csv")) == 4


def test_an_animal_in_two_groups_gets_one_row_per_group(tmp_path, monkeypatch):
    # e.g. QUAD 1105 relabelled as a control, TA 1105 treated
    meta = sections(tmp_path, [("QUAD", "1105", "ctrl"), ("QUAD", "1106", "ctrl"),
                               ("TA", "1105", "treated"), ("TA", "1106", "ctrl")])
    shown = panels(monkeypatch, tmp_path, meta)
    assert shown[("QUAD", "ctrl", "1105")] == "data" and shown[("QUAD", "treated", "1105")] == "empty"
    assert shown[("TA", "treated", "1105")] == "data" and shown[("TA", "ctrl", "1105")] == "empty"


def test_the_figure_carries_no_notes_about_one_dataset(tmp_path, monkeypatch):
    """Facts about one dataset (notes about single animals) belong in that dataset's documentation, not in a figure
    every run draws: the footnote is the method note only and names no animal."""
    figs = []
    monkeypatch.setattr(pf.plt, "close", figs.append)
    pf.make(tmp_path, sections(tmp_path, [("TA", "1106", "ctrl"), ("TA", "1104", "treated")]), tmp_path / "f",
            tmp_path / "t")
    footnote = " ".join(t.get_text() for t in figs[0].texts)
    assert footnote.startswith("Included fibers, all sections of an animal pooled")
    assert set(re.findall(r"\b1\d{3}\b", footnote)) <= {"1000"}  # no animal number (1000 is the CV's scale)
