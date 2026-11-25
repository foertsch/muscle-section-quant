import json
import shutil
import sys

import numpy as np
import pandas as pd
import pytest

from msq.psr import summary

ROWS = [("A1_s1", "control", "a1", 1, 0.06), ("A1_s2", "control", "a1", 2, 0.08), ("A2_s1", "control", "a2", 1, 0.07),
        ("B1_s1", "group_a", "b1", 1, 0.20), ("B1_s2", "group_a", "b1", 2, 0.30)]


@pytest.fixture
def run(tmp_path):
    pd.DataFrame([{"sample": s, "group": g, "animal": a, "section": n, "muscle": "QUAD", "path": ""}
                  for s, g, a, n, _ in ROWS]).to_csv(tmp_path / "samples.csv", index=False)
    rng = np.random.default_rng(0)
    for s, g, a, n, coll in ROWS:
        d = tmp_path / "sections" / "base" / s
        d.mkdir(parents=True)
        (d / "section.json").write_text(json.dumps({"collagen_outside_fraction": coll, "red_fraction": coll + 0.02,
                                                    "section_area_mm2": 10.0}))
        k = 40
        feret = rng.normal(45 if g == "control" else 30, 5, k)
        pd.DataFrame({"label": np.arange(1, k + 1), "min_feret_um": feret, "max_feret_um": 1.5 * feret,
                      "area_um2": np.pi * (feret / 2) ** 2, "in_section": True, "touches_border": False,
                      "core_bg": 0, "core_red": 0, "core_yellow": 100, "core_other": 0, "band_bg": 0, "band_red": 0,
                      "band_yellow": 0, "band_other": 0}).to_csv(d / "objects.csv", index=False)
    return tmp_path


def test_animals_average_their_sections_and_genotypes_their_animals(run, tmp_path, example):
    sections = summary.section_table(run, example)
    assert len(sections) == 5 and sections["n_fibers"].min() > 0
    animals = summary.animal_table(sections, summary.fiber_size.fibers(run))
    a1 = animals.set_index("animal").loc["a1"]
    assert a1["n_sections"] == 2 and a1["collagen_outside_fraction"] == pytest.approx(0.07)
    g = summary.genotype_table(animals, example).set_index("group")
    assert list(g.index) == ["control", "group_a"] and g.loc["control", "n_animals"] == 2
    assert g.loc["control", "collagen_outside_fraction_mean"] == pytest.approx(0.07)   # (0.07 + 0.07) / 2 animals
    assert g.loc["group_a", "collagen_outside_fraction_mean"] == pytest.approx(0.25)
    summary.plot_collagen(sections, animals, example, tmp_path / "c.png")
    summary.plot_vs_size(animals, example, tmp_path / "s.png")
    assert (tmp_path / "c.png").exists() and (tmp_path / "s.png").exists()


def test_a_group_the_cohort_does_not_know_is_left_out_of_the_genotype_table(run, example):
    sections = summary.section_table(run, example)
    animals = summary.animal_table(sections.assign(group=sections["group"].replace("group_a", "other")),
                                   summary.fiber_size.fibers(run))
    assert set(animals["group"]) == {"control", "other"}
    assert list(summary.genotype_table(animals, example)["group"]) == ["control"]


def test_a_partial_section_folder_left_by_a_killed_copy_is_ignored(run, example):
    # a section folder is built as <sample>.partial and renamed when complete; a killed run leaves the .partial
    shutil.copytree(run / "sections" / "base" / "A1_s1", run / "sections" / "base" / "A1_s1.partial")
    assert len(summary.section_table(run, example)) == 5
    assert set(summary.fiber_size.fibers(run)["sample"]) == {s for s, *_ in ROWS}


def test_main_exports_tables_for_r(run, tmp_path, monkeypatch, example_cohorts):
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["summary", str(run), "--cohort", "example", "--out", str(out)])
    summary.main()
    fibers = pd.read_csv(out / "fibers.csv.gz")
    palette = pd.read_csv(out / "palette.csv")
    assert {"group", "animal", "min_feret_um", "max_feret_um", "area_um2"} <= set(fibers.columns) and len(fibers) == 200
    assert list(palette["group"]) == ["control", "group_a"] and palette.loc[0, "control"]   # only groups with data
