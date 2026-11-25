"""Layer 1 for psr: msq.psr reproduces tests/golden/psr/expected/ from tests/golden/psr/inputs.npz.

The run is made exactly as golden.json's "how" says the golden was made by psr-quant v1.0.0: section.run_frame per
sample with the stored labels instead of Cellpose, a sample table written by hand with the generic groups, and the summary
tables with those groups as the cohort.
"""

import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from msq.core.segment import FixedLabels
from msq.psr import section, summary
from msq.psr.settings import Cohort


def golden_cohort(groups: dict) -> Cohort:
    """The generic groups of golden.json as a cohort (the golden samples are named by hand, so no folder rules)."""
    return Cohort(name="golden", folders={}, order=tuple(groups["order"]), control=groups["control"],
                  labels=dict(groups["labels"]), colours=dict(groups["colours"]), animal=r"(\w+)")


def make_run(golden: Path, work: Path, change_params=None) -> Path:
    """Run the golden design into work/run and return it; change_params edits the section parameters (a test of the
    test)."""
    meta = json.loads((golden / "golden.json").read_text(encoding="utf-8"))
    with np.load(golden / "inputs.npz") as z:
        arrays = {k: z[k] for k in z.files}
    px_um = float(arrays["px_um"])
    run = work / "run"
    params = section.params_for("base")
    if change_params:
        params = change_params(params)
    for d in meta["design"]:
        name = d["sample"]
        section.run_frame(arrays[f"{name}/frame_bgr"], px_um, run / "sections" / "base" / name, params, name,
                          source=f"synthetic:{name}", segment=FixedLabels(arrays[f"{name}/labels_ds6"]), variant="base")
    pd.DataFrame([{"sample": d["sample"], "group": d["group"], "muscle": d["muscle"], "animal": d["animal"],
                   "section": d["section"], "path": f"synthetic:{d['sample']}"} for d in meta["design"]]
                 ).to_csv(run / "samples.csv", index=False)
    summary.write_tables(run, run / "tables", golden_cohort(meta["groups"]))
    return run


def test_psr_reproduces_its_golden(golden_dirs, check_golden, tmp_path, capsys):
    if "psr" not in golden_dirs:
        pytest.skip("no psr golden")
    golden = golden_dirs["psr"]
    check_golden(golden, make_run(golden, tmp_path))


def test_the_builders_still_make_the_golden_inputs(golden_dirs):
    """The golden's inputs are what the synthetic builders in msq.core make from the design's seeds."""
    from msq.core.synthetic import psr as builders

    if "psr" not in golden_dirs:
        pytest.skip("no psr golden")
    golden = golden_dirs["psr"]
    meta = json.loads((golden / "golden.json").read_text(encoding="utf-8"))
    with np.load(golden / "inputs.npz") as z:
        d = meta["design"][0]
        s = builders.synthetic_psr_section(zlib.crc32(d["sample"].encode()), d["fibrosis"])
        assert np.array_equal(s["frame"], z[f"{d['sample']}/frame_bgr"])


def test_a_changed_number_fails_the_golden(golden_dirs, tmp_path, capsys):
    """The comparison is not vacuous: one parameter moved by a little changes the tables and the golden says so."""
    from dataclasses import replace

    from tests.golden.conftest import compare_golden

    if "psr" not in golden_dirs:
        pytest.skip("no psr golden")
    golden = golden_dirs["psr"]
    run = make_run(golden, tmp_path, lambda p: replace(p, hue_yellow_to=35))  # the fibers (hue about 41) stop being yellow
    diffs = compare_golden(golden, run)
    assert any(d.startswith("tables/animals.csv") for d in diffs)
    assert any(d.startswith("tables/sections.csv") for d in diffs)
