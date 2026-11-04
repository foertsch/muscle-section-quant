"""The synthetic demo runs the whole pipeline: tables, figures and both reports, with no scans and no model."""

import importlib.util
from pathlib import Path

import pandas as pd

DEMO = Path(__file__).resolve().parents[2] / "demo" / "run_demo.py"
spec = importlib.util.spec_from_file_location("run_demo", DEMO)
run_demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_demo)


def test_the_demo_produces_tables_figures_and_reports(tmp_path):
    out = tmp_path / "output"
    assert run_demo.run_demo(out, sections=1) == 0
    run = out / "run"
    for rel in ("tables/animals.csv", "tables/thresholds.json", "figures/pct_positive_call.png",
                "reports/threshold_decision.html", "reports/ta_ctrl_vs_treated.html"):
        assert (run / rel).is_file(), rel
    animals = pd.read_csv(run / "tables" / "animals.csv", dtype={"animal": str}).set_index("animal")
    assert set(animals.index) == {"1001", "1002", "1003", "1101", "1102"}
    assert (animals.loc[["1101", "1102"], "pct_positive_50"] > 20).all()   # treated rings are bright over part
    assert (animals.loc[["1001", "1002", "1003"], "pct_positive_50"] < 10).all()  # control rings are dim
    assert set(animals.loc[["1001", "1002", "1003"], "threshold_from"]) == {"other control animals"}
    assert "% of fibers positive" in run_demo.summary(run)
    assert not list(run.rglob("*.nd2"))
