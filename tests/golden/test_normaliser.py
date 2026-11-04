"""The normaliser itself: what it drops, what it parses and what it still catches."""

import gzip
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.golden import conftest
from tests.golden.conftest import (OTHER_PLATFORM_RTOL, PIPELINES, GOLDEN_ROOT, compare_golden, current_file,
                                   file_diffs, golden_meta, json_diffs, normalise_json, npz_diffs, read_table,
                                   renamed_json, renamed_path, renamed_table, table_diffs)


def test_volatile_keys_and_paths_are_dropped():
    a = {"threshold": 1.5, "started": "2026-10-27T07:53:49", "source": "/Users/someone/scans/x.nd2", "runtime_s": 3.2,
         "software": {"numpy": "2.5.4"}, "nested": {"code_version": "v1.0.0-dirty", "keep": [1, 2]},
         "note": "C:\\Users\\lab\\run", "name": "TA_sec1_0001"}
    b = {"threshold": 1.5, "started": "2027-01-01T00:00:00", "source": "/home/other/scans/y.nd2", "runtime_s": 9.9,
         "software": {"numpy": "3.0.0"}, "nested": {"code_version": "v2.0.0", "keep": [1, 2]},
         "note": "/Volumes/share/run", "name": "TA_sec1_0001"}
    assert normalise_json(a) == normalise_json(b)
    assert normalise_json(a)["note"] == "<path>" and "started" not in normalise_json(a)
    assert json_diffs(normalise_json(a), normalise_json(b)) == []


def test_a_changed_number_is_reported_with_its_key_path():
    a = {"muscles": {"TA": {"threshold": 260.0, "50": {"ctrl_fpr_pooled": 0.02}}}, "levels": ["50", "60"]}
    b = {"muscles": {"TA": {"threshold": 261.5, "50": {"ctrl_fpr_pooled": 0.02}}}, "levels": ["50", "60"]}
    assert json_diffs(a, b) == ["$.muscles.TA.threshold: differs by 1.5"]
    assert json_diffs(a, {**a, "levels": ["50"]}) == ["$.levels: 2 items expected, 1 found"]
    assert json_diffs(a, {"muscles": a["muscles"]}) == ["$.levels: missing"]
    assert json_diffs({"x": float("nan")}, {"x": float("nan")}) == []
    assert json_diffs({"x": None}, {"x": 0}) == ["$.x: NoneType expected, int found"]
    assert json_diffs({"x": 2}, {"x": 2.0}) == []  # the same number, written as int or float


def test_tables_compare_parsed_values_not_text(tmp_path):
    (tmp_path / "a.csv").write_text("sample,label,value,flag,note\nTA_1,1,0.5,True,\nTA_1,2,1e3,False,x\n")
    (tmp_path / "b.csv").write_text("note,sample,label,value,flag\n,TA_1,1,5e-1,True\nx,TA_1,2,1000.0,False\n")
    assert file_diffs(tmp_path / "a.csv", tmp_path / "b.csv") == []
    with gzip.open(tmp_path / "b.csv.gz", "wt") as f:
        f.write((tmp_path / "b.csv").read_text())
    assert file_diffs(tmp_path / "a.csv", tmp_path / "b.csv.gz") == []


def test_a_changed_cell_is_reported_with_column_count_and_largest_difference(tmp_path):
    e = pd.DataFrame({"label": [1, 2, 3], "v": [0.5, 1.0, float("nan")], "s": ["a", "b", None]})
    a = e.copy()
    a.loc[1, "v"] = 1.25
    a.loc[2, "s"] = "c"
    assert table_diffs(e, a) == ["s: 1 of 3 cells differ", "v: 1 of 3 cells differ, largest difference 0.25"]
    assert table_diffs(e, e.copy()) == []
    assert table_diffs(e, e.iloc[:2]) == ["3 rows expected, 2 found"]
    assert table_diffs(e, e.drop(columns="s")) == ["columns missing: s"]
    assert table_diffs(e, e.assign(extra=1)) == ["columns unexpected: extra"]


def test_volatile_columns_are_dropped(tmp_path):
    (tmp_path / "a.csv").write_text("sample,path\nTA_1,/Users/x/a.nd2\n")
    (tmp_path / "b.csv").write_text("sample,path\nTA_1,D:\\scans\\a.nd2\n")
    assert file_diffs(tmp_path / "a.csv", tmp_path / "b.csv") == []
    assert list(read_table(tmp_path / "a.csv").columns) == ["sample"]


def test_npz_arrays_compare_element_by_element(tmp_path):
    e = {"counts": np.arange(6).reshape(2, 3), "edges": np.array([0.0, float("nan"), 2.0])}
    a = {"counts": np.arange(6).reshape(2, 3), "edges": np.array([0.0, float("nan"), 2.0])}
    assert npz_diffs(e, a) == []
    a["counts"] = a["counts"].copy()
    a["counts"][1, 2] = 9
    assert npz_diffs(e, a) == ["counts: 1 of 6 elements differ, largest difference 4"]
    assert npz_diffs(e, {"counts": e["counts"]}) == ["edges: missing"]
    np.savez_compressed(tmp_path / "x.npz", **e)
    np.savez(tmp_path / "y.npz", **{k: v.copy() for k, v in e.items()})
    assert file_diffs(tmp_path / "x.npz", tmp_path / "y.npz") == []  # compressed or not


def test_compare_golden_reports_missing_and_changed_files(tmp_path):
    golden = tmp_path / "golden"
    (golden / "expected" / "tables").mkdir(parents=True)
    (golden / "expected" / "tables" / "animals.csv").write_text("animal,pct\n0001,2.0\n")
    (golden / "expected" / "run.json").write_text(json.dumps({"threshold": 1.0, "finished": "now"}))
    (golden / "golden.json").write_text(json.dumps({"compare": ["tables/animals.csv", "run.json"],
                                                    "volatile_keys": ["custom"]}))
    actual = tmp_path / "actual"
    (actual / "tables").mkdir(parents=True)
    (actual / "tables" / "animals.csv").write_text("animal,pct\n0001,2.5\n")
    assert compare_golden(golden, actual) == ["tables/animals.csv: pct: 1 of 1 cells differ, largest difference 0.5",
                                              "run.json: missing"]
    shutil.copy(golden / "expected" / "tables" / "animals.csv", actual / "tables" / "animals.csv")
    (actual / "run.json").write_text(json.dumps({"threshold": 1.0, "finished": "later", "custom": "dropped too"}))
    assert compare_golden(golden, actual) == []


@pytest.mark.parametrize("pipeline", PIPELINES)
def test_each_golden_folder_is_complete_and_reproduces_itself(pipeline):
    """Every compared file exists, inputs.npz stays under 2 MB, and the golden compared with itself is clean."""
    golden = Path(__file__).parent / pipeline
    if not (golden / "golden.json").exists():
        pytest.skip(f"no golden for {pipeline} yet")
    meta = golden_meta(golden)
    assert meta["pipeline"] == pipeline and meta["compare"]
    assert (golden / "inputs.npz").stat().st_size < 2_000_000
    for rel in meta["compare"]:
        assert (golden / "expected" / rel).is_file(), rel
    assert compare_golden(golden, golden / "expected", actual_renamed=True) == []
    assert not (golden / "run_output.log").exists()  # the log carries paths and times; it is not kept


def test_the_rename_map_reads_an_old_expected_file_as_the_new_one(tmp_path):
    """Written before "bar" and "cutoff" became "threshold": columns, keys, file names and listed wording are renamed,
    every value stays."""
    old = pd.DataFrame({"sample": ["a", "b"], "bar": [1.5, 2.5], "bar_from": ["all controls", "borrowed bar"]})
    assert list(renamed_table(old).columns) == ["sample", "threshold", "threshold_from"]
    assert renamed_table(old)["threshold_from"].tolist() == ["all controls", "borrowed threshold"]
    assert renamed_table(old)["threshold"].tolist() == [1.5, 2.5]
    obj = {"bars": {"TA": {"bar": 1.5, "bar_source": "x"}}, "reports": ["reports/cutoff_decision.html"]}
    assert renamed_json(obj) == {"thresholds": {"TA": {"threshold": 1.5, "threshold_source": "x"}},
                                 "reports": ["reports/threshold_decision.html"]}
    assert renamed_path("calls/laminin/bar.json") == "calls/laminin/threshold.json"
    assert renamed_path("bars.csv") == "thresholds.csv" and renamed_path("calls/laminin/necrosis.csv") == \
        "calls/laminin/necrosis.csv"
    (tmp_path / "bar.json").write_text(json.dumps({"QUAD": {"bar": 1.0}}))
    (tmp_path / "new").mkdir()
    (tmp_path / "new" / "threshold.json").write_text(json.dumps({"QUAD": {"threshold": 1.0}}))
    assert file_diffs(tmp_path / "bar.json", tmp_path / "new" / "threshold.json") == []
    assert current_file(tmp_path, "threshold.json") == tmp_path / "bar.json"  # a folder written before the rename
    assert current_file(tmp_path / "new", "threshold.json") == tmp_path / "new" / "threshold.json"


def test_the_rename_map_renames_and_nothing_else(tmp_path):
    (tmp_path / "bar.json").write_text(json.dumps({"QUAD": {"bar": 1.0}}))
    (tmp_path / "new").mkdir()
    (tmp_path / "new" / "threshold.json").write_text(json.dumps({"QUAD": {"threshold": 1.5}}))
    assert file_diffs(tmp_path / "bar.json", tmp_path / "new" / "threshold.json") == ["$.QUAD.threshold: differs by 0.5"]
    (tmp_path / "new" / "threshold.json").write_text(json.dumps({"QUAD": {"bar": 1.0}}))  # the old name is not the new
    assert file_diffs(tmp_path / "bar.json", tmp_path / "new" / "threshold.json") == [
        "$.QUAD.bar: unexpected", "$.QUAD.threshold: missing"]
    with pytest.raises(AssertionError, match="merged"):
        renamed_json({"bar": 1, "threshold": 2})
    with pytest.raises(AssertionError, match="merged"):
        renamed_table(pd.DataFrame({"bar": [1], "threshold": [2]}))


def test_float_rounding_is_tolerated_only_on_another_platform():
    e = pd.DataFrame({"v": [45.123456789012345, 0.0, 1200.5]})
    rounded = e.assign(v=[45.123456789012350, 1e-15, 1200.5000000000002])  # the last bits, as on Linux x86_64
    moved = e.assign(v=[45.1234568, 0.0, 1200.5])  # a real change, in the 8th digit
    assert len(table_diffs(e, rounded)) == 1 and table_diffs(e, rounded, OTHER_PLATFORM_RTOL) == []
    assert table_diffs(e, moved, OTHER_PLATFORM_RTOL) == table_diffs(e, moved)
    assert json_diffs({"t": 913.0000000000001}, {"t": 913.0}, rtol=OTHER_PLATFORM_RTOL) == []
    assert json_diffs({"t": 913.0001}, {"t": 913.0}, rtol=OTHER_PLATFORM_RTOL) != []
    assert npz_diffs({"h": np.array([1.0, 2.0])}, {"h": np.array([1.0, 2.0 + 1e-15])}, OTHER_PLATFORM_RTOL) == []


def test_the_golden_is_exact_where_it_was_made_and_tolerant_elsewhere(tmp_path, monkeypatch):
    actual = tmp_path / "run"
    shutil.copytree(GOLDEN_ROOT / "dystrophin" / "expected", actual)
    meta = golden_meta(GOLDEN_ROOT / "dystrophin")
    table = next(rel for rel in meta["compare"] if rel.endswith("fibers.csv"))
    df = pd.read_csv(actual / table)
    column = next(c for c in df.columns if df[c].dtype == float and df[c].abs().max() > 1)
    df[column] = df[column] * (1 + 1e-15)
    df.to_csv(actual / table, index=False)
    monkeypatch.setattr(conftest.runinfo, "platform_key", lambda: meta["produced"]["platform"])
    assert any(d.startswith(table) for d in compare_golden(GOLDEN_ROOT / "dystrophin", actual, actual_renamed=True))
    monkeypatch.setattr(conftest.runinfo, "platform_key", lambda: "linux-x86_64")
    assert compare_golden(GOLDEN_ROOT / "dystrophin", actual, actual_renamed=True) == []
