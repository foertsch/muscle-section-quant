"""The psr runner (run_pipeline.py) on the scans of test_checks.py, with the section step and the summary tables
replaced by stand-ins that record what they are asked for. A whole run on synthetic sections is the adapter test in
tests/regression."""

import json
from pathlib import Path

import pandas as pd
import pytest

from msq.core import runner
from msq.core.problems import ExitStatus
from msq.core.synthetic.psr import header
from msq.psr import checks, section, summary
from msq.psr.run_pipeline import Psr
from tests.psr.conftest import example_cohorts  # noqa: F401  (a fixture: the generic cohorts)
from tests.psr.test_checks import scans  # noqa: F401  (a fixture: the scan folder)

SAMPLES = ["QUAD_sec1_1001-101_control", "QUAD_sec2_1001-101_control", "QUAD_sec1_1001-102_control",
           "QUAD_sec1_1002-101_group_a", "QUAD_sec3_1002-102_group_a"]


@pytest.fixture
def site(tmp_path, scans, monkeypatch, example_cohorts):  # noqa: F811  (the imported fixtures)
    seen = {"sections": [], "tables": [], "fail": set(), "scans": scans, "run": tmp_path / "run"}

    def fake_run(path, out_dir, sample, variant="base", source=None, segment=None):
        if sample in seen["fail"]:
            raise RuntimeError("out of memory")
        seen["sections"].append((sample, variant, segment))
        out_dir.mkdir(parents=True, exist_ok=True)  # as section.run, which replaces a folder that is there
        (out_dir / "section.json").write_text("{}")

    def fake_tables(run, out, cohort, variant="base", holes=None):
        seen["tables"].append((out, cohort.name, variant))
        return pd.DataFrame({"animal": ["1001-101"], "x": [1.23456]}), pd.DataFrame({"genotype": ["g"], "y": [2.0]})

    monkeypatch.setattr(checks, "nd2_header", lambda p: header())
    monkeypatch.setattr(section, "run", fake_run)
    monkeypatch.setattr(summary, "write_tables", fake_tables)
    monkeypatch.setattr(runner, "gpu_device", lambda: "cpu")
    monkeypatch.setattr(runner, "weights_file", lambda: Path("no weights here"))
    return seen


def start(seen, **options):
    return runner.run(Psr(), seen["scans"], seen["run"], cohort="example", **options)


def test_check_only_writes_the_table_with_the_imaging_settings_and_the_checks(site, capsys):
    with pytest.raises(SystemExit) as e:
        runner.main(Psr(), [str(site["scans"]), str(site["run"]), "--cohort", "example", "--check-only"])
    assert e.value.code == ExitStatus.DONE and site["sections"] == []
    df = pd.read_csv(site["run"] / "samples.csv")
    assert list(df.columns[: len(checks.COLUMNS)]) == checks.COLUMNS and "exposure_ms" in df.columns
    assert "single fields" in (site["run"] / "checks.txt").read_text()
    assert "QUAD_sec1_1001-101_control" in capsys.readouterr().out
    assert json.loads((site["run"] / "progress.json").read_text())["state"] == "checked"


def test_a_run_measures_every_section_and_writes_the_tables(site):
    assert start(site) == ExitStatus.DONE
    assert [s[0] for s in site["sections"]] == SAMPLES and {s[1] for s in site["sections"]} == {"base"}
    assert site["tables"] == [(site["run"] / "tables", "example", "base")]
    info = json.loads((site["run"] / "run_settings.json").read_text())
    assert info["cohort"] == "example" and info["variants"] == ["base"] and info["failed_sections"] == []


def test_a_run_that_stopped_goes_on_where_it_stopped_and_redo_starts_over(site):
    site["fail"] = {SAMPLES[2]}
    assert start(site) == ExitStatus.INCOMPLETE
    site["fail"], site["sections"] = set(), []
    assert start(site) == ExitStatus.DONE and [s[0] for s in site["sections"]] == [SAMPLES[2]]
    site["sections"].clear()
    start(site, redo=True)
    assert [s[0] for s in site["sections"]] == SAMPLES


def test_a_section_is_finished_when_every_variant_is_there_and_the_cached_labels_reach_it(site):
    assert start(site, variants=["base", "ds4"], segment=lambda sample: f"labels of {sample}") == ExitStatus.DONE
    assert [s[1] for s in site["sections"]][:4] == ["base", "ds4", "base", "ds4"]
    assert site["sections"][0][2] == f"labels of {SAMPLES[0]}"
    assert site["tables"][0][2] == "base"  # the tables come from the first variant
    (site["run"] / "sections" / "ds4" / SAMPLES[0] / "section.json").unlink()
    site["sections"].clear()
    start(site, variants=["base", "ds4"])
    assert [(s[0], s[1]) for s in site["sections"]] == [(SAMPLES[0], "base"), (SAMPLES[0], "ds4")]  # both again


def test_when_no_section_can_be_measured_there_are_no_tables(site):
    site["fail"] = set(SAMPLES)
    assert start(site) == ExitStatus.INCOMPLETE and site["tables"] == []
    assert json.loads((site["run"] / "progress.json").read_text())["state"] == "failed"
