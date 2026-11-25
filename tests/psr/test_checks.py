from pathlib import Path

import pandas as pd
import pytest

from msq.core.synthetic.psr import DESCRIPTION, HEADER, IMAGING
from msq.psr import checks


def touch(root: Path, rel: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")
    return p


@pytest.fixture
def scans(tmp_path):
    root = tmp_path / "scans"
    for rel in [
        "1001 control/1001-101-QUAD/1001-101-20full.nd2",
        "1001 control/1001-101-QUAD/2-1001-101-20full.nd2",
        "1001 control/1001-101-QUAD/1001-101-10.1.nd2",       # single field: left out
        "1001 control/1001-102-quad/1001-102-20x-full.nd2",
        "1002 group a/1002-101-quad/1002-101-20x-full.nd2",
        "1002 group a/1002-101-quad/1002-101-40.2x.nd2",     # single field: left out
        "1002 group a/1002-102-QUAD/3-1002-102-20full.nd2",
        "1002 group a/notes/overview.nd2",                   # cannot be read
    ]:
        touch(root, rel)
    return root


def by_where(problems):
    return {p.where: p for p in problems}


def test_sample_table_from_folder(scans, example):
    df, problems = checks.samples_from_folder(scans, example)
    assert list(df.columns) == checks.COLUMNS
    assert list(df["sample"]) == ["QUAD_sec1_1001-101_control", "QUAD_sec2_1001-101_control", "QUAD_sec1_1001-102_control",
                                  "QUAD_sec1_1002-101_group_a", "QUAD_sec3_1002-102_group_a"]  # control group first
    assert all(Path(p).is_absolute() for p in df["path"])
    found = by_where(problems)
    assert found[str(Path("1002 group a/notes/overview.nd2"))].level == "warning"
    assert found[""].level == "note" and "2 single fields" in found[""].message
    assert not checks.errors(problems)


def test_folder_animal_mismatch_is_a_warning(tmp_path, example):
    touch(tmp_path, "1002 group a/1002-101-quad/1002-103-20x-full.nd2")
    df, problems = checks.samples_from_folder(tmp_path, example)
    assert df["animal"].tolist() == ["1002-103"]
    assert [p.level for p in problems] == ["warning"] and "1002-101" in problems[0].message


def test_empty_folder_is_an_error(tmp_path, example):
    df, problems = checks.samples_from_folder(tmp_path, example)
    assert df.empty and [p.level for p in problems] == ["error"]


def test_imaging_settings_from_description():
    got = checks.imaging_settings(DESCRIPTION, "01.01.2026  12:00:00")
    assert got == IMAGING


def test_imaging_settings_missing_fields_are_left_out():
    assert checks.imaging_settings("Exposure: 1.7 ms\n", "") == {"exposure_ms": 1.7}


def test_check_files_passes_and_adds_imaging(scans, example):
    df, _ = checks.samples_from_folder(scans, example)
    problems, imaging = checks.check_files(df, read_header=lambda p: HEADER)
    assert problems == []
    assert list(imaging.index) == list(df["sample"])
    assert imaging.loc["QUAD_sec1_1001-101_control", "exposure_ms"] == 11.0


def test_check_files_errors(scans, example):
    df, _ = checks.samples_from_folder(scans, example)
    bad = {
        "QUAD_sec1_1001-101_control": {**HEADER, "channels": ["DAPI", "Brightfield"]},
        "QUAD_sec2_1001-101_control": {**HEADER, "px_um": 0.8727},           # a 10x field named full
        "QUAD_sec1_1001-102_control": {**HEADER, "sizes": {"Y": 100, "X": 120}},  # not RGB
        "QUAD_sec1_1002-101_group_a": {**HEADER, "bits": 12},
    }
    by_path = dict(zip(df["path"], df["sample"]))

    def read(p):
        s = by_path[str(p)]
        if s == "QUAD_sec3_1002-102_group_a":
            raise OSError("truncated")
        return bad[s]

    problems, imaging = checks.check_files(df, read_header=read)
    found = {p.where: p.message for p in problems}
    assert "Brightfield" in found["QUAD_sec1_1001-101_control"]
    assert "pixel size" in found["QUAD_sec2_1001-101_control"]
    assert "colour" in found["QUAD_sec1_1001-102_control"]
    assert "8-bit" in found["QUAD_sec1_1002-101_group_a"]
    assert "cannot be opened" in found["QUAD_sec3_1002-102_group_a"]
    assert all(p.level == "error" for p in problems)
    assert "QUAD_sec3_1002-102_group_a" not in imaging.index


def test_imaging_problems_report_differences_and_sessions(example):
    samples = pd.DataFrame({"sample": ["a", "b", "c"], "group": ["control", "control", "group_a"]})
    imaging = pd.DataFrame([IMAGING, {**IMAGING, "exposure_ms": 30.0, "acquired": "2026-01-08 13:00"},
                            {**IMAGING, "acquired": "2026-01-01 15:00"}], index=["a", "b", "c"])
    problems = checks.imaging_problems(samples, imaging, example)
    text = " | ".join(str(p) for p in problems)
    assert "exposure" in text and "11 ms in 2" in text and "b (30 ms)" in text
    assert "2026-01-01" in text and "2026-01-08" in text  # sessions per group
    assert not checks.errors(problems)


def test_imaging_without_acquisition_times_reports_the_settings_only(example):
    samples = pd.DataFrame({"sample": ["a", "b"], "group": ["control", "group_a"]})
    imaging = pd.DataFrame([IMAGING, {**IMAGING, "exposure_ms": 30.0}], index=["a", "b"]).drop(columns="acquired")
    problems = checks.imaging_problems(samples, imaging, example)
    assert [p.level for p in problems] == ["note"] and "exposure" in problems[0].message
    assert "imaged on" not in str(problems[0])


def test_group_never_imaged_with_controls_is_a_warning(example):
    samples = pd.DataFrame({"sample": ["a", "b"], "group": ["control", "group_a"]})
    imaging = pd.DataFrame([IMAGING, {**IMAGING, "acquired": "2026-06-05 10:00"}], index=["a", "b"])
    warnings = [p for p in checks.imaging_problems(samples, imaging, example) if p.level == "warning"]
    assert len(warnings) == 1 and "group_a" in warnings[0].message and "same day" in warnings[0].message


def test_check_design(example):
    rows = [("QUAD_sec1_1001-101_control", "control", "1001-101", 1), ("QUAD_sec2_1001-101_control", "control", "1001-101", 2),
            ("QUAD_sec1_1001-102_control", "control", "1001-102", 1), ("QUAD_sec1_1002-101_group_a", "group_a", "1002-101", 1)]
    df = pd.DataFrame(rows, columns=["sample", "group", "animal", "section"]).assign(muscle="QUAD", path="")
    problems = checks.check_design(df, example)
    found = by_where(problems)
    assert found["group_a"].level == "warning" and "1 animal" in found["group_a"].message
    assert not checks.errors(problems)

    twice = pd.concat([df, df.iloc[[0]]])
    assert any(p.level == "error" and "more than once" in p.message for p in checks.check_design(twice, example))
    no_ctrl = df[df["group"] != "control"]
    assert any(p.level == "error" and "control" in p.message for p in checks.check_design(no_ctrl, example))


def test_an_empty_sample_table_is_an_error(example):
    empty = pd.DataFrame(columns=["sample", "group", "animal", "section", "muscle", "path"])
    assert checks.check_design(empty, example) == [checks.Problem("error", "the sample table is empty")]


def test_animal_in_two_groups_is_an_error(example):
    rows = [("QUAD_sec1_1001-101_control", "control", "1001-101", 1), ("QUAD_sec2_1001-101_group_a", "group_a", "1001-101", 2)]
    df = pd.DataFrame(rows, columns=["sample", "group", "animal", "section"]).assign(muscle="QUAD", path="")
    found = by_where(checks.check_design(df, example))
    assert found["1001-101"].level == "error" and "more than one group" in found["1001-101"].message


def test_a_scan_in_the_wrong_folder_takes_the_file_group(example_named, tmp_path):
    touch(tmp_path, "Control/901 ctl QUAD/901-ctl-quad-20full.nd2")
    touch(tmp_path, "Control/901 ctl QUAD/911-GRA-quad-20full.nd2")
    touch(tmp_path, "Group_A/911 GRA QUAD/2-911-GRA-quad-20full.nd2")
    df, problems = checks.samples_from_folder(tmp_path, example_named)
    assert list(df["sample"]) == ["QUAD_sec1_901_control", "QUAD_sec1_911_group_a", "QUAD_sec2_911_group_a"]
    wrong = [p for p in problems if "file name says group group_a" in p.message]
    assert len(wrong) == 1 and wrong[0].level == "warning"
    assert not checks.errors(checks.check_design(df, example_named))
