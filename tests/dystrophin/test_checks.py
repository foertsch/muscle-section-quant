from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pytest

from msq.dystrophin import checks  # noqa: E402
from msq.core.synthetic.dystrophin import HEADER  # noqa: E402


def touch(root: Path, rel: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")
    return p


@pytest.fixture
def scans(tmp_path):
    root = tmp_path / "scans"
    touch(root, "Neg-ctrl-TA/1106/TA sec1 1106 20x.nd2")
    touch(root, "Neg-ctrl-TA/1107/TA sec1 1107 20x.nd2")
    touch(root, "1104 TA/TA sec2 1104 20x.nd2")
    touch(root, "1104 TA/Edited sec2 TA 1104 20x.nd2")  # same section, edited copy
    touch(root, "1104 TA/TA sec2 1104 4x.nd2")
    touch(root, "misc/overview.nd2")
    return root


def levels(problems):
    return sorted(p.level for p in problems)


def test_sample_table_from_folder(scans):
    df, problems = checks.samples_from_folder(scans)
    assert list(df["sample"]) == ["TA_sec1_1106_20x_ctrl", "TA_sec1_1107_20x_ctrl", "TA_sec2_1104_20x_treated"]
    assert list(df.columns) == checks.COLUMNS
    assert Path(df.loc[2, "path"]).name == "TA sec2 1104 20x.nd2"  # the raw file wins over the edited copy
    assert all(Path(p).is_absolute() for p in df["path"])
    msgs = {p.where: p for p in problems}
    assert msgs["TA_sec2_1104_20x_treated"].level == "warning"  # duplicate
    assert msgs[str(Path("misc/overview.nd2"))].level == "warning"  # name cannot be read
    assert msgs[""].level == "note" and "1 at 4x" in msgs[""].message  # other magnification
    assert not checks.errors(problems)


def test_the_magnification_and_the_file_suffix_can_be_chosen(scans):
    touch(scans, "1104 TA/TA sec2 1104 40x.nd2")
    touch(scans, "1104 TA/TA sec4 1104 20x.npz")
    npz, _ = checks.samples_from_folder(scans, suffix=".npz")
    assert list(npz["sample"]) == ["TA_sec4_1104_20x_treated"]
    forty, problems = checks.samples_from_folder(scans, mag="40x")
    assert list(forty["sample"]) == ["TA_sec2_1104_40x_treated"]
    assert [p.level for p in problems] == ["warning", "note"] and "only 40x scans are analysed" in problems[-1].message


def test_empty_folder_is_an_error(tmp_path):
    df, problems = checks.samples_from_folder(tmp_path)
    assert df.empty and levels(problems) == ["error"]


def test_check_files(scans):
    df, _ = checks.samples_from_folder(scans)
    assert checks.check_files(df, read_header=lambda p: HEADER) == []

    headers = {
        "TA_sec1_1106_20x_ctrl": {**HEADER, "channels": ["DAPI", "FITC", "Cy5"]},  # no GFP
        "TA_sec1_1107_20x_ctrl": {**HEADER, "px_um": 1.83},  # a 4x scan named 20x
        "TA_sec2_1104_20x_treated": {**HEADER, "sizes": {"Z": 5, "C": 3, "Y": 100, "X": 100}},
    }
    by_path = dict(zip(df["path"], df["sample"]))
    problems = checks.check_files(df, read_header=lambda p: headers[by_path[str(p)]])
    text = {p.where: p.message for p in problems}
    assert "GFP" in text["TA_sec1_1106_20x_ctrl"]
    assert "not a 20x scan" in text["TA_sec1_1107_20x_ctrl"]
    assert "dimensions" in text["TA_sec2_1104_20x_treated"]
    assert levels(problems) == ["error"] * 3


def test_unreadable_and_missing_files(scans):
    df, _ = checks.samples_from_folder(scans)
    Path(df.loc[0, "path"]).unlink()

    def broken(p):
        raise OSError("not an ND2 file")

    problems = checks.check_files(df, read_header=broken)
    assert "file not found" in problems[0].message
    assert all("cannot be opened" in p.message for p in problems[1:])
    assert levels(problems) == ["error"] * 3


def table(rows):
    """(sample, group, muscle, animal) rows; the section comes from the sample name."""
    df = pd.DataFrame([dict(zip(["sample", "group", "muscle", "animal"], r)) for r in rows])
    return df.assign(section=df["sample"].str.extract(r"_sec(\d+)_", expand=False).astype(int))


def test_design_needs_controls_or_a_borrowed_threshold():
    df = table([("TA_sec1_1104_20x_treated", "treated", "TA", "1104")])
    (p,) = checks.check_design(df)
    assert p.level == "error" and "no untreated control" in p.message
    (p,) = checks.check_design(df, borrowed={"TA": {"threshold": 449.0, "source": "run A"}})
    assert p.level == "warning" and "449" in p.message and "run A" in p.message


def test_design_warnings_and_errors():
    df = table([("TA_sec1_1106_20x_ctrl", "ctrl", "TA", "1106"),
                ("TA_sec1_1104_20x_treated", "treated", "TA", "1104"),
                ("QUAD_sec1_1106_20x_ctrl", "ctrl", "QUAD", "1106"),
                ("QUAD_sec1_1107_20x_ctrl", "ctrl", "QUAD", "1107"),
                ("QUAD_sec1_1107_20x_ctrl", "control", "QUAD", "1107"),
                ("EDL_sec1_1107_20x_ctrl", "ctrl", "EDL", "1107")])
    problems = checks.check_design(df)

    def has(level, where, text):
        return any(p.level == level and p.where == where and text in p.message for p in problems)

    assert has("warning", "TA", "only 1 control animal")
    assert has("error", "QUAD_sec1_1107_20x_ctrl", "more than once")
    assert has("error", "QUAD_sec1_1107_20x_ctrl", "group 'control'")
    assert has("error", "EDL_sec1_1107_20x_ctrl", "muscle 'EDL'")
    assert has("note", "QUAD", "controls only")


def has(problems, level, where, text):
    return any(p.level == level and p.where == where and text in p.message for p in problems)


@pytest.mark.parametrize("animal", ["", "1104/2", "1104\\2", "14 11", "1104_L"])
def test_animal_numbers_that_break_sample_or_folder_names_are_errors(animal):
    df = table([("TA_sec1_1106_20x_ctrl", "ctrl", "TA", "1106"), ("TA_sec1_1107_20x_ctrl", "ctrl", "TA", "1107"),
                (f"TA_sec2_{animal}_20x_treated", "treated", "TA", animal)])
    if animal == "":
        df.loc[2, "animal"] = float("nan")  # how read_samples reads an empty cell
    problems = checks.check_design(df)
    assert has(problems, "error", f"TA_sec2_{animal}_20x_treated", "must be letters, digits or '-'")
    assert len(checks.errors(problems)) == 1


def test_an_empty_animal_on_a_control_row_is_an_error_not_a_crash():
    df = table([("TA_sec1_1106_20x_ctrl", "ctrl", "TA", "1106"), ("TA_sec1__20x_ctrl", "ctrl", "TA", ""),
                ("TA_sec2_1104_20x_treated", "treated", "TA", "1104")])
    df.loc[1, "animal"] = float("nan")
    problems = checks.check_design(df)
    assert has(problems, "error", "TA_sec1__20x_ctrl", "must be letters, digits or '-'")


def test_valid_animal_numbers_pass():
    df = table([("TA_sec1_1106_20x_ctrl", "ctrl", "TA", "1106"), ("TA_sec1_1107-2_20x_ctrl", "ctrl", "TA", "1107-2"),
                ("TA_sec2_M7_20x_treated", "treated", "TA", "M7")])
    assert checks.errors(checks.check_design(df)) == []


def test_one_section_in_both_groups_is_an_error(tmp_path):
    root = tmp_path / "scans"
    touch(root, "Neg-ctrl-TA/1106/TA sec1 1106 20x.nd2")
    touch(root, "Neg-ctrl-TA/1107/TA sec1 1107 20x.nd2")
    touch(root, "1104 TA/TA sec2 1104 20x.nd2")
    touch(root, "1104 TA/TA sec1 1106 20x.nd2")  # a control scan copied into a treated folder
    df, _ = checks.samples_from_folder(root)
    problems = checks.check_design(df)
    assert has(problems, "error", "TA_sec1_1106_20x_ctrl, TA_sec1_1106_20x_treated", "more than once")
    assert has(problems, "error", "TA", "animal 1106 is both control and treated")


def test_an_animal_in_both_groups_of_one_muscle_is_an_error():
    df = table([("TA_sec1_1106_20x_ctrl", "ctrl", "TA", "1106"), ("TA_sec1_1107_20x_ctrl", "ctrl", "TA", "1107"),
                ("TA_sec2_1107_20x_treated", "treated", "TA", "1107"),
                ("QUAD_sec1_1105_20x_ctrl", "ctrl", "QUAD", "1105"), ("QUAD_sec1_1106_20x_ctrl", "ctrl", "QUAD", "1106"),
                ("TA_sec1_1105_20x_treated", "treated", "TA", "1105")])
    problems = checks.check_design(df)
    assert has(problems, "error", "TA", "animal 1107 is both control and treated")
    assert len(checks.errors(problems)) == 1  # 1105: control in QUAD, treated in TA, which is fine


def test_group_comes_from_folders_below_the_scan_folder(tmp_path):
    root = tmp_path / "Neg-ctrl comparison 2027" / "batch2"  # the scan folder sits inside a Neg-ctrl* folder
    touch(root, "1104 TA/TA sec2 1104 20x.nd2")
    touch(root, "Neg-ctrl-TA/1107/TA sec1 1107 20x.nd2")
    df, _ = checks.samples_from_folder(root)
    assert dict(zip(df["animal"], df["group"])) == {"1104": "treated", "1107": "ctrl"}


def test_control_folders_spelled_another_way_give_a_warning(tmp_path):
    root = tmp_path / "scans"
    touch(root, "Neg-ctrl-TA/1106/TA sec1 1106 20x.nd2")
    touch(root, "Neg ctrl TA/1107/TA sec1 1107 20x.nd2")
    touch(root, "NegCtrl/1108/TA sec1 1108 20x.nd2")
    touch(root, "controls_TA/1109/TA sec1 1109 20x.nd2")
    touch(root, "1104 TA/TA sec2 1104 20x.nd2")
    df, problems = checks.samples_from_folder(root)
    warned = {p.where for p in problems if "looks like a control folder" in p.message}
    assert warned == {"Neg ctrl TA", "NegCtrl", "controls_TA"}
    assert set(df.loc[df["group"] == "treated", "animal"]) == {"1104", "1107", "1108", "1109"}


def test_file_names_with_underscores_and_extra_numbers(tmp_path):
    root = tmp_path / "scans"
    touch(root, "Neg-ctrl-TA/1106/TA_sec1_1106_20x.nd2")
    touch(root, "1104 TA/TA sec2 1104 20x 1500 ms.nd2")
    df, problems = checks.samples_from_folder(root)
    assert list(df["sample"]) == ["TA_sec1_1106_20x_ctrl", "TA_sec2_1500_20x_treated"]
    assert has(problems, "warning", str(Path("1104 TA/TA sec2 1104 20x 1500 ms.nd2")),
               "animal read as 1500, but the file name also contains 1104")


DESCRIPTION = """Camera Name: Camera Nikon DS-Qi2
Plane #1:
 Name: DAPI
 Camera Settings:
  Exposure: 200 ms
  Gain: 1.0x
 Microscope Settings:   Microscope: Ti2 Microscope
  Nikon Ti2, Illuminator(DIA) Iris intensity: 23.8
  Sola II, Illuminator(Sola) Iris intensity: 10.0
Plane #2:
 Name: GFP
 Camera Settings:
  Exposure: 2 s
  Gain: 1.0x
 Microscope Settings:   Microscope: Ti2 Microscope
  Nikon Ti2, Illuminator(DIA) Iris intensity: 23.8
  Sola II, Illuminator(Sola) Iris intensity: 10.0
"""


def test_imaging_settings_must_match_within_a_muscle(scans):
    df, _ = checks.samples_from_folder(scans)
    df = pd.concat([df, df.assign(sample=df["sample"].str.replace("TA", "QUAD"), muscle="QUAD")], ignore_index=True)
    gfp = {s: dict(HEADER["gfp"]) for s in df["sample"]}
    for s in gfp:
        if s.startswith("QUAD"):
            gfp[s]["exposure"] = "1 s"  # another exposure than TA, but the same within QUAD: fine
    gfp["TA_sec1_1107_20x_ctrl"]["gain"] = "2.0x"
    rows = iter(df.itertuples())  # check_files reads the headers in table order
    problems = checks.check_files(df, read_header=lambda p: {**HEADER, "gfp": gfp[next(rows).sample]})
    assert [(p.level, p.where) for p in problems] == [("warning", "TA")]
    assert "camera gain differs" in problems[0].message and "TA_sec1_1107_20x_ctrl (2.0x)" in problems[0].message
    no_settings = {k: v for k, v in HEADER.items() if k != "gfp"}  # files whose settings cannot be read
    assert checks.check_files(df, read_header=lambda p: no_settings) == []


def test_a_borrowed_threshold_whose_settings_were_not_compared_says_so():
    df = table([("TA_sec1_1104_20x_treated", "treated", "TA", "1104")])
    (p,) = checks.check_design(df, borrowed={"TA": {"threshold": 449.0, "source": "run A", "checked": False}})
    assert p.level == "warning" and "could not be compared" in p.message


def test_problems_carry_a_code_and_old_ones_without_it_still_load():
    df = table([("TA_sec1_1104_20x_treated", "treated", "TA", "1104")])
    (p,) = checks.check_design(df)
    assert p.code == checks.NO_CONTROLS
    assert checks.Problem(**asdict(p)) == p  # checks.json round trip
    old = checks.Problem(**{"level": "error", "message": "no untreated control sections.", "where": "TA"})
    assert old.code == "" and str(old) == "ERROR: TA: no untreated control sections."


def test_other_cellpose_weights_give_a_warning():
    reference = {"file": "cpsam_v2", "bytes": 1, "sha256": checks.settings.REFERENCE_WEIGHTS_SHA256}
    assert checks.check_weights(None) == [] and checks.check_weights(reference) == []
    (p,) = checks.check_weights({**reference, "sha256": "ab" * 32})
    assert p.level == "warning" and p.code == checks.OTHER_WEIGHTS and "not the reference ones" in p.message
