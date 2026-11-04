import pandas as pd

from msq.core.io import RunFolder
from msq.core.problems import Problem
from msq.core.samples import (check_groups, differing, imaging_days, problems_text, read_headers, read_samples,
                              scan_files, write_checks)
from msq.core.groups import Groups

GROUPS = Groups(order=("ctl", "a"), control="ctl", labels={"ctl": "C", "a": "A"}, colours={"ctl": "#0", "a": "#1"},
                animal="(x)")


def test_scan_files_lists_the_scans_sorted_and_reports_none(tmp_path):
    for rel in ("b/2.nd2", "a/1.nd2", "a/notes.txt", "c/3.npz"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(b"")
    files, problems = scan_files(tmp_path)
    assert [f.name for f in files] == ["1.nd2", "2.nd2"] and problems == []
    files, problems = scan_files(tmp_path, ".npz")
    assert [f.name for f in files] == ["3.npz"]
    files, problems = scan_files(tmp_path / "a", ".tif")
    assert files == [] and problems == [Problem("error", "no .tif scans found in this folder or its subfolders",
                                                str(tmp_path / "a"))]


def test_read_samples_keeps_ids_as_text(tmp_path):
    (tmp_path / "samples.csv").write_text("sample,group,muscle,animal,section\nQ_1,ctl,QUAD,0042,1\n")
    t = read_samples(tmp_path / "samples.csv")
    assert t["animal"].tolist() == ["0042"] and t["section"].tolist() == [1]


def test_read_headers_yields_what_opens_and_reports_what_does_not(tmp_path):
    names = ["ok1", "bad", "ok2", "gone"]
    for n in names[:3]:
        (tmp_path / f"{n}.nd2").write_bytes(b"")
    samples = pd.DataFrame({"sample": [f"s_{n}" for n in names], "rel": [f"r_{n}" for n in names],
                            "path": [str(tmp_path / f"{n}.nd2") for n in names]})

    def read(path):
        if path.name == "bad.nd2":
            raise OSError("no")
        return {"h": path.name}

    problems = []
    seen = [(r.sample, h, len(problems)) for r, h in read_headers(samples, read, problems)]
    # a problem is added when its row comes up, between the rows yielded: the caller's problems follow in table order
    assert seen == [("s_ok1", {"h": "ok1.nd2"}, 0), ("s_ok2", {"h": "ok2.nd2"}, 1), ("s_gone", {"h": "gone.nd2"}, 1)]
    assert [(p.level, p.where) for p in problems] == [("error", "s_bad")]
    assert problems[0].message == "the file cannot be opened (OSError); it may be damaged or still copying"
    named = []
    list(read_headers(samples, read, named, where="rel"))
    assert [p.where for p in named] == ["r_bad"]
    missing = []
    list(read_headers(samples, read, missing, must_exist=True))
    assert [(p.where, p.message) for p in missing] == [
        ("s_bad", "the file cannot be opened (OSError); it may be damaged or still copying"),
        ("s_gone", f"file not found: {tmp_path / 'gone.nd2'}")]


def test_check_groups_names_unknown_groups_and_animals_in_two_groups():
    t = pd.DataFrame({"sample": ["1", "2", "3", "4"], "group": ["ctl", "a", "z", "a"],
                      "animal": ["x", "x", "y", "y"]})
    problems = check_groups(t, GROUPS)
    assert [(p.level, p.where, p.message) for p in problems] == [
        ("error", "3", "group 'z' must be one of ctl, a"),
        ("error", "x", "animal x is in more than one group (ctl, a)"),
        ("error", "y", "animal y is in more than one group (z, a)")]
    assert check_groups(t.iloc[[0, 3]], GROUPS) == []


def test_differing_reports_the_usual_value_and_the_others():
    s = pd.Series({"a": 1.0, "b": 2.0, "c": 1.0, "d": 3.0})
    assert differing(s) == ("1.0", 2, "b (2.0), d (3.0)")
    assert differing(s, lambda v: f"{v:g} ms") == ("1 ms", 2, "b (2 ms), d (3 ms)")
    assert differing(pd.Series({"a": "x", "b": "x"})) is None and differing(pd.Series(dtype=float)) is None


def imaged(days, groups):
    return pd.Series(days), pd.Series(groups)


def test_imaging_days_notes_the_days_of_each_group_and_warns_about_groups_apart_from_the_control():
    day, group = imaged(["d1", "d2", "d1", "d3", "d3"], ["ctl", "ctl", "ctl", "a", "a"])
    notes = imaging_days(day, group, "ctl")
    assert [(p.level, p.where, p.message) for p in notes] == [
        ("note", "ctl", "imaged on 2 days: d1 (2), d2 (1)"), ("note", "a", "imaged on 1 days: d3 (2)"),
        ("warning", "a", "a was never imaged on the same day as the ctl controls, so a difference between imaging "
                         "days cannot be told apart from a difference between groups")]
    per_muscle = imaging_days(day, group, "ctl", "QUAD")
    assert [p.message for p in per_muscle][1] == "QUAD imaged on 1 day: d3 (2)"
    assert per_muscle[2].message.startswith("QUAD: a was never imaged on the same day as the ctl controls")
    assert imaging_days(*imaged(["d1", "d1"], ["ctl", "a"]), "ctl")[-1].level == "note"  # same day: no warning


def test_imaging_days_skips_scans_without_a_day_or_a_group():
    day, group = imaged(["d1", None, "d1", "d2"], ["ctl", "ctl", None, "a"])
    assert [p.message for p in imaging_days(day, group, "ctl")][:2] == ["imaged on 1 days: d1 (1)",
                                                                       "imaged on 1 days: d2 (1)"]


def test_write_checks_writes_the_text_and_the_sample_table(tmp_path):
    run = RunFolder(tmp_path / "new" / "run")
    text = write_checks(run, [Problem("warning", "10 µm …", "s")], pd.DataFrame({"sample": ["s"]}))
    assert text == "WARNING: s: 10 µm …" and run.checks.read_bytes() == (text + "\n").encode("utf-8")
    assert run.samples.read_text() == "sample\ns\n"
    assert problems_text([]) == "no problems found" and problems_text([], "No problems found.") == "No problems found."
    assert write_checks(RunFolder(tmp_path / "other"), []) == "no problems found"
    assert not (tmp_path / "other" / "samples.csv").exists()
