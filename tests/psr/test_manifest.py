from pathlib import Path

import pytest

from msq.psr import manifest as mf

ROOT = Path("/data/scans")


@pytest.mark.parametrize("rel, expected", [
    # (group, muscle, animal, section, full)
    ("1002 group a/1002-101-quad/1002-101-20x-full.nd2", ("group_a", "QUAD", "1002-101", 1, True)),
    ("1002 group a/1002-101-quad/2-1002-101-20x-full.nd2", ("group_a", "QUAD", "1002-101", 2, True)),
    ("1002 group a/1002-102-QUAD/3-1002-102-20full.nd2", ("group_a", "QUAD", "1002-102", 3, True)),
    ("1001 control/1001-103-quad/2-1001-103-20xfull.nd2", ("control", "QUAD", "1001-103", 2, True)),
    ("1003 group b/1003 101 QUAD/1003-101-20full.nd2", ("group_b", "QUAD", "1003-101", 1, True)),
    ("1003 group b/1003 101 QUAD/3-1003-101-10.1.nd2", ("group_b", "QUAD", "1003-101", 3, False)),
    ("1002 group a/1002-103-quad/2-1002-103-40.1.2x.nd2", ("group_a", "QUAD", "1002-103", 2, False)),
    ("1001 control/1001-102-quad/1001-102-30x.nd2", ("control", "QUAD", "1001-102", 1, False)),
])
def test_parse_handles_naming_variants(example, rel, expected):
    r = mf.parse(ROOT / rel, ROOT, example)
    assert (r["group"], r["muscle"], r["animal"], r["section"], r["full"]) == expected


def test_sample_id(example):
    r = mf.parse(ROOT / "1001 control/1001-103-quad/2-1001-103-20xfull.nd2", ROOT, example)
    assert mf.sample_id(r) == "QUAD_sec2_1001-103_control"


def test_group_comes_from_the_folders_below_root_only(example):
    root = Path("/lab/group a archive")  # a group word above root does not count
    assert mf.parse(root / "1001 control/1001-103-quad/1001-103-20xfull.nd2", root, example)["group"] == "control"


def test_unknown_group_folder_raises(example):
    with pytest.raises(ValueError, match="group"):
        mf.parse(ROOT / "9999 other/9999-001-quad/9999-001-20full.nd2", ROOT, example)


def test_animal_missing_raises(example):
    with pytest.raises(ValueError, match="animal"):
        mf.parse(ROOT / "1002 group a/overview/overview-20full.nd2", ROOT, example)


def test_folder_and_file_disagree_on_animal(example):
    r = mf.parse(ROOT / "1002 group a/1002-101-quad/1002-103-20x-full.nd2", ROOT, example)
    assert r["animal"] == "1002-103" and r["folder_animal"] == "1002-101"


def test_muscle_from_folder_or_file(example):
    r = mf.parse(ROOT / "1002 group a/1002-101/1002-101-QUAD-20full.nd2", ROOT, example)
    assert r["muscle"] == "QUAD"
    with pytest.raises(ValueError, match="muscle"):
        mf.parse(ROOT / "1002 group a/1002-101/1002-101-20full.nd2", ROOT, example)


def test_group_folder_must_match_as_a_whole(example):
    with pytest.raises(ValueError, match="group"):
        mf.parse(ROOT / "1002 group a old/1002-101-quad/1002-101-20x-full.nd2", ROOT, example)


# ---------------------------------------------------------------- a cohort that numbers animals alone and names the group in the file


@pytest.mark.parametrize("rel, expected", [
    # (group, animal, section, full, folder_animal, file_group)
    ("Control/901 ctl QUAD/901-ctl-quad-20full.nd2", ("control", "901", 1, True, "901", "control")),
    ("Control/901 ctl QUAD/2-901-ctl-quad-20full.nd2", ("control", "901", 2, True, "901", "control")),
    ("Control/902 ctl quad/2_903_ctl_10.1x.nd2", ("control", "903", 2, False, "902", "control")),
    ("Group_A/911 GRA QUAD/3-911-GRA-quad-20full.nd2", ("group_a", "911", 3, True, "911", "group_a")),
    ("Group_B/921_GRB_QUAD/921-GRB-quad-20full.nd2", ("group_b", "921", 1, True, "921", "group_b")),
    # a word that only starts like a group's pattern names no group
    ("Group_B/922 GRB QUAD/2-922-GRC-quad-20full.nd2", ("group_b", "922", 2, True, "922", None)),
    # filed in the wrong folder: the folder gives the group, the file name says otherwise
    ("Control/901 ctl QUAD/911-GRA-quad-20full.nd2", ("control", "911", 1, True, "901", "group_a")),
])
def test_named_cohort_naming(example_named, rel, expected):
    r = mf.parse(ROOT / rel, ROOT, example_named)
    assert (r["group"], r["animal"], r["section"], r["full"], r["folder_animal"], r["file_group"]) == expected
    assert r["muscle"] == "QUAD"


def test_named_cohort_sample_id(example_named):
    r = mf.parse(ROOT / "Control/901 ctl QUAD/2-901-ctl-quad-20full.nd2", ROOT, example_named)
    assert mf.sample_id(r) == "QUAD_sec2_901_control"


def test_cohort_mismatch_raises(example):
    with pytest.raises(ValueError, match="cohort example"):
        mf.parse(ROOT / "Group_A/911 GRA QUAD/911-GRA-quad-20full.nd2", ROOT, example)
