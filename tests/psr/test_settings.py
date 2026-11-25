import re

import pytest

from msq.core import config, groups
from msq.psr import settings

from tests.psr.conftest import example_config

TABLE = {"folders": {"ctl": "control", "treated": "group_a"}, "order": ["control", "group_a"], "control": "control",
         "labels": {"control": "Control", "group_a": "Group A"}, "colours": {"control": "#444444", "group_a": "#1b9e77"},
         "animal": r"(\d{3})"}


def test_the_example_file_has_two_generic_cohorts(example, example_named):
    assert example.order == example_named.order == ("control", "group_a", "group_b")
    assert example.control == example_named.control == "control"
    assert set(example.labels) == set(example.colours) == set(example.folders.values()) == set(example.order)
    assert example.file_groups == {} and set(example_named.file_groups) == set(example_named.order)
    assert {k for k, v in example_config().items() if isinstance(v, dict)} == {"example", "example_named"}


def test_a_local_cohorts_file_wins_over_the_example(tmp_path, monkeypatch):
    monkeypatch.setenv(config.ENV_VAR, str(tmp_path))
    (tmp_path / "cohorts.local.toml").write_text(
        '[mine]\norder = ["control", "group_a"]\ncontrol = "control"\nanimal = \'(\\d{3})\'\n'
        '[mine.folders]\nctl = "control"\n[mine.labels]\ncontrol = "C"\ngroup_a = "A"\n'
        '[mine.colours]\ncontrol = "#000000"\ngroup_a = "#ffffff"\n')
    c = settings.load_cohort("mine")
    assert c.name == "mine" and c.folders == {"ctl": "control"} and c.order == ("control", "group_a")
    assert c.file_groups == {}


def test_unknown_cohort_raises(monkeypatch):
    monkeypatch.setattr(groups, "local_config", lambda name, root=None: example_config())
    with pytest.raises(ValueError, match="unknown cohort 'other'.*example, example_named"):
        settings.load_cohort("other")


def test_the_repository_example_loads_without_a_local_file(monkeypatch, tmp_path):
    monkeypatch.setenv(config.ENV_VAR, str(tmp_path))  # an empty folder first; the example comes from the repository
    monkeypatch.chdir(tmp_path)
    if config.find_config("cohorts")[1]:
        pytest.skip("a cohorts.local.toml exists here, and wins over the example")
    assert settings.load_cohort("example").control == "control"


@pytest.mark.parametrize("change, message", [
    ({"control": "other"}, "control group 'other' is not in its order"),
    ({"labels": {"control": "Control"}}, "no labels for group_a"),
    ({"colours": {"group_a": "#1b9e77"}}, "no colours for control"),
    ({"folders": {"ctl": "control", "x": "group_c"}}, "group_c named in folders or file_groups"),
    ({"file_groups": {"group_c": "gc"}}, "group_c named in folders or file_groups"),
])
def test_a_cohort_must_be_consistent(change, message):
    with pytest.raises(ValueError, match=message):
        settings.Cohort.from_config({**TABLE, **change}, "t")


def test_an_invalid_animal_pattern_fails_when_the_cohort_is_made():
    with pytest.raises(re.error):
        settings.Cohort.from_config({**TABLE, "animal": "(\\d{3}"}, "t")


def test_a_missing_key_names_the_cohort_and_the_key():
    table = {k: v for k, v in TABLE.items() if k != "animal"}
    with pytest.raises(ValueError, match="cohort t: the table has no 'animal'"):
        settings.Cohort.from_config(table, "t")
