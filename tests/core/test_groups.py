from dataclasses import dataclass

import pandas as pd
import pytest

from msq.core import config
from msq.core.groups import Groups, load_groups

TABLE = {"order": ["ctl", "a", "b"], "control": "ctl", "labels": {"ctl": "Control", "a": "A", "b": "B"},
         "colours": {"ctl": "#444444", "a": "#1b9e77", "b": "#d95f02"}, "animal": r"(\d{3})"}


def groups(**change) -> Groups:
    t = {**TABLE, **change}
    return Groups(order=tuple(t["order"]), control=t["control"], labels=t["labels"], colours=t["colours"],
                  animal=t["animal"])


@pytest.mark.parametrize("change, message", [
    ({"control": "x"}, "the groups config: the control group 'x' is not in its order"),
    ({"labels": {"ctl": "C"}}, "the groups config: no labels for a, b"),
    ({"colours": {}}, "the groups config: no colours for ctl, a, b"),
])
def test_groups_must_be_consistent(change, message):
    with pytest.raises(ValueError, match=message):
        groups(**change)


def test_an_invalid_animal_pattern_fails_when_the_groups_are_made():
    with pytest.raises(Exception, match="missing"):
        groups(animal="(\\d{3}")


def test_subclasses_name_themselves_in_the_errors():
    @dataclass(frozen=True)
    class Named(Groups):
        name: str = "mine"

        @property
        def who(self) -> str:
            return f"cohort {self.name}"

    with pytest.raises(ValueError, match="cohort mine: no labels for a, b"):
        Named(order=("ctl", "a", "b"), control="ctl", labels={"ctl": "C"}, colours=TABLE["colours"], animal="x")


def test_control_label_names_the_control():
    assert groups().control_label == "Control"


def test_sort_key_orders_the_group_column_only():
    t = pd.DataFrame({"group": ["b", "ctl", "a", "ctl"], "animal": ["3", "2", "1", "1"]})
    out = t.sort_values(["group", "animal"], key=groups().sort_key(), kind="stable")
    assert list(zip(out["group"], out["animal"])) == [("ctl", "1"), ("ctl", "2"), ("a", "1"), ("b", "3")]
    assert t.sort_values("animal", key=groups().sort_key())["animal"].tolist() == ["1", "1", "2", "3"]


def test_the_palette_lists_the_groups_in_order_and_only_those_present():
    g = groups(order=["a", "ctl", "b"])
    p = g.palette()
    assert p["group"].tolist() == ["a", "ctl", "b"] and p["control"].tolist() == [False, True, False]
    assert p["label"].tolist() == ["A", "Control", "B"] and p["colour"].tolist()[1] == "#444444"
    assert g.palette({"b", "a", "unknown"})["group"].tolist() == ["a", "b"]
    assert g.palette(pd.Series(["b"]))["group"].tolist() == ["b"]
    assert g.palette([]).empty and list(g.palette([]).columns) == ["group", "label", "colour", "control"]


@dataclass(frozen=True)
class FromTable(Groups):
    name: str | None = None

    @classmethod
    def from_config(cls, cfg: dict, name: str | None = None) -> "FromTable":
        return cls(order=tuple(cfg["order"]), control=cfg["control"], labels=cfg["labels"], colours=cfg["colours"],
                   animal=cfg["animal"], name=name)


def test_load_groups_reads_a_whole_file_or_one_table_of_it(tmp_path, monkeypatch):
    monkeypatch.setenv(config.ENV_VAR, str(tmp_path))
    (tmp_path / "solo.example.toml").write_text(
        'order = ["ctl", "a"]\ncontrol = "ctl"\nanimal = "(\\\\d{3})"\n[labels]\nctl = "C"\na = "A"\n'
        '[colours]\nctl = "#000000"\na = "#ffffff"\n')
    g = load_groups("solo", FromTable)
    assert g.order == ("ctl", "a") and g.name is None and g.labels == {"ctl": "C", "a": "A"}
    (tmp_path / "many.local.toml").write_text(
        'ignored = 1\n[one]\norder = ["ctl"]\ncontrol = "ctl"\nanimal = "x"\n[one.labels]\nctl = "C"\n'
        '[one.colours]\nctl = "#000000"\n')
    one = load_groups("many", FromTable, "one")
    assert one.name == "one" and one.order == ("ctl",)
    with pytest.raises(ValueError, match="unknown cohort 'two' .known in many.local.toml: one"):
        load_groups("many", FromTable, "two")
