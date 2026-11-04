"""The groups of an experiment: their order, the control, their names and colours in tables and figures, and how an
animal is numbered in a file name. dystrophin has two (settings.GROUPS); psr and necrosis read the lab's from
configuration outside the repository, as a subclass that adds what names a group on disk (load_groups).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeVar

import pandas as pd

from msq.core.config import local_config


@dataclass(frozen=True)
class Groups:
    """The groups of one dataset."""

    order: tuple[str, ...]            # plotting and table order, control first
    control: str                      # the control group (necrosis: the reference group, which sets the threshold)
    labels: dict[str, str]            # group -> name in figures and tables
    colours: dict[str, str]           # group -> colour
    animal: str                       # regular expression; its groups, joined by "-", are the animal id

    def __post_init__(self) -> None:
        if self.control not in self.order:
            raise ValueError(f"{self.who}: the control group {self.control!r} is not in its order")
        for what, table in (("labels", self.labels), ("colours", self.colours)):
            missing = [g for g in self.order if g not in table]
            if missing:
                raise ValueError(f"{self.who}: no {what} for {', '.join(missing)}")
        re.compile(self.animal)  # a bad pattern fails here, not on the first scan

    @property
    def who(self) -> str:
        """What error messages call these groups."""
        return "the groups config"

    @property
    def control_label(self) -> str:
        return self.labels[self.control]

    def sort_key(self) -> Callable[[pd.Series], pd.Series]:
        """key= for DataFrame.sort_values: a column called 'group' sorts in this order, any other column as it is."""
        order = {g: i for i, g in enumerate(self.order)}
        return lambda s: s.map(order) if s.name == "group" else s

    def palette(self, present=None) -> pd.DataFrame:
        """group, label, colour and control flag in plotting order; present: only the groups that have data."""
        shown = [g for g in self.order if present is None or g in set(present)]
        return pd.DataFrame({"group": shown, "label": [self.labels[g] for g in shown],
                             "colour": [self.colours[g] for g in shown],
                             "control": [g == self.control for g in shown]})


G = TypeVar("G", bound=Groups)


def load_groups(file: str, cls: type[G], name: str | None = None, root: Path | None = None) -> G:
    """cls.from_config() of <file>.local.toml, else <file>.example.toml (msq.core.config says where they are looked
    for). name: the table of one dataset in a file that holds several, else the whole file."""
    config = local_config(file, root)
    if name is None:
        return cls.from_config(config)
    if not isinstance(config.get(name), dict):
        known = [k for k, v in config.items() if isinstance(v, dict)]
        raise ValueError(f"unknown cohort '{name}' (known in {Path(config['_source']).name}: {', '.join(known)})")
    return cls.from_config(config[name], name)
