"""What the assays' sample tables and checks have in common: finding the scans, reading the table back, opening each
scan's header, the checks on groups, a setting that differs between scans, the days each group was imaged, and
writing checks.txt."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterator

import pandas as pd

from msq.core.groups import Groups
from msq.core.io import RunFolder
from msq.core.problems import Problem


def scan_files(root: Path, suffix: str = ".nd2") -> tuple[list[Path], list[Problem]]:
    """Every scan below root, sorted, and an error when there is none."""
    files = sorted(root.rglob(f"*{suffix}"))
    problems = [] if files else [Problem("error", f"no {suffix} scans found in this folder or its subfolders",
                                         str(root))]
    return files, problems


def read_samples(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"animal": str, "sample": str, "group": str, "muscle": str})


def read_headers(samples: pd.DataFrame, read_header: Callable[[Path], dict], problems: list[Problem],
                 where: str = "sample", must_exist: bool = False) -> Iterator[tuple[object, dict]]:
    """(row, header) of every scan that opens, one at a time, so the caller's problems for a scan follow the ones
    added here. A scan that does not open (or, with must_exist, is not there) is added to problems, named by its
    column where."""
    for r in samples.itertuples():
        path = Path(r.path)
        if must_exist and not path.exists():
            problems.append(Problem("error", f"file not found: {path}", r.sample))
            continue
        try:
            h = read_header(path)
        except Exception as e:  # any reader failure means the file cannot be used
            problems.append(Problem("error", f"the file cannot be opened ({type(e).__name__}); it may be damaged "
                                             "or still copying", getattr(r, where)))
            continue
        yield r, h


def check_groups(samples: pd.DataFrame, groups: Groups) -> list[Problem]:
    """A group that is not one of the dataset's, and an animal that is in more than one group."""
    problems = []
    for r in samples.itertuples():
        if r.group not in groups.order:
            problems.append(Problem("error", f"group '{r.group}' must be one of {', '.join(groups.order)}",
                                    r.sample))
    groups_per_animal = samples.groupby("animal")["group"].unique()
    for animal, gs in groups_per_animal[groups_per_animal.map(len) > 1].items():
        problems.append(Problem("error", f"animal {animal} is in more than one group ({', '.join(gs)})", animal))
    return problems


def differing(values: pd.Series, show: Callable = str) -> tuple[str, int, str] | None:
    """For a setting read per scan (index: sample): the usual value as shown, how many scans have it and the others
    as 'sample (value)', or None when all scans agree. values: without missing ones."""
    if values.nunique() <= 1:
        return None
    usual = values.value_counts().index[0]
    return show(usual), (values == usual).sum(), ", ".join(f"{s} ({show(v)})" for s, v in values.items() if v != usual)


def imaging_days(day: pd.Series, group: pd.Series, control: str, muscle: str | None = None) -> list[Problem]:
    """A note per group with the days it was imaged (and the scans of each day), and a warning for a group never
    imaged on a day with the control. day, group: one entry per scan. muscle: named in the messages when the days are
    per muscle (the note then says '1 day'; without it, as psr always said, '1 days')."""
    problems = []
    days = {g: sorted(day[group == g].dropna().unique()) for g in dict.fromkeys(group.dropna())}
    for g, ds in days.items():
        unit = "day" if muscle and len(ds) == 1 else "days"
        problems.append(Problem("note", f"{muscle + ' ' if muscle else ''}imaged on {len(ds)} {unit}: " + ", ".join(
            f"{d} ({((day == d) & (group == g)).sum()})" for d in ds), g))
    control_days = set(days.get(control, []))
    for g, ds in days.items():
        if g != control and control_days and not control_days & set(ds):
            problems.append(Problem("warning", f"{muscle + ': ' if muscle else ''}{g} was never imaged on the same day "
                                               f"as the {control} controls, so a difference between imaging days "
                                               "cannot be told apart from a difference between groups", g))
    return problems


def problems_text(problems: list[Problem], none: str = "no problems found") -> str:
    return "\n".join(str(p) for p in problems) or none


def write_checks(run: RunFolder, problems: list[Problem], samples: pd.DataFrame | None = None) -> str:
    """checks.txt (one problem per line), and samples.csv when samples is given, into the run folder; returns the
    text."""
    run.root.mkdir(parents=True, exist_ok=True)
    text = problems_text(problems)
    run.checks.write_text(text + "\n", encoding="utf-8")
    if samples is not None:
        samples.to_csv(run.samples, index=False)
    return text
