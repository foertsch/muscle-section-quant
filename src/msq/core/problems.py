"""Problems a run reports before, during and after it works, and the exit statuses a runner ends with.

Every assay's checks return a list of Problem. "error" stops a run before it starts, "warning" is reported and the
run goes on, "note" is for information only. A problem may carry a code, so code that acts on one (the runner, the
app) keeps working when its wording changes. checks.json round-trips through dataclasses.asdict; problems written
before the code field existed load with an empty code.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

LEVELS = ("error", "warning", "note")


@dataclass
class Problem:
    level: str  # "error", "warning" or "note"
    message: str
    where: str = ""  # sample name, group, muscle or file
    code: str = ""  # for code that acts on a problem, so a reworded message does not break it

    def __str__(self) -> str:
        return f"{self.level.upper()}: {self.where + ': ' if self.where else ''}{self.message}"


def errors(problems: list[Problem]) -> list[Problem]:
    return [p for p in problems if p.level == "error"]


class ExitStatus(IntEnum):
    """What a runner's exit status means (dystrophin-quant's run_pipeline, kept for every assay)."""

    DONE = 0  # finished
    CHECKS_FAILED = 1  # stopped by errors in the checks, nothing measured
    INCOMPLETE = 2  # done, but some sections, reports or figures failed
    STOPPED = 3  # stopped on request before all sections were measured
    CRASHED = 4  # stopped by an unexpected error (the traceback is in the log)
    LOCKED = 5  # not started: another runner holds the run folder's lock
