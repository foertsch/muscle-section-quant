"""What the pipeline accepts and how it reads the experiment, in one place.

The sample table (manifest.py) and the checks (checks.py) read these values, so changing the scope here changes
both. Image-processing parameters live with the section step.

How the groups of a dataset are named is not here: it differs per dataset and the lab's own definitions stay out of
the repository. A Cohort says how one dataset names its groups on disk and in figures, which group is the control and
how its animals are numbered; load_cohort() reads it from cohorts.local.toml (cohorts.example.toml has generic groups)
and every function that needs groups takes it as an argument.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from msq.core.groups import Groups, load_groups

MUSCLES = ("QUAD",)                 # muscle names read from folder or file names
CHANNEL = "Brightfield"             # the one ND2 channel of the colour camera (Nikon DS-Fi3)
FILE_ORDER = "BGR"                  # nd2 returns the colour planes in this order: read as-is, PSR yellow looks cyan
                                    # and red looks blue. Flip to RGB on read.
FULL_SCAN_MARK = "full"             # whole-section scans have this in the name; the single fields do not
PIXEL_UM_RANGE = (0.20, 0.23)       # 20x full scans are 0.2167 µm/px; the single fields are 0.22 to 0.87


@dataclass(frozen=True)
class Cohort(Groups):
    """The groups of one dataset (see Groups) and how its scans name them: the folders, the file names."""

    name: str
    folders: dict[str, str]           # top folder below the scan folder, without a leading number, any case -> group
    file_groups: dict[str, str] = field(default_factory=dict)  # group -> pattern of a whole word in the file name

    @property
    def who(self) -> str:
        return f"cohort {self.name}"

    def __post_init__(self) -> None:
        super().__post_init__()
        unknown = sorted((set(self.folders.values()) | set(self.file_groups)) - set(self.order))
        if unknown:
            raise ValueError(f"cohort {self.name}: {', '.join(unknown)} named in folders or file_groups but not in "
                             "its order")

    @classmethod
    def from_config(cls, cfg: dict, name: str | None = None) -> "Cohort":
        """A Cohort from one table of cohorts.local.toml."""
        try:
            return cls(name=name, folders=dict(cfg["folders"]), order=tuple(cfg["order"]), control=cfg["control"],
                       labels=dict(cfg["labels"]), colours=dict(cfg["colours"]), animal=cfg["animal"],
                       file_groups=dict(cfg.get("file_groups", {})))
        except KeyError as e:
            raise ValueError(f"cohort {name}: the table has no {e.args[0]!r}") from None


def load_cohort(name: str, root: Path | None = None) -> Cohort:
    """The cohort called name in cohorts.local.toml (else cohorts.example.toml); see msq.core.config for where the
    file is looked for."""
    return load_groups("cohorts", Cohort, name, root)
