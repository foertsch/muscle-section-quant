"""Folder layouts, so every command agrees on where raw data, intermediate output and results live.

The code lives in this package; the data and results live elsewhere, in one of two layouts.

A project folder (the step-by-step commands' defaults point here). It is the folder named by the environment
variable DYSTRO_WORKSPACE, or else the current directory, read when a command starts (project()):

    data/                     raw ND2 scans
    manifest_20x.csv          sample table (manifest.py)
    output/                   per-section intermediates: fibers.csv, sectors.csv.gz, labels, maps, logs
    deliverables/             everything to look at or share: reports/, figures/, tables/

A run folder, written by run_pipeline.py: one self-contained folder per run (RunFolder, from msq.core.io).
"""

import os
from dataclasses import dataclass
from pathlib import Path

from msq.core.io import RunFolder  # noqa: F401  (the run folder layout, as paths.RunFolder)

ENV_VAR = "DYSTRO_WORKSPACE"


@dataclass(frozen=True)
class ProjectFolder:
    """The project-folder layout, which the step-by-step commands and the scripts read and write."""

    root: Path

    @property
    def manifest(self) -> Path:
        return self.root / "manifest_20x.csv"

    @property
    def output(self) -> Path:
        return self.root / "output"

    @property
    def deliverables(self) -> Path:
        return self.root / "deliverables"

    @property
    def reports(self) -> Path:
        return self.deliverables / "reports"

    @property
    def figures(self) -> Path:
        return self.deliverables / "figures"

    @property
    def figures_method(self) -> Path:
        return self.figures / "method"

    @property
    def figures_results(self) -> Path:
        return self.figures / "results"

    @property
    def tables(self) -> Path:
        return self.deliverables / "tables"


def project() -> ProjectFolder:
    """The project folder: DYSTRO_WORKSPACE if set, else the current directory."""
    return ProjectFolder(Path(os.environ.get(ENV_VAR) or Path.cwd()))
