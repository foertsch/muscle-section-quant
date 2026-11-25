"""The R side: the testthat suites (shared plot functions, the assays' measures, the figure and export drivers)."""

import shutil
import subprocess
from pathlib import Path

import pytest

from msq.psr import fiber_size

ROOT = Path(__file__).resolve().parents[1]


def r_available() -> bool:
    if shutil.which("Rscript") is None:
        return False
    r = subprocess.run(["Rscript", "-e", "cat(all(sapply(c('testthat','ggplot2','dplyr','readr','multcomp','tidyr'), requireNamespace, quietly=TRUE)))"],
                       capture_output=True, text=True)
    return r.stdout.strip() == "TRUE"


@pytest.mark.skipif(not r_available(), reason="Rscript with testthat, ggplot2, dplyr, readr, multcomp and tidyr needed")
def test_r_suites_pass():
    r = subprocess.run(["Rscript", "-e", "testthat::test_dir('R/tests', stop_on_failure = TRUE, reporter = 'summary')"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, (r.stdout[-3000:] + r.stderr[-3000:])


@pytest.mark.skipif(not r_available(), reason="Rscript with testthat, ggplot2, dplyr, readr, multcomp and tidyr needed")
def test_the_size_plots_draw_the_measure_columns_python_writes():
    """The only constants the psr size plots share with Python are the column names of fibers.csv.gz."""
    r = subprocess.run(["Rscript", "-e", "source('R/psr/measures.R'); cat(names(SIZE_MEASURES))"],
                       cwd=ROOT, capture_output=True, text=True)
    assert tuple(r.stdout.split()) == fiber_size.MEASURES, r.stderr
