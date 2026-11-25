"""Fixtures shared by the psr tests: the generic cohorts of cohorts.example.toml, whatever cohorts.local.toml exists."""

import tomllib
from pathlib import Path

import pytest

from msq.core import groups
from msq.psr import settings

EXAMPLE = Path(__file__).resolve().parents[2] / "cohorts.example.toml"


def example_config() -> dict:
    """cohorts.example.toml as local_config() returns it, so a lab's cohorts.local.toml never reaches a test."""
    data = tomllib.loads(EXAMPLE.read_text(encoding="utf-8"))
    data["_source"], data["_local"] = str(EXAMPLE), False
    return data


@pytest.fixture
def example_cohorts(monkeypatch):
    """load_cohort() reads the example file only (for the command-line mains)."""
    monkeypatch.setattr(groups, "local_config", lambda name, root=None: example_config())


@pytest.fixture(scope="session")
def example() -> settings.Cohort:
    """Numbered group folders and two-part animal numbers: <scan folder>/1001 control/1001-101-quad/...nd2."""
    return settings.Cohort.from_config(example_config()["example"], "example")


@pytest.fixture(scope="session")
def example_named() -> settings.Cohort:
    """One folder per group, animals numbered alone, the group also named in the file name."""
    return settings.Cohort.from_config(example_config()["example_named"], "example_named")
