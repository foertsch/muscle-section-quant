"""The console scripts in pyproject.toml point at functions that exist."""

import importlib
import tomllib
from pathlib import Path

import pytest

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"
SCRIPTS = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]["scripts"]


def test_msq_psr_is_a_script_on_run_pipeline_main():
    assert SCRIPTS["msq-psr"] == "msq.psr.run_pipeline:main"


@pytest.mark.parametrize("name", sorted(SCRIPTS))
def test_every_script_target_imports_and_is_callable(name):
    module, func = SCRIPTS[name].split(":")
    assert callable(getattr(importlib.import_module(module), func))
