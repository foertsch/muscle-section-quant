"""Local configuration that never enters the repository: cohorts, groups, cluster settings.

Each configuration <name> ships as <name>.example.toml (tracked, generic) and is used as <name>.local.toml
(untracked, the lab's). local_config() reads the local file from the config folder (MSQ_CONFIG_DIR, else the
repository root, else the working directory) and falls back to the example, so tests and the demo run with generic
groups and the lab runs with its own. Nothing is read at import time: callers ask for what they need and pass it on.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

ENV_VAR = "MSQ_CONFIG_DIR"


def config_dirs(root: Path | None = None) -> list[Path]:
    """Where configuration files are looked for, in order."""
    dirs = []
    if os.environ.get(ENV_VAR):
        dirs.append(Path(os.environ[ENV_VAR]))
    if root is not None:
        dirs.append(root)
    from msq.core.runinfo import repo_root

    if repo_root() is not None:
        dirs.append(repo_root())
    dirs.append(Path.cwd())
    return list(dict.fromkeys(dirs))


def find_config(name: str, root: Path | None = None) -> tuple[Path, bool]:
    """(path, is_local) of <name>.local.toml, else <name>.example.toml, searching config_dirs()."""
    for d in config_dirs(root):
        if (d / f"{name}.local.toml").is_file():
            return d / f"{name}.local.toml", True
    for d in config_dirs(root):
        if (d / f"{name}.example.toml").is_file():
            return d / f"{name}.example.toml", False
    raise FileNotFoundError(f"no {name}.local.toml or {name}.example.toml in {', '.join(map(str, config_dirs(root)))}")


def local_config(name: str, root: Path | None = None) -> dict:
    """The parsed configuration, with '_source' (the file) and '_local' (whether it is the lab's own file) added."""
    path, is_local = find_config(name, root)
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    data["_source"], data["_local"] = str(path), is_local
    return data
