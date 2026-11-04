"""What every run records about itself, so a number can be traced to the code, the packages and the weights that
made it: the code version (a git tag or commit), the versions of the packages that decide the numbers, the size and
SHA-256 of the Cellpose weights, and the machine's peak memory.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path


PACKAGE = "muscle-section-quant"
PACKAGES = ("cellpose", "torch", "numpy", "scipy", "scikit-image", "pandas", "nd2")


def repo_root() -> Path | None:
    """The repository this package is installed from in development, or None when installed from a wheel."""
    root = Path(__file__).resolve().parents[3]
    return root if (root / ".git").exists() else None


def code_version() -> str:
    """A git tag or commit of the development checkout ('v1.0.0', 'v1.0.0-3-g1a2b3c4-dirty'), else the installed
    package version, else 'unknown'."""
    root = repo_root()
    if root is not None:
        try:
            r = subprocess.run(["git", "describe", "--tags", "--always", "--dirty"], cwd=root, capture_output=True,
                               text=True, timeout=10)
            if r.stdout.strip():
                return r.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    try:
        return f"{PACKAGE} {metadata.version(PACKAGE)}"
    except metadata.PackageNotFoundError:
        return "unknown"


def packages(names: tuple[str, ...] = PACKAGES, missing_ok: bool = False) -> dict[str, str | None]:
    """Python and the versions of the packages that decide the numbers (None for a missing one when missing_ok)."""
    out: dict[str, str | None] = {"python": platform.python_version()}
    for p in names:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            if not missing_ok:
                raise
            out[p] = None
    return out


def weights(path: Path) -> dict | None:
    """Name, size and SHA-256 of a weights file (a few seconds for 1.2 GB), or None when it is not there yet."""
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            digest.update(chunk)
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def platform_key() -> str:
    return f"{sys.platform}-{platform.machine().lower()}"


def peak_memory_gb() -> float:
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e9 if sys.platform == "darwin" else peak / 1e6  # bytes on macOS, kB on Linux
