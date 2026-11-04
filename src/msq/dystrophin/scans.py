"""Where the runner reads scans from: Nikon ND2 files by default, or any reader with the same three members.

A reader has `suffix` (the file ending the sample table looks for), `read(path)` (the full-resolution dystrophin and
actin planes, pixel size, saturation level and channel names, as measure.read_nd2 returns them) and `header(path)`
(dimensions, channel names, pixel size and the dystrophin channel's imaging settings, as checks.nd2_header returns
them). The runner, the measurement and both reports take one; without one they read ND2 files.

NpzScans reads synthetic sections saved with save_npz: the demo and the tests run the whole pipeline on them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from msq.dystrophin import checks
from msq.dystrophin import measure


class ND2Scans:
    """Nikon ND2 files: what the runner reads when it is given no reader."""

    suffix = ".nd2"

    def read(self, path: Path) -> dict:
        return measure.read_nd2(path)

    def header(self, path: Path) -> dict:
        return checks.nd2_header(path)


IMAGING = {"exposure": "2 s", "gain": "1.0x", "light": "10.0"}  # one setting for every synthetic scan


def save_npz(path: Path, section: dict) -> Path:
    """A synthetic section (msq.core.synthetic.dystrophin.synthetic_section) as one compressed file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, gfp=section["gfp"], act=section["act"], px_um=section["px_um"], sat=section["sat"],
                        channels=np.array(section["channels"]))
    return path


class NpzScans:
    """Synthetic sections saved by save_npz, read as if they were ND2 scans."""

    suffix = ".npz"

    def read(self, path: Path) -> dict:
        with np.load(path) as z:
            return {"gfp": z["gfp"], "act": z["act"], "px_um": float(z["px_um"]), "sat": int(z["sat"]),
                    "channels": [str(c) for c in z["channels"]]}

    def header(self, path: Path) -> dict:
        with np.load(path) as z:
            gfp, channels, px_um = z["gfp"].shape, [str(c) for c in z["channels"]], float(z["px_um"])
        return {"sizes": {"C": len(channels), "Y": gfp[0], "X": gfp[1]}, "channels": channels, "px_um": px_um,
                "gfp": dict(IMAGING)}
