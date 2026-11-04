"""Reading Nikon ND2 scans: the header, the channel order by name, the imaging settings in the text description, and
a memory-mapped frame to read in strips.

The channel order differs between files, so planes are always found by name (channel_index). Scans are opened once
and read in strips (images.py) because a whole section is several GB at full resolution.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

# setting -> its label in a 'Plane #n:' block of the description, as NIS Elements writes them
IMAGING_LABELS = {"exposure": "Exposure", "gain": "Gain", "light": r"Sola II, Illuminator\(Sola\) Iris intensity"}


def channel_index(names: list[str], wanted: str) -> int:
    """Index of a channel by name; the ND2 channel order differs between files."""
    hits = [i for i, n in enumerate(names) if n.strip().lower() == wanted.lower()]
    if len(hits) != 1:
        raise ValueError(f"expected exactly one channel named {wanted!r}, got {names}")
    return hits[0]


def channel_settings(description: str, channel: str, labels: dict[str, str] = IMAGING_LABELS) -> dict[str, str]:
    """The settings of one channel as NIS Elements writes them in an ND2's text description (one 'Plane #n:' block
    per channel), as the raw strings ('2 s', '1.0x', '10.0'). Settings it does not find are left out."""
    for block in re.split(r"^\s*Plane #\d+:", description, flags=re.M)[1:]:
        name = re.search(r"^\s*Name:\s*(.+)$", block, re.M)
        if name and name.group(1).strip().lower() == channel.lower():
            found = {key: re.search(rf"^\s*{label}:\s*(.+)$", block, re.M) for key, label in labels.items()}
            return {key: m.group(1).strip() for key, m in found.items() if m}
    return {}


def to_ms(value: str) -> float:
    """'500 ms' -> 500.0, '2 s' -> 2000.0, '1.5 min' -> 90000.0 (NIS Elements switches units)."""
    m = re.fullmatch(r"\s*([\d.]+)\s*(ms|s|min)?\s*", value)
    if m is None:
        raise ValueError(f"cannot read the exposure '{value}'")
    return float(m.group(1)) * {"ms": 1, "s": 1000, "min": 60000, None: 1}[m.group(2)]


def acquired(date: str, fmt: str = "%Y-%m-%d %H:%M:%S") -> str | None:
    """The acquisition time as NIS Elements writes it ('23.07.2026  13:46:15') in ISO order; None when empty."""
    if not date.strip():
        return None
    return datetime.strptime(" ".join(date.split()), "%d.%m.%Y %H:%M:%S").strftime(fmt)


@dataclass
class Header:
    sizes: dict
    channels: list[str]
    px_um: float
    bits: int
    description: str
    date: str


def nd2_header(path: Path) -> Header:
    """Dimensions, channel names, pixel size, bit depth and the text description, without reading the pixels."""
    import nd2

    with nd2.ND2File(path) as f:
        info = f.text_info or {}
        return Header(dict(f.sizes), [c.channel.name for c in f.metadata.channels], float(f.voxel_size().x),
                      f.attributes.bitsPerComponentSignificant, info.get("description", ""), info.get("date", ""))


@dataclass
class Scan:
    """An open scan: the reader, its memory-mapped frame in the file's own order, and the header fields."""

    reader: object
    frame: np.ndarray
    px_um: float
    channels: list[str]
    sizes: dict
    bits: int
    description: str
    date: str

    def close(self) -> None:
        self.reader.close()

    def __enter__(self) -> "Scan":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def open_scan(path: Path, require: set[str] | None = None) -> Scan:
    """Open a scan for strip-wise reading. require: the dimension names the file must have, e.g. {"C", "Y", "X"}."""
    import nd2

    f = nd2.ND2File(path)
    try:
        sizes = dict(f.sizes)
        if require is not None and set(sizes) != require:
            raise ValueError(f"{Path(path).name}: expected dimensions {sorted(require)}, got {sizes}")
        names = [c.channel.name for c in f.metadata.channels]
        info = f.text_info or {}
        frame = f.read_frame(0)
        return Scan(f, frame, float(f.voxel_size().x), names, sizes, f.attributes.bitsPerComponentSignificant,
                    info.get("description", ""), info.get("date", ""))
    except Exception:
        f.close()
        raise
