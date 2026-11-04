"""One section, start to finish: the part of a per-section run that the assays share.

An assay supplies what is its own (read and downsample the scan, find the tissue, segment, measure) as three methods;
run_section does the rest: the section's folder is written as <out>.partial and renamed when complete (an interrupted
run leaves the .partial), each variant is segmented with its own segmenter, and the section's record is stamped with
the run time, the peak memory and where the numbers came from (code version, Cellpose weights).

    prepare(scan, variants)                 the grid images, masks and section-wide record of one scan -> Prepared
    params(variant)                         the settings of a variant (they decide the segmenter's thresholds)
    run_variant(prepared, variant, segment) one variant segmented and measured -> Outputs

Layout says where the files go: psr writes one variant into the section folder (section.json); necrosis writes a
folder per variant and the section's own record beside them (prep.json). Dystrophin keeps its own driver, because it
resumes from cached labels.
"""

from __future__ import annotations

import functools
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import tifffile

from msq.core import runinfo
from msq.core.io import partial_folder
from msq.core.segment import CellposeSegmenter, Segmenter, segmenter_for, weights_file

SECTION_JSON = "section.json"  # the record of one variant


@dataclass
class Prepared:
    """What a scan gives before any variant runs."""

    state: Any                                           # the assay's own grid images and masks, for run_variant
    files: dict[str, Any] = field(default_factory=dict)  # written once into the section folder (see SectionFolder)
    record: dict = field(default_factory=dict)           # the section's own record (Layout.shared_json)


@dataclass
class Outputs:
    """What one variant makes."""

    record: dict                                         # its section.json
    files: dict[str, Any] = field(default_factory=dict)  # written into the variant's folder (see SectionFolder)


@dataclass(frozen=True)
class Layout:
    """How an assay lays out a section folder and spells its records.

    shared_json: None = one variant, its files and section.json in the section folder itself (psr); a name = the
        section's own record is written under that name and every variant gets a folder <section>/<variant>/ (necrosis).
    seconds_in: the dict of the stamped record that also holds the run time as "total" (psr: "seconds"); None = the
        record gets "seconds_total".
    The stamped record is the section's own (shared_json), else the one variant's."""

    shared_json: str | None = None
    seconds_in: str | None = None


class SectionFolder:
    """A folder being written (the .partial one): files go in by their suffix, a variant gets a folder of its own."""

    def __init__(self, path: Path):
        self.path = path

    def variant(self, name: str) -> SectionFolder:
        (self.path / name).mkdir()
        return SectionFolder(self.path / name)

    def write(self, files: dict[str, Any]) -> None:
        """name -> data: a DataFrame for .csv (no index), an array or (array, tifffile options) for .tif (zlib), a
        dict of arrays for .npz."""
        for name, data in files.items():
            target = self.path / name
            if name.endswith(".csv"):
                data.to_csv(target, index=False)
            elif name.endswith(".npz"):
                np.savez_compressed(target, **data)
            elif name.endswith(".tif"):
                img, options = data if isinstance(data, tuple) else (data, {})
                tifffile.imwrite(target, img, compression="zlib", **options)
            else:
                raise ValueError(f"no writer for {name!r}")

    def write_json(self, name: str, record: dict) -> None:
        (self.path / name).write_text(json.dumps(record, indent=1))


# ---------------------------------------------------------------- where the numbers came from

_code_version = functools.cache(runinfo.code_version)  # one git call per process
_WEIGHTS: dict[tuple, dict | None] = {}                 # (file, size, mtime) -> its record; hashing takes seconds


def cellpose_weights(segmenters: Sequence[Segmenter]) -> dict | None:
    """Name, size and SHA-256 of the Cellpose weights if one of the segmenters is Cellpose, else None: with cached
    labels or a stand-in the weights played no part. Hashed once per process and file."""
    models = [s.model for s in segmenters if isinstance(s, CellposeSegmenter)]
    if not models:
        return None
    path = weights_file(models[0])
    try:
        stat = path.stat()
    except OSError:
        return None  # not downloaded
    key = (path, stat.st_size, stat.st_mtime_ns)
    if key not in _WEIGHTS:
        _WEIGHTS[key] = runinfo.weights(path)
    return _WEIGHTS[key]


def stamp(record: dict, layout: Layout, elapsed_s: float, segmenters: Sequence[Segmenter]) -> None:
    """Run time, peak memory, code version and Cellpose weights into the section's record."""
    if layout.seconds_in:
        record[layout.seconds_in]["total"] = round(elapsed_s, 1)
    else:
        record["seconds_total"] = round(elapsed_s, 1)
    record["peak_memory_gb"] = round(runinfo.peak_memory_gb(), 2)
    record["code_version"] = _code_version()
    record["cellpose_weights"] = cellpose_weights(segmenters)


# ---------------------------------------------------------------- the driver


def run_section(assay: Any, scan: Any, out: Path, variants: Sequence[str],
                segment: Segmenter | Mapping[str, Segmenter] | None, layout: Layout) -> Path:
    """Process one scan with each variant (assay: the three methods of the module docstring) and write the section
    folder out. segment: see segmenter_for (Cellpose by default, one segmenter for all, or one per variant)."""
    if layout.shared_json is None and len(variants) != 1:
        raise ValueError(f"this layout holds one variant per section folder, got {list(variants)}")
    t0 = time.time()
    prepared = assay.prepare(scan, variants)
    used, records = [], []
    with partial_folder(out) as tmp:
        root = SectionFolder(tmp)
        root.write(prepared.files)
        for variant in variants:
            segmenter = segmenter_for(segment, variant, assay.params(variant))
            outputs = assay.run_variant(prepared, variant, segmenter)
            folder = root if layout.shared_json is None else root.variant(variant)
            folder.write(outputs.files)
            used.append(segmenter)
            records.append((folder, outputs.record))
        stamp(records[0][1] if layout.shared_json is None else prepared.record, layout, time.time() - t0, used)
        for folder, record in records:
            folder.write_json(SECTION_JSON, record)
        if layout.shared_json is not None:
            root.write_json(layout.shared_json, prepared.record)
    return out
