"""Checks a folder of scans before anything runs, so problems show up as plain sentences instead of a crash an
hour into the run, and writes the sample table with each scan's imaging settings.

samples_from_folder: the sample table, read from folder and file names (manifest.parse), and the files left out.
check_files: every scan opens and is one 8-bit colour brightfield image at the 20x pixel size; returns the imaging
settings per scan (exposure, gain, white balance, light, sharpening, date).
imaging_problems: which settings differ between scans, and on which days each group was imaged.
check_design: each section once, known groups, controls present, enough animals per group.

Each problem has a level: "error" stops the run before it starts, "warning" is reported and the run goes on,
"note" is for information only.

check_scans puts them together; the command line is `python -m msq.psr.run_pipeline SCANS RUN --cohort NAME
--check-only`, which writes samples.csv and checks.txt.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

import pandas as pd

from msq.core import nd2
from msq.core.problems import Problem, errors  # noqa: F401  (checks.Problem and checks.errors)
from msq.core.samples import check_groups, differing, imaging_days, read_headers, read_samples, scan_files  # noqa: F401
from msq.psr import manifest, settings
from msq.psr.settings import Cohort

COLUMNS = ["sample", "group", "muscle", "animal", "section", "path"]
IMAGING_FIELDS = {  # column: (label in the ND2 description, unit to strip, type)
    "exposure_ms": ("Exposure", "ms", float),
    "gain": ("Gain", "x", float),
    "wb_red": ("WB Red", "", float),
    "wb_blue": ("WB Blue", "", float),
    "light": (r"Illuminator\(DIA\) Iris intensity", "", float),
    "sharpness": ("Sharpness", "", str),
}
IMAGING_SHOWN = {"exposure_ms": ("exposure", " ms"), "gain": ("camera gain", "x"), "wb_red": ("white balance red", ""),
                 "wb_blue": ("white balance blue", ""), "light": ("transmitted light intensity", ""),
                 "sharpness": ("in-camera sharpening", "")}
MIN_ANIMALS = 3


# ---------------------------------------------------------------- sample table


def samples_from_folder(root: Path, cohort: Cohort) -> tuple[pd.DataFrame, list[Problem]]:
    """One row per whole-section scan under root, with absolute paths, control group first."""
    files, problems = scan_files(root)
    rows, n_fields = [], 0
    for p in files:
        rel = p.relative_to(root)
        try:
            r = manifest.parse(p, root, cohort)
        except ValueError as e:
            problems.append(Problem("warning", f"left out: {e}", str(rel)))
            continue
        if not r["full"]:
            n_fields += 1
            continue
        if r["folder_animal"] and r["folder_animal"] != r["animal"]:
            problems.append(Problem("warning", f"the file name says animal {r['animal']} but its folder says "
                                               f"{r['folder_animal']}; the file name is used", str(rel)))
        if r["file_group"] and r["file_group"] != r["group"]:
            problems.append(Problem("warning", f"the file name says group {r['file_group']} but it sits in the "
                                               f"{r['group']} folder; the file name is used", str(rel)))
            r["group"] = r["file_group"]
        rows.append({**r, "sample": manifest.sample_id(r), "path": str(p.resolve())})
    if n_fields:
        problems.append(Problem("note", f"left out {n_fields} single fields (no '{settings.FULL_SCAN_MARK}' in the "
                                        "name); only whole-section scans are analysed"))
    df = pd.DataFrame(rows, columns=COLUMNS)
    df = df.sort_values(["group", "animal", "section"], key=cohort.sort_key(), kind="stable").reset_index(drop=True)
    return df, problems


# ---------------------------------------------------------------- files


def imaging_settings(description: str, date: str) -> dict:
    """Camera and light settings as NIS Elements writes them in an ND2's text description, plus the acquisition
    time ('01.01.2026  12:00:00' -> '2026-01-01 12:00'). Settings it does not find are left out."""
    out = {}
    for key, (label, unit, kind) in IMAGING_FIELDS.items():
        m = re.search(rf"^\s*(?:[^\n:]*, )?{label}:\s*(.+?)\s*$", description, re.M)
        if m:
            value = m.group(1).removesuffix(unit).strip() if unit else m.group(1).strip()
            out[key] = kind(value)
    when = nd2.acquired(date, "%Y-%m-%d %H:%M")
    if when:
        out["acquired"] = when
    return out


def nd2_header(path: Path) -> dict:
    """Dimensions, channel names, pixel size, bit depth and imaging settings, without reading the pixels."""
    h = nd2.nd2_header(path)
    return {"sizes": h.sizes, "channels": h.channels, "px_um": h.px_um, "bits": h.bits,
            "imaging": imaging_settings(h.description, h.date)}


def check_files(samples: pd.DataFrame, read_header: Callable[[Path], dict] | None = None
                ) -> tuple[list[Problem], pd.DataFrame]:
    """Problems per scan, and the imaging settings of every scan that opens (index: sample)."""
    read_header = read_header or nd2_header
    lo, hi = settings.PIXEL_UM_RANGE
    problems, imaging = [], {}
    for r, h in read_headers(samples, read_header, problems):
        if set(h["sizes"]) != {"Y", "X", "S"} or h["sizes"].get("S") != 3:
            problems.append(Problem("error", "expected one stitched colour (RGB) image, got dimensions "
                                             f"{h['sizes']}", r.sample))
        if [c.strip().lower() for c in h["channels"]] != [settings.CHANNEL.lower()]:
            problems.append(Problem("error", f"expected one channel named {settings.CHANNEL}; the file has "
                                             f"{', '.join(h['channels'])}", r.sample))
        if not lo <= h["px_um"] <= hi:
            problems.append(Problem("error", f"pixel size {h['px_um']:.4f} µm is not a 20x full scan (expected "
                                             f"{lo} to {hi} µm); is this a single field named 'full'?", r.sample))
        if h["bits"] != 8:
            problems.append(Problem("error", f"expected 8-bit colour, got {h['bits']} bits", r.sample))
        imaging[r.sample] = {**h["imaging"], "px_um": h["px_um"], "height": h["sizes"].get("Y"),
                             "width": h["sizes"].get("X")}
    return problems, pd.DataFrame.from_dict(imaging, orient="index")


def shown(key: str, value) -> str:
    unit = IMAGING_SHOWN[key][1]
    return f"{value:g}{unit}" if isinstance(value, float) else f"{value}{unit}"


def imaging_problems(samples: pd.DataFrame, imaging: pd.DataFrame, cohort: Cohort) -> list[Problem]:
    """A note per setting that differs between scans, the imaging days per group, and a warning for a group that
    was never imaged on the same day as the controls (then day and group cannot be told apart)."""
    problems = []
    for key, (what, _) in IMAGING_SHOWN.items():
        if key not in imaging:
            continue
        if found := differing(imaging[key].dropna(), lambda v: shown(key, v)):
            usual, n, other = found
            problems.append(Problem("note", f"{what} differs between scans: {usual} in {n}, but {other}"))
    if "acquired" not in imaging:
        return problems
    day = imaging["acquired"].str[:10]
    group = samples.set_index("sample")["group"].reindex(day.index)
    return problems + imaging_days(day, group, cohort.control)


# ---------------------------------------------------------------- design


def check_design(samples: pd.DataFrame, cohort: Cohort) -> list[Problem]:
    if samples.empty:
        return [Problem("error", "the sample table is empty")]
    problems = []
    twice = samples["sample"].duplicated(keep=False)
    for s in dict.fromkeys(samples.loc[twice, "sample"]):
        files = ", ".join(Path(p).name for p in samples.loc[samples["sample"] == s, "path"])
        problems.append(Problem("error", f"the same section is in the table more than once ({files}); leave out "
                                         "all but one", s))
    problems += check_groups(samples, cohort)
    if not (samples["group"] == cohort.control).any():
        problems.append(Problem("error", f"no control sections (group {cohort.control})"))
    for g, d in samples.groupby("group", sort=False):
        n = d["animal"].nunique()
        if n < MIN_ANIMALS:
            problems.append(Problem("warning", f"only {n} animal{'s' if n != 1 else ''}; the group comparison needs "
                                               f"at least {MIN_ANIMALS}", g))
        per_animal = d.groupby("animal")["section"].nunique()
        few = per_animal[per_animal < per_animal.max()]
        if len(few):
            problems.append(Problem("note", "fewer sections than the others: " + ", ".join(
                f"{a} ({n})" for a, n in few.items()), g))
    return problems


# ---------------------------------------------------------------- put together


def check_scans(scans: Path, cohort: Cohort) -> tuple[pd.DataFrame, list[Problem]]:
    """The sample table of a scan folder with each scan's imaging settings, and every problem the checks found."""
    samples, problems = samples_from_folder(scans, cohort)
    file_problems, imaging = check_files(samples, read_header=nd2_header)
    problems += file_problems + check_design(samples, cohort)
    if not imaging.empty:
        problems += imaging_problems(samples, imaging, cohort)
    return samples.join(imaging, on="sample"), problems
