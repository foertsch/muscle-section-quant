"""Checks a folder of scans before anything runs, so problems show up as plain sentences instead of a crash an
hour into the run.

samples_from_folder: the sample table, read from folder and file names (manifest.parse), and the files left out.
check_files: every scan exists, opens, holds one image with the dystrophin and actin channels, and is a 20x scan.
check_design: the table is consistent (each section once, usable animal numbers, an animal in one group per muscle)
and every muscle has untreated controls (or a threshold borrowed from earlier).

Each problem has a level: "error" stops the run before it starts, "warning" is reported and the run goes on,
"note" is for information only.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

import pandas as pd

from msq.core import nd2
from msq.core.nd2 import channel_settings
from msq.core.problems import Problem, errors  # noqa: F401  (checks.Problem and checks.errors for the runner and app)
from msq.core.samples import differing, read_headers, read_samples, scan_files  # noqa: F401  (checks.read_samples)
from msq.dystrophin import manifest, settings

COLUMNS = ["sample", "group", "muscle", "animal", "section", "mag", "edited", "note", "path"]
NO_CONTROLS, REPORT_FAILED = "no_controls", "report_failed"  # Problem codes the runner and the app act on
OTHER_WEIGHTS = "other_weights"
MIXED_CODE, MIXED_WEIGHTS = "mixed_code", "mixed_weights"  # sections of one run made by other code or weights
ANIMAL_OK = re.compile(r"[A-Za-z0-9-]+")  # the animal is part of the sample name and of a folder name
CONTROL_LIKE = re.compile(r"ctrl|control|\bneg", re.I)
CONTROL_PREFIX_SHOWN = "Neg-ctrl"  # settings.CONTROL_FOLDER_PREFIX as people write it
IMAGING_NAMES = {"exposure": "exposure", "gain": "camera gain", "light": "light intensity (Sola iris)"}


# ---------------------------------------------------------------- sample table


def looks_like_control(folder: str) -> bool:
    """A folder name that seems meant for controls but does not start with the prefix ('Neg ctrl', 'NegCtrl', ...)."""
    return bool(CONTROL_LIKE.search(folder.replace("_", " "))) and not folder.lower().startswith(
        settings.CONTROL_FOLDER_PREFIX)


def samples_from_folder(root: Path, mag: str = settings.MAGNIFICATION,
                        suffix: str = ".nd2") -> tuple[pd.DataFrame, list[Problem]]:
    """One row per section found under root, with absolute paths. Group comes from the folders below root (a folder
    named Neg-ctrl* holds controls), muscle, section, animal and magnification from the file name. suffix: the scan
    files' ending (another reader's, see scans.py)."""
    files, problems = scan_files(root, suffix)
    rows, other_mag, folders = [], [], set()
    for p in files:
        rel = p.relative_to(root)
        try:
            r = manifest.parse(p, root)
        except ValueError:
            problems.append(Problem(
                "warning", "left out: muscle, section, animal or magnification cannot be read from the file name "
                f"(expected e.g. 'TA sec2 1001 {mag}.nd2', muscle one of {', '.join(settings.MUSCLES)})", str(rel)))
            continue
        if r["mag"] != mag:
            other_mag.append(r["mag"])
            continue
        others = [n for n in manifest.animal_numbers(p) if n != r["animal"]]
        if others:
            problems.append(Problem("warning", f"animal read as {r['animal']}, but the file name also contains "
                                               f"{', '.join(others)}; correct the animal in the table if that is wrong",
                                    str(rel)))
        folders.add(rel.parent)
        rows.append({**r, "sample": manifest.sample_id(r), "path": str(p.resolve())})
    if other_mag:
        counts = ", ".join(f"{n} at {m}" for m, n in sorted(pd.Series(other_mag).value_counts().items()))
        problems.append(Problem("note", f"left out {len(other_mag)} scans of another magnification ({counts}); only "
                                        f"{mag} scans are analysed"))
    for f in sorted({Path(*d.parts[:i + 1]) for d in folders for i, part in enumerate(d.parts)
                     if looks_like_control(part)}):
        problems.append(Problem("warning", "this folder's name looks like a control folder but does not start with "
                                           f"'{CONTROL_PREFIX_SHOWN}', so its scans are counted as treated", str(f)))
    df = pd.DataFrame(rows, columns=COLUMNS)
    # an 'Edited' file and a raw file of the same section: keep the raw one
    df = df.sort_values(["sample", "edited"], kind="stable")
    for s, g in df.groupby("sample"):
        if len(g) > 1:
            kept = Path(g["path"].iloc[0]).name
            dropped = ", ".join(Path(x).name for x in g["path"].iloc[1:])
            problems.append(Problem("warning", f"{len(g)} files describe this section; using '{kept}', "
                                               f"leaving out '{dropped}'", s))
    df = df.drop_duplicates("sample", keep="first")
    df = df.sort_values(["group", "muscle", "animal", "section"]).reset_index(drop=True)
    return df[COLUMNS], problems


# ---------------------------------------------------------------- files


def nd2_header(path: Path) -> dict:
    """Image dimensions, channel names, pixel size and the dystrophin channel's imaging settings, without reading
    the pixels."""
    h = nd2.nd2_header(path)
    return {"sizes": h.sizes, "channels": h.channels, "px_um": h.px_um,
            "gfp": channel_settings(h.description, settings.DYSTROPHIN_CHANNEL)}


def imaging_problems(seen: list[tuple[str, str, dict]]) -> list[Problem]:
    """A warning per muscle and setting when its scans were not all taken with the same dystrophin exposure, gain
    or light intensity. seen: (sample, muscle, settings) per scan."""
    problems = []
    for m in dict.fromkeys(muscle for _, muscle, _ in seen):
        for key, what in IMAGING_NAMES.items():
            values = pd.Series({s: g[key] for s, muscle, g in seen if muscle == m and key in g}, dtype=object)
            if found := differing(values):
                usual, n, other = found
                problems.append(Problem(
                    "warning", f"the {settings.DYSTROPHIN_CHANNEL} {what} differs between the scans: {usual} in "
                               f"{n}, but {other}. The threshold set on the controls only fits scans "
                               "taken with the same settings.", m))
    return problems


def check_files(samples: pd.DataFrame, read_header: Callable[[Path], dict] = nd2_header) -> list[Problem]:
    problems, seen = [], []
    lo, hi = settings.PIXEL_UM_RANGE
    for r, h in read_headers(samples, read_header, problems, must_exist=True):
        if set(h["sizes"]) != {"C", "Y", "X"}:
            problems.append(Problem("error", "expected one stitched image with channels, got dimensions "
                                             f"{h['sizes']} (time series, z-stack or several positions?)", r.sample))
        names = [n.strip().lower() for n in h["channels"]]
        for wanted, what in ((settings.DYSTROPHIN_CHANNEL, "dystrophin"), (settings.ACTIN_CHANNEL, "actin")):
            if names.count(wanted.lower()) != 1:
                problems.append(Problem("error", f"needs exactly one channel named {wanted} ({what}); the file has "
                                                 f"{', '.join(h['channels'])}", r.sample))
        if not lo <= h["px_um"] <= hi:
            problems.append(Problem("error", f"pixel size {h['px_um']:.3f} µm is not a {settings.MAGNIFICATION} scan "
                                             f"(expected {lo} to {hi} µm)", r.sample))
        seen.append((r.sample, r.muscle, h.get("gfp", {})))
    return problems + imaging_problems(seen)


def check_weights(weights: dict | None) -> list[Problem]:
    """A warning when the Cellpose weights are not the reference ones (settings.REFERENCE_WEIGHTS_SHA256).
    weights: msq.core.runinfo.weights() of this computer's file, None while it is not downloaded yet."""
    if weights is None or weights["sha256"] == settings.REFERENCE_WEIGHTS_SHA256:
        return []
    return [Problem("warning", f"the Cellpose weights ({weights['file']}, SHA-256 {weights['sha256'][:12]}…) are not "
                               f"the reference ones ({settings.REFERENCE_WEIGHTS_SHA256[:12]}…), so segmentation may "
                               "differ from earlier runs. This run is still consistent within itself, but a threshold "
                               "is not borrowed between runs with different weights.", code=OTHER_WEIGHTS)]


def listed(names: list[str], n: int = 5) -> str:
    return ", ".join(names[:n]) + (", …" if len(names) > n else "")


def check_provenance(infos: dict[str, dict], code_version: str, weights: dict | None) -> list[Problem]:
    """Warnings when the sections of a run were not all made by this code and these Cellpose weights.
    After a resume, a section measured earlier keeps the numbers of the code and weights that made it.
    infos: each finished section's run_info.json; code_version, weights: this run's (weights None when unknown).
    A section without a record of its code was measured before records were kept (dystrophin-quant v1.0.0); one
    without a record of its weights (a segmentation cached from elsewhere) is not compared."""
    problems = []
    no_record = sorted(s for s, i in infos.items() if "code_version" not in i)
    other_code = sorted(s for s, i in infos.items() if "code_version" in i and i["code_version"] != code_version)
    if other_code:
        versions = sorted({infos[s]["code_version"] for s in other_code})
        problems.append(Problem(
            "warning", f"{len(other_code)} section(s) were measured by other code ({', '.join(versions)}) than this "
                       f"run ({code_version}): {listed(other_code)}. Their numbers are those of that code; measure "
                       "them again (--redo) if the code changed what it measures.", code=MIXED_CODE))
    if no_record:
        problems.append(Problem(
            "warning", f"{len(no_record)} section(s) carry no record of the code that measured them (measured before "
                       f"records were kept): {listed(no_record)}. Measure them again (--redo) to be sure all sections "
                       "of this run come from the same code.", code=MIXED_CODE))
    ours = (weights or {}).get("sha256")
    other_weights = sorted(s for s, i in infos.items()
                           if ours and (i.get("cellpose_weights") or {}).get("sha256") not in (None, ours))
    if other_weights:
        problems.append(Problem(
            "warning", f"{len(other_weights)} section(s) were segmented with other Cellpose weights than this "
                       f"computer's ({ours[:12]}…): {listed(other_weights)}. Their outlines may differ from sections "
                       "segmented now; measure them again (--redo) to segment every section with the same weights.",
            code=MIXED_WEIGHTS))
    return problems


# ---------------------------------------------------------------- design


def check_design(samples: pd.DataFrame, borrowed: dict[str, dict] | None = None) -> list[Problem]:
    """borrowed: {muscle: {"threshold": float, "source": str, "checked": bool}} for muscles whose threshold may come
    from an earlier run (run_pipeline.load_reference)."""
    borrowed = borrowed or {}
    problems = []
    if samples.empty:
        return [Problem("error", "the sample table is empty")]
    section = samples[["muscle", "section", "animal"]].fillna("").astype(str).agg(" ".join, axis=1)
    twice = section.duplicated(keep=False)
    for _, g in samples[twice].groupby(section[twice], sort=False):  # also when the copies differ in group
        problems.append(Problem("error", "the same section (muscle, section and animal) is in the table more than "
                                         "once; leave out all but one", ", ".join(dict.fromkeys(g["sample"]))))
    for r in samples.itertuples():
        if r.group not in settings.GROUPS.order:
            problems.append(Problem("error", f"group '{r.group}' must be '{manifest.CONTROL}' or "
                                             f"'{manifest.TREATED}'", r.sample))
        if r.muscle not in settings.MUSCLES:
            problems.append(Problem("error", f"muscle '{r.muscle}' must be one of {', '.join(settings.MUSCLES)}",
                                    r.sample))
        if pd.isna(r.animal) or not ANIMAL_OK.fullmatch(str(r.animal)):
            shown = "" if pd.isna(r.animal) else r.animal
            problems.append(Problem("error", f"animal '{shown}' must be letters, digits or '-' (no spaces, '_' or "
                                             "'/'), because it becomes part of the sample and folder names", r.sample))
    for m, g in samples.groupby("muscle"):
        groups = g[g["group"].isin(settings.GROUPS.order)].groupby("animal")["group"].nunique()
        for a in groups.index[groups > 1]:
            problems.append(Problem(
                "error", f"animal {a} is both control and treated. Correct the group in the table; an animal is one "
                         "unit of comparison, so a control leg and a treated leg of one animal are not supported yet.",
                m))
        ctrl_animals = sorted(g.loc[g["group"] == manifest.CONTROL, "animal"].dropna().astype(str).unique())
        if not ctrl_animals:
            if m in borrowed:
                unchecked = "" if borrowed[m].get("checked", True) else (
                    " Its measurement settings or Cellpose weights could not be compared with this run's.")
                problems.append(Problem(
                    "warning", f"no untreated controls in this run; the threshold ({borrowed[m]['threshold']:.0f}) is "
                               f"borrowed from {borrowed[m]['source']}. That is only valid if staining and imaging "
                               f"were identical (antibody lot, exposure, laser power, gain).{unchecked}", m))
            else:
                problems.append(Problem(
                    "error", "no untreated control sections. Add control scans (in a folder whose name starts with "
                             "'Neg-ctrl') or borrow the threshold of an earlier run.", m, code=NO_CONTROLS))
        elif len(ctrl_animals) < settings.MIN_CONTROL_ANIMALS:
            problems.append(Problem(
                "warning", f"only {len(ctrl_animals)} control animal ({', '.join(ctrl_animals)}); the threshold rests "
                           "on it alone and no held-out check is possible.", m))
        if not (g["group"] == manifest.TREATED).any():
            problems.append(Problem("note", "controls only, no treated sections", m))
    return problems
