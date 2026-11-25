"""Read group, muscle, animal and section from a scan's folders and file name.

Layout of the scan folder (cohorts.example.toml, "example"):

    <scan folder>/1001 group a/1001-101-quad/2-1001-101-20x-full.nd2
                  group folder    animal folder  section 2 (no prefix = section 1), animal 1001-101, whole section

A cohort that numbers animals without the cohort prefix and names the group in the file (cohorts.example.toml,
"example_named"):

    <scan folder>/Control/901 ctl QUAD/2-901-ctl-quad-20full.nd2

Group comes from the top folder below the scan folder (the cohort's folders, matched as a whole after dropping a
leading number), animal and section from the file name, muscle from the animal folder or the file name. When a file
name also names a group (the cohort's file_groups), that is returned as file_group so the checks can catch scans filed
in the wrong folder. Whole-section scans have "full" in the name; the others are
single fields at 10x, 20x or 40x (e.g. 3-1003-101-10.1.nd2) and are left out of the analysis.

Sample names follow <MUSCLE>_sec<n>_<animal>_<group>, e.g. QUAD_sec2_1002-201_group_a.
"""

from __future__ import annotations

import re
from pathlib import Path

from msq.psr import settings
from msq.psr.settings import Cohort

SECTION_PREFIX = re.compile(r"^([1-9])[-_](?=\d)")    # 2-1001-101-..., 2_901_ctl_...
LEADING_NUMBER = re.compile(r"^\d+[\s_-]*")             # "1001 group a" -> "group a"
MUSCLE = re.compile(rf"(?<![A-Za-z])({'|'.join(settings.MUSCLES)})(?![A-Za-z])", re.I)


def animal_in(name: str, cohort: Cohort) -> str | None:
    m = re.search(cohort.animal, name)
    return "-".join(m.groups()) if m else None


def group_of(path: Path, root: Path, cohort: Cohort) -> str:
    top = path.relative_to(root).parts[0]
    group = cohort.folders.get(LEADING_NUMBER.sub("", top).strip().lower())
    if group is None:
        raise ValueError(f"cannot read the group from folder '{top}' (cohort {cohort.name}: expected one of "
                         f"{', '.join(cohort.folders)})")
    return group


def group_in_name(stem: str, cohort: Cohort) -> str | None:
    """The group a file name names, if exactly one (e.g. 902-GRA-quad-20full -> group_a)."""
    words = re.split(r"[-_\s.]+", stem.lower())
    hits = [g for g, pat in cohort.file_groups.items() if any(re.fullmatch(pat, w) for w in words)]
    return hits[0] if len(hits) == 1 else None


def parse(path: Path, root: Path, cohort: Cohort) -> dict:
    """Group, muscle, animal and section of one ND2 path below root, and whether it is a whole-section scan.
    folder_animal is the animal its folder is named after (None when the folder has no animal number), file_group
    the group its file name names (None when it names none)."""
    if path.parent == root:
        raise ValueError("scan sits directly in the scan folder, so it has no group folder")
    group = group_of(path, root, cohort)
    stem = path.stem
    animal = animal_in(stem, cohort)
    if animal is None:
        raise ValueError(f"cannot read the animal from the file name (cohort {cohort.name})")
    muscle = MUSCLE.search(path.parent.name) or MUSCLE.search(stem)
    if muscle is None:
        raise ValueError(f"cannot read the muscle ({', '.join(settings.MUSCLES)}) from the folder or file name")
    prefix = SECTION_PREFIX.match(stem)
    return {
        "group": group,
        "muscle": muscle.group(1).upper(),
        "animal": animal,
        "section": int(prefix.group(1)) if prefix else 1,
        "full": settings.FULL_SCAN_MARK in stem.lower(),
        "folder_animal": animal_in(path.parent.name, cohort),
        "file_group": group_in_name(stem, cohort),
    }


def sample_id(r: dict) -> str:
    return f"{r['muscle']}_sec{r['section']}_{r['animal']}_{r['group']}"
