"""Read group, muscle, section, animal and magnification from a scan's folder and file name, and write a project
folder's sample table (manifest_20x.csv).

Sample names follow <MUSCLE>_sec<n>_<animal>_<mag>_<group>, e.g. TA_sec2_1001_20x_treated. The step-by-step QC
reads muscle, section, animal and magnification back from these name parts; the runner and the reports use the
table's columns. Group comes from the folders below the scan folder (Neg-ctrl* =
ctrl, else treated); animal, muscle and section come from the file name, because some control folders are named
after a different ID than the files inside (a folder named after one animal can hold another's files). The table itself is built by
checks.samples_from_folder, for the runner and for this script alike.

Usage:
    python -m msq.dystrophin.manifest SCANS --mag 20x --out manifest_20x.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from msq.dystrophin import paths
from msq.dystrophin import settings

MUSCLE = re.compile(rf"\b({'|'.join(settings.MUSCLES)})\b", re.I)
SECTION = re.compile(r"\bsec\s*(\d+)", re.I)
MAG = re.compile(r"(\d+)x", re.I)
ANIMAL = re.compile(settings.GROUPS.animal)
CONTROL, TREATED = settings.GROUPS.order
MANIFEST_COLUMNS = ["path", "group", "muscle", "animal", "section", "mag", "edited", "note", "sample"]


def words(path: Path) -> str:
    """The file name as the patterns read it: '_' counts as a space, so 'TA_sec2_1001_20x.nd2' is read too."""
    return path.stem.replace("_", " ")


def animal_numbers(path: Path) -> list[str]:
    """Every distinct animal-like number (1000 to 1999) in the file name, in order."""
    return list(dict.fromkeys(ANIMAL.findall(words(path))))


def is_control(path: Path, root: Path | None = None) -> bool:
    """A scan is a control when a folder it sits in starts with the control prefix. Only the folders below root
    count, so a scan folder that itself sits inside, say, 'Neg-ctrl comparison/' does not make everything a control."""
    folders = (path.relative_to(root) if root else path).parent.parts
    return any(part.lower().startswith(settings.CONTROL_FOLDER_PREFIX) for part in folders)


def parse(path: Path, root: Path | None = None) -> dict:
    """Group, muscle, animal, section and magnification of one ND2 path. root: the scan folder (see is_control)."""
    name = words(path)
    muscle = MUSCLE.search(name)
    section = SECTION.search(name)
    mag = MAG.search(name.replace("sec", " "))
    animal = ANIMAL.findall(name)
    if not (muscle and section and mag and animal):
        raise ValueError(f"cannot parse {path}")
    return {
        "group": CONTROL if is_control(path, root) else TREATED,
        "muscle": muscle.group(1).upper(),
        "animal": animal[-1],
        "section": int(section.group(1)),
        "mag": f"{mag.group(1)}x",
        "edited": path.stem.lower().startswith("edited"),
        "note": "lower_exposure" if "lower_exposure" in path.stem.lower() else "",
    }


def sample_id(r: dict) -> str:
    return f"{r['muscle']}_sec{r['section']}_{r['animal']}_{r['mag']}_{r['group']}"


def main() -> None:
    from msq.dystrophin import checks  # here, because checks imports this module

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--mag", default=settings.MAGNIFICATION)
    ap.add_argument("--out", type=Path, default=paths.project().manifest)
    args = ap.parse_args()
    df, problems = checks.samples_from_folder(args.root, args.mag)
    for p in problems:
        print(p)
    df[MANIFEST_COLUMNS].to_csv(args.out, index=False)
    print(df.drop(columns="path").to_string())
    print(df.groupby(["group", "muscle"])["animal"].agg(["nunique", "count"]))


if __name__ == "__main__":
    main()
