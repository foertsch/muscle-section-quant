"""The dystrophin pipeline on synthetic sections: no scans, no GPU, no Cellpose weights needed.

    uv run python demo/run_demo.py              # about a minute; results in demo/output/run/
    uv run python demo/run_demo.py --cellpose   # segment with Cellpose-SAM instead of the known outlines
                                                # (downloads 1.15 GB of weights on first use)

Builds a scan folder of seeded synthetic sections of a tibialis anterior (TA): three untreated control animals and
two treated ones, two sections each. Every section is a disc of tissue with about 300 fibers, each with an actin
signal and a dystrophin ring over part of its edge (control rings dim, treated rings bright over a random share of
the edge), plus a few things the pipeline must leave out or flag: a fiber with bright cytoplasm, a fiber cut by the
image edge, a torn-off fragment and blank padding along one side. (The tests' sections also carry a vessel-like
hollow object and debris inside the tissue; here they are left out, because the reports show windows free of
excluded objects.)

The sections are saved as .npz files and read with msq.dystrophin.scans.NpzScans, and the pipeline runs exactly as
on ND2 scans (run_pipeline.run): sample table and checks, measurement, the threshold from the controls, the calls,
tables, figures and both interactive reports. Without --cellpose, each section's known outlines are put in the run
folder first, as a segmentation cached from an earlier run, so no model is needed.

Look at:
    demo/output/run/reports/threshold_decision.html   move the threshold and see the calls change
    demo/output/run/reports/ta_ctrl_vs_treated.html controls and treated side by side
    demo/output/run/figures/pct_positive_call.png  % positive per animal
    demo/output/run/tables/animals.csv             the numbers
"""

from __future__ import annotations

import argparse
import contextlib
import io
import shutil
import sys
import zlib
from pathlib import Path

import pandas as pd
import tifffile

from msq.core.synthetic.dystrophin import synthetic_section
from msq.dystrophin import run_pipeline
from msq.dystrophin.scans import NpzScans, save_npz

HERE = Path(__file__).resolve().parent
ANIMALS = {"ctrl": ("1001", "1002", "1003"), "treated": ("1101", "1102")}  # made-up animal numbers
SIZE, PITCH, TISSUE_RADIUS = 2800, 104, 0.40  # large enough for the reports' 360 um windows
FIBER_RADIUS = (46, 54)  # px at 0.36 um/px: fibers of about 700 to 1,200 um2, as in mouse muscle


def scan_path(scans: Path, group: str, animal: str, section: int) -> Path:
    """The folder layout the sample table reads: controls in a folder that starts with Neg-ctrl."""
    folder = scans / "Neg-ctrl-TA" / animal if group == "ctrl" else scans / f"{animal} TA"
    return folder / f"TA sec{section} {animal} 20x.npz"


def build_scans(scans: Path, sections: int = 2, size: int = SIZE) -> dict[Path, dict]:
    """Write the synthetic scans; returns each scan's section (with its known labels)."""
    out = {}
    for group, animals in ANIMALS.items():
        for animal in animals:
            for sec in range(1, sections + 1):
                p = scan_path(scans, group, animal, sec)
                s = synthetic_section(zlib.crc32(p.name.encode()), group, size=size, pitch=PITCH,
                                      tissue_radius=TISSUE_RADIUS, interior_artifacts=False,
                                      fiber_radius=FIBER_RADIUS)
                save_npz(p, s)
                out[p.resolve()] = s
    return out


def run_demo(out: Path, cellpose: bool = False, sections: int = 2, size: int = SIZE, quiet: bool = True) -> int:
    shutil.rmtree(out, ignore_errors=True)
    scans, run = out / "scans", out / "run"
    built = build_scans(scans, sections, size)
    reader = NpzScans()
    log = io.StringIO()
    redirect = (contextlib.redirect_stdout(log), contextlib.redirect_stderr(log)) if quiet else ()
    with contextlib.ExitStack() as stack:
        for r in redirect:
            stack.enter_context(r)
        code = run_pipeline.run(scans, run, check_only=True, reader=reader)  # the sample table and the checks
        if code != 0:
            return code
        if not cellpose:  # the known outlines, as a segmentation cached from an earlier run
            samples = pd.read_csv(run / "samples.csv", dtype=str)
            for sample, path in zip(samples["sample"], samples["path"]):
                (run / "sections" / sample).mkdir(parents=True, exist_ok=True)
                tifffile.imwrite(run / "sections" / sample / "labels_ds4.tif", built[Path(path)]["labels_ds"],
                                 compression="zlib")
        code = run_pipeline.run(scans, run, reader=reader)
    return code


def summary(run: Path) -> str:
    animals = pd.read_csv(run / "tables" / "animals.csv", dtype={"animal": str})
    lines = ["% of fibers positive (mean of each animal's sections):"]
    for r in animals.sort_values(["group", "animal"]).itertuples():
        lines.append(f"  {r.animal}  {r.group:<8} {r.pct_positive_50:5.1f}%   ({r.n_called:,} fibers, threshold from "
                     f"{r.threshold_from})")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=HERE / "output")
    ap.add_argument("--cellpose", action="store_true", help="segment with Cellpose-SAM instead of the known outlines")
    ap.add_argument("--verbose", action="store_true", help="show the pipeline's own output")
    args = ap.parse_args()
    code = run_demo(args.out, args.cellpose, quiet=not args.verbose)
    run = args.out / "run"
    if code != 0:
        sys.exit(f"the run ended with status {code}; see {run / 'run.log'}")
    print(summary(run))
    print(f"\nsegmentation: {'Cellpose-SAM' if args.cellpose else 'the sections own outlines (no model)'}")
    print(f"results in {run}: tables/, figures/, reports/ (open reports/threshold_decision.html)")


if __name__ == "__main__":
    main()
