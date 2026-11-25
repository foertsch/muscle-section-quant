"""The PSR fibrosis pipeline on synthetic sections: no scans, no GPU, no Cellpose weights needed.

    uv run python demo/run_psr_demo.py     # results in demo/output/psr/ (a few tens of seconds on CPU)

Builds seeded synthetic picrosirius red sections (msq.core.synthetic.psr): a control group with little collagen and
a treated group (group_a of cohorts.example.toml) with smaller, more widely spaced fibers and so more collagen between
them, two animals per group and two sections per animal. Each section is run the way the golden test runs it:
section.run_frame with the fibers' known outlines instead of Cellpose, then msq.psr.summary writes the tables and the
two Python figures; the freezing-artifact survey (hole_survey.py) supplies the share the R figures mark. If Rscript is on the PATH, the R figures are drawn as well (R/figures.R); without R that step is
skipped.

Look at:
    demo/output/psr/tables/sections.csv   per section: collagen_outside_fraction (the primary readout) and more
    demo/output/psr/tables/animals.csv    per animal
    demo/output/psr/figures/              the R figures, if R ran
    demo/output/psr/run/sections/base/    one folder per section (normalised colour, classes, outlines)
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
import zlib
from pathlib import Path

import pandas as pd

from msq.core.segment import FixedLabels
from msq.core.synthetic.psr import synthetic_psr_section
from msq.psr import hole_survey, section, summary
from msq.psr.settings import load_cohort

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# animal numbers and groups as in the PSR golden: control, and group_a (cohorts.example.toml) with more collagen
DESIGN = {"control": ("A01", "A02"), "group_a": ("B01", "B02")}
FIBROSIS = {"control": "low", "group_a": "high"}  # synthetic_psr_section's two settings
MUSCLE = "QUAD"


def build_design(sections: int = 2) -> list[dict]:
    return [{"sample": f"{MUSCLE}_sec{sec}_{animal}_{group}", "group": group, "muscle": MUSCLE, "animal": animal,
             "section": sec} for group, animals in DESIGN.items() for animal in animals
            for sec in range(1, sections + 1)]


def run_demo(out: Path, sections: int = 2, size: int = 3600, r_figures: bool = True) -> dict:
    """Build, run and summarise the synthetic design into out; returns how long it took and whether R drew figures."""
    t0 = time.time()
    shutil.rmtree(out, ignore_errors=True)
    run, tables, figures = out / "run", out / "tables", out / "figures"
    params = section.params_for("base")
    design, holes = build_design(sections), []
    for d in design:
        s = synthetic_psr_section(zlib.crc32(d["sample"].encode()), FIBROSIS[d["group"]], size=size)
        section.run_frame(s["frame"], s["px_um"], run / "sections" / "base" / d["sample"], params, d["sample"],
                          source=f"synthetic:{d['sample']}", segment=FixedLabels(s["labels_ds"]), variant="base")
        holes.append({"sample": d["sample"], **hole_survey.survey(s["frame"], s["px_um"])[0]})
    pd.DataFrame([{**d, "path": f"synthetic:{d['sample']}"} for d in design]).to_csv(run / "samples.csv", index=False)
    tables.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(holes)[["sample", "holey_area"]].to_csv(out / "hole_survey.csv", index=False)  # the R figures mark it
    summary.write_tables(run, tables, load_cohort("example"), holes=out / "hole_survey.csv")
    drawn = r_figures and draw_r_figures(tables, figures)
    return {"seconds": round(time.time() - t0, 1), "r_figures": drawn}


def draw_r_figures(tables: Path, figures: Path) -> bool:
    """R/figures.R on the tables; False (and a note) if Rscript is missing or the script fails."""
    rscript = shutil.which("Rscript")
    if rscript is None:
        print("Rscript not found: skipping the R figures")
        return False
    done = subprocess.run([rscript, str(ROOT / "R" / "figures.R"), "psr", str(tables), str(figures)],
                          capture_output=True, text=True, cwd=ROOT)
    if done.returncode != 0:
        print(f"the R figures failed (status {done.returncode}); the Python tables and figures are complete:\n"
              f"{done.stderr[-600:]}")
    return done.returncode == 0


def report(out: Path) -> str:
    animals = pd.read_csv(out / "tables" / "animals.csv", dtype={"animal": str})
    lines = ["collagen outside the fiber outlines, % of red plus yellow (mean of each animal's sections):"]
    for r in animals.sort_values(["group", "animal"]).itertuples():
        lines.append(f"  {r.animal}  {r.group:<8} {100 * r.collagen_outside_fraction:5.1f}%")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=HERE / "output" / "psr")
    ap.add_argument("--no-r", action="store_true", help="skip the R figures")
    args = ap.parse_args()
    got = run_demo(args.out, r_figures=not args.no_r)
    print(report(args.out))
    print(f"\nsegmentation: the sections' own outlines (no model); {got['seconds']} s")
    print(f"results in {args.out}: tables/" + (", figures/" if got["r_figures"] else "")
          + ", run/sections/base/ (use demo/make_crops.py with assay = \"psr\" for method crops)")


if __name__ == "__main__":
    sys.exit(main())
