"""Readouts per section, animal and genotype: collagen outside the fibers, and fiber size.

Collagen outside the fibers (section.json `collagen_outside_fraction`), the primary readout, called "collagen outside
the fiber outlines": red pixels in the main section outside every Cellpose-SAM outline, divided by the red plus
yellow pixels in the main section. Only the main section is measured (the largest tissue piece,
msq.core.tissue.main_section). All red outside the outlines counts (perimysium, epimysium, tendon, vessel walls).
Red inside any outline does not count, including outlines of objects that are not fibers, so the readout depends on
the segmentation. `red_fraction` (all red / (red + yellow) in the main section) is a secondary readout.

Folds, dust, edge staining and tendon are not masked, and collagen is counted over the whole main section. No section
is excluded; sections with freezing artifacts are marked, not removed (see --holes below).

Fiber size from the outlines that pass the fiber rule in fiber_size.py. An animal's value is the mean of its
sections (collagen) or the median of all its fibers (size); a genotype's is the mean of its animals, which are the
unit of comparison.

Writes to OUT:
    sections.csv, animals.csv, genotypes.csv
    fibers.csv.gz                      every fiber that passes the rule (for the R figures, R/figures.R)
    palette.csv                        genotype order, display name and colour (from the cohort)
    collagen_outside_by_genotype.png   one dot per animal, small marks per section, line = genotype mean
    collagen_vs_fiber_size.png         per animal: collagen outside the fibers against median min Feret

Usage:
    python -m msq.psr.summary RUN --out OUT --cohort NAME [--variant base] [--holes HOLE_SURVEY.csv]

With --holes, sections.csv gets freezing_artifact_area: the share of the section's fiber area in 200 um tiles with
freezing artifacts (small white holes inside fibers, hole_survey.py). The R figures mark sections where that share is
above 0.3; no section is left out of the tables or the statistics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from msq.core.calls import animal_means, group_stats  # noqa: E402
from msq.psr import checks, fiber_size  # noqa: E402
from msq.psr.settings import Cohort, load_cohort  # noqa: E402


def section_table(run: Path, cohort: Cohort, variant: str = "base") -> pd.DataFrame:
    samples = checks.read_samples(run / "samples.csv").set_index("sample")
    fibers = fiber_size.fibers(run, variant)
    size = fibers.groupby("sample").agg(n_fibers=("label", "size"), min_feret_median_um=("min_feret_um", "median"),
                                        max_feret_median_um=("max_feret_um", "median"),
                                        area_median_um2=("area_um2", "median"))
    rows = []
    for d in sorted((run / "sections" / variant).iterdir()):
        if d.name.endswith(".partial") or not (d / "section.json").exists():  # .partial: a copy still under way
            continue
        info = json.loads((d / "section.json").read_text())
        s = samples.loc[d.name]
        rows.append({"sample": d.name, "group": s["group"], "animal": s["animal"], "section": int(s["section"]),
                     "collagen_outside_fraction": info["collagen_outside_fraction"],
                     "red_fraction": info["red_fraction"], "section_area_mm2": info["section_area_mm2"]})
    out = pd.DataFrame(rows).join(size, on="sample")
    unknown = sorted(set(out["group"]) - set(cohort.order))
    if unknown:
        raise ValueError(f"groups {', '.join(unknown)} are not in cohort {cohort.name}; pass the cohort this run "
                         "was made with")
    return out


def animal_table(sections: pd.DataFrame, fibers: pd.DataFrame) -> pd.DataFrame:
    a = animal_means(sections, ["group", "animal"], ["collagen_outside_fraction", "red_fraction"])
    size = fibers.groupby("animal").agg(n_fibers=("label", "size"), min_feret_median_um=("min_feret_um", "median"),
                                        max_feret_median_um=("max_feret_um", "median"))
    return a.join(size, on="animal")


def genotype_table(animals: pd.DataFrame, cohort: Cohort) -> pd.DataFrame:
    cols = ["collagen_outside_fraction", "red_fraction", "min_feret_median_um", "max_feret_median_um"]
    g = group_stats(animals, ["group"], cols, sd="std")
    g = g[g["group"].isin(cohort.order)].sort_values("group", key=cohort.sort_key(), kind="stable", ignore_index=True)
    g.insert(1, "genotype", g["group"].map(cohort.labels))
    return g


def edge(colour: str) -> tuple:
    """A darker shade of the same hue for outlines (saffron is too light on white)."""
    r, g, b = mcolors.to_rgb(colour)
    return (0.7 * r, 0.7 * g, 0.7 * b)


def plot_collagen(sections: pd.DataFrame, animals: pd.DataFrame, cohort: Cohort, out: Path) -> None:
    groups = [g for g in cohort.order if g in set(animals["group"])]
    fig, ax = plt.subplots(figsize=(5.6, 4.4), dpi=150)
    rng = np.random.default_rng(0)
    for i, g in enumerate(groups):
        colour = cohort.colours[g]
        s = sections[sections["group"] == g]
        a = animals[animals["group"] == g]
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(s)), 100 * s["collagen_outside_fraction"], s=14,
                   facecolors="none", edgecolors=edge(colour), linewidths=0.8, zorder=2)
        ax.scatter(i + rng.uniform(-0.08, 0.08, len(a)), 100 * a["collagen_outside_fraction"], s=60, color=colour,
                   edgecolors=edge(colour), linewidths=1, zorder=3)
        m = 100 * a["collagen_outside_fraction"].mean()
        ax.hlines(m, i - 0.28, i + 0.28, color="0.15", linewidth=1.5, zorder=4)
        ax.text(i + 0.31, m, f"{m:.1f}%", va="center", fontsize=8.5, color="0.15")
    ax.set_xticks(range(len(groups)), [f"{cohort.labels[g]}\n(n = {(animals['group'] == g).sum()})"
                                       for g in groups])
    ax.set_xlim(-0.6, len(groups) - 0.25)
    ax.set_ylabel("Collagen outside the fibers (% of red + yellow)")
    ax.set_ylim(0, None)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.9", linewidth=0.6, zorder=0)
    ax.set_title("Large dots: animals (mean of sections). Rings: sections. Line: genotype mean.", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_vs_size(animals: pd.DataFrame, cohort: Cohort, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 4.4), dpi=150)
    for g in [x for x in cohort.order if x in set(animals["group"])]:
        a = animals[animals["group"] == g]
        colour = cohort.colours[g]
        ax.scatter(a["min_feret_median_um"], 100 * a["collagen_outside_fraction"], s=60, color=colour,
                   edgecolors=edge(colour), linewidths=1, label=cohort.labels[g], zorder=3)
    ax.set_xlabel("Median min Feret diameter of the animal's fibers (µm)")
    ax.set_ylabel("Collagen outside the fibers (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="0.9", linewidth=0.6, zorder=0)
    ax.legend(frameon=False, fontsize=8.5)
    ax.set_title("One point per animal", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def write_tables(run: Path, out: Path, cohort: Cohort, variant: str = "base", holes: Path | None = None
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every table and figure of this module into out; returns the animal and the genotype table."""
    out.mkdir(parents=True, exist_ok=True)
    sections = section_table(run, cohort, variant)
    if holes:
        survey = pd.read_csv(holes)[["sample", "holey_area"]].rename(
            columns={"holey_area": "freezing_artifact_area"})  # share of fiber area in tiles with freezing artifacts
        sections = sections.merge(survey, on="sample", how="left")
    fibers = fiber_size.fibers(run, variant)
    animals = animal_table(sections, fibers)
    genotypes = genotype_table(animals, cohort)
    sections.round(4).to_csv(out / "sections.csv", index=False)
    fibers[["group", "animal", "sample", "label", "min_feret_um", "max_feret_um", "area_um2"]].round(2).to_csv(
        out / "fibers.csv.gz", index=False)
    cohort.palette(sections["group"]).to_csv(out / "palette.csv", index=False)   # groups with data, for the stats
    animals.round(4).to_csv(out / "animals.csv", index=False)
    genotypes.round(4).to_csv(out / "genotypes.csv", index=False)
    plot_collagen(sections, animals, cohort, out / "collagen_outside_by_genotype.png")
    plot_vs_size(animals, cohort, out / "collagen_vs_fiber_size.png")
    return animals, genotypes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cohort", required=True, help="the dataset's cohort in cohorts.local.toml")
    ap.add_argument("--variant", default="base")
    ap.add_argument("--holes", type=Path, help="hole_survey.csv, to add each section's freezing-artifact share")
    args = ap.parse_args()
    animals, genotypes = write_tables(args.run, args.out, load_cohort(args.cohort), args.variant, args.holes)
    print(animals.round(3).to_string(index=False))
    print(genotypes.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
