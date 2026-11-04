"""Fiber size distribution (min Feret diameter) per animal, QUAD and TA side by side.

There is no fiber-type stain in these scans, so this is size only. Included fibers of all sections of an
animal are pooled; each row is one animal, % of fibers per 5 µm bin, with the median (line) and the
variance coefficient (SD / mean x 1000; above 250 is the usual sign of dystrophic size variation).

Writes deliverables/figures/results/fiber_size_distribution.png and deliverables/tables/fiber_size_by_animal.csv.

Usage:
    python -m msq.dystrophin.plot_fiber_size output/full_20x --manifest manifest_20x.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from msq.dystrophin import compare_report as cr  # noqa: E402
from msq.dystrophin import paths  # noqa: E402
from msq.dystrophin import qc  # noqa: E402
from msq.dystrophin import settings  # noqa: E402

EDGES = np.arange(0, 101, 5.0)  # µm; fibers of 100 µm and more count in the last bin


def load_sizes(results: Path, meta: pd.DataFrame) -> pd.DataFrame:
    """Included fibers of every section in meta with their size, muscle, group and animal."""
    rows = []
    for s, m in meta.iterrows():
        f = pd.read_csv(results / s / "fibers.csv", usecols=["include", "min_feret_um", "area_um2"])
        f = f[f["include"]].drop(columns="include")
        rows.append(f.assign(sample=s, muscle=m["muscle"], group=m["group"], animal=str(m["animal"])))
    return pd.concat(rows, ignore_index=True)


def size_stats(v: pd.Series) -> dict:
    v = v.dropna()
    return {"n": int(len(v)), "median": float(v.median()), "q25": float(v.quantile(0.25)),
            "q75": float(v.quantile(0.75)), "mean": float(v.mean()), "sd": float(v.std()),
            "cv_x1000": float(1000 * v.std() / v.mean()), "pct_below_20um": float(100 * (v < 20).mean())}


def animal_order(sizes: pd.DataFrame) -> list[tuple[str, str]]:
    """(group, animal), controls first."""
    pairs = sizes[["group", "animal"]].drop_duplicates()
    return sorted(map(tuple, pairs.to_numpy()), key=lambda p: (p[0] != settings.GROUPS.control, p[1]))


def stats_table(sizes: pd.DataFrame) -> pd.DataFrame:
    rows = [{"muscle": m, "group": g, "animal": a, **size_stats(d["min_feret_um"])}
            for (m, g, a), d in sizes.groupby(["muscle", "group", "animal"])]
    return pd.DataFrame(rows)


def plot(sizes: pd.DataFrame, table: pd.DataFrame, out: Path) -> None:
    muscles = [m for m in settings.MUSCLES if m in set(sizes["muscle"])]
    order = animal_order(sizes)
    hists = {(m, g, a): cr.hist_pct(d["min_feret_um"], EDGES)
             for (m, g, a), d in sizes.groupby(["muscle", "group", "animal"])}
    ymax = 1.12 * max(h.max() for h in hists.values())
    fig, axs = plt.subplots(len(order), len(muscles), figsize=(6.4 * len(muscles), 1.55 * len(order) + 1.2),
                            sharex=True, sharey=True, squeeze=False)
    centers, width = (EDGES[:-1] + EDGES[1:]) / 2, np.diff(EDGES)[0]
    for j, m in enumerate(muscles):
        axs[0, j].set_title(m, loc="left", fontsize=12, color=qc.INK, fontweight="bold")
        for i, (g, a) in enumerate(order):
            ax = axs[i, j]
            if (m, g, a) in hists:
                ax.bar(centers, hists[(m, g, a)], width=width * 0.86, color=qc.GROUP_COLORS[g], linewidth=0)
                st = table[(table["muscle"] == m) & (table["group"] == g) & (table["animal"] == a)].iloc[0]
                ax.axvline(st["median"], color=qc.INK, lw=1.2)
                ax.text(0.99, 0.92, f"median {st['median']:.1f} µm · CV {st['cv_x1000']:.0f} · n {st['n']:,}",
                        transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=qc.INK)
            else:  # rows are shared by both muscles, and this animal has no sections of this one
                ax.text(0.5, 0.5, "no sections", transform=ax.transAxes, ha="center", va="center", fontsize=8.5,
                        color=qc.MUTED)
            if j == 0:
                ax.text(-0.13, 0.5, f"{a}\n{settings.GROUP_NAMES[g]}", transform=ax.transAxes, ha="right", va="center",
                        fontsize=9.5, color=qc.INK)
            ax.grid(axis="x", visible=False)
            ax.set_ylim(0, ymax)
        axs[-1, j].set_xlabel("min Feret diameter (µm)")
        axs[-1, j].set_xticks(EDGES[::2])
    for i in range(len(order)):
        axs[i, 0].set_ylabel("% of fibers", fontsize=8)
    fig.text(0.01, 0.005, "Included fibers, all sections of an animal pooled; % of fibers per 5 µm bin (last bin: 95 µm "
             "and above). Line: median. CV = SD / mean × 1000. No fiber-type stain in these scans, so this is size only.",
             fontsize=7.5, color=qc.MUTED)
    fig.tight_layout(rect=(0.04, 0.02, 1, 1))
    qc.save(fig, out, dpi=170)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--figures", type=Path, default=paths.project().figures_results)
    ap.add_argument("--tables", type=Path, default=paths.project().tables)
    args = ap.parse_args()
    meta = pd.read_csv(args.manifest, dtype={"animal": str}).set_index("sample")
    make(args.results, meta, args.figures, args.tables)


def make(results: Path, meta: pd.DataFrame, figures: Path, tables: Path) -> None:
    """Figure and table for the sections in meta (indexed by sample) that have results."""
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    meta = meta[[(results / s / "fibers.csv").exists() for s in meta.index]]
    sizes = load_sizes(results, meta)
    table = stats_table(sizes)
    table.round(2).to_csv(tables / "fiber_size_by_animal.csv", index=False)
    plot(sizes, table, figures / "fiber_size_distribution.png")
    print(table.round(1).to_string(index=False))
    print(f"wrote {figures / 'fiber_size_distribution.png'} and {tables / 'fiber_size_by_animal.csv'}")


if __name__ == "__main__":
    main()
