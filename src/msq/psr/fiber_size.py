"""Fiber size per genotype: min Feret diameter, max Feret diameter and cross-sectional area.

Reads a run folder (sections/<variant>/<sample>/objects.csv from section.py, samples.csv from checks.py) and writes
to OUT:
    fiber_size_sections.csv     per section: n fibers, median, IQR, variance coefficient
    fiber_size_groups.csv       the same per genotype (fibers pooled; animals are not yet weighted equally)

Which objects count as fibers (a provisional rule): in the main section, not touching the image edge,
FIBER_AREA_UM2 in area, and an outline that is less than half collagen.

Usage:
    python -m msq.psr.fiber_size RUN --out OUT --cohort NAME [--variant base]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from msq.core.geometry import feret_px
from msq.psr import checks
from msq.psr.settings import Cohort, load_cohort

FIBER_AREA_UM2 = (50.0, 15000.0)
MAX_RED_SHARE = 0.5
MEASURES = ("min_feret_um", "max_feret_um", "area_um2")


def add_max_feret(o: pd.DataFrame, folder: Path) -> pd.DataFrame:
    """Max Feret from the saved outlines, for sections measured before section.py recorded it."""
    import json

    import tifffile
    from skimage.measure import regionprops

    info = json.loads((folder / "section.json").read_text())
    labels = tifffile.imread(folder / f"labels_ds{info['params']['downsample']}.tif")
    fmax = {rp.label: feret_px(rp.image)[1] * info["grid_um"] for rp in regionprops(labels)}
    return o.assign(max_feret_um=o["label"].map(fmax))


def fibers(run: Path, variant: str = "base") -> pd.DataFrame:
    samples = checks.read_samples(run / "samples.csv").set_index("sample")
    rows = []
    for d in sorted((run / "sections" / variant).iterdir()):
        if d.name.endswith(".partial") or not (d / "objects.csv").exists():  # .partial: a copy still under way
            continue
        o = pd.read_csv(d / "objects.csv")
        if o.empty:
            continue
        if "max_feret_um" not in o:
            o = add_max_feret(o, d)
        px = o[[c for c in o.columns if c.startswith(("core_", "band_"))]].sum(axis=1)
        red_share = (o["core_red"] + o["band_red"]) / px
        keep = (o["in_section"] & ~o["touches_border"] & o["area_um2"].between(*FIBER_AREA_UM2)
                & (red_share < MAX_RED_SHARE))
        f = o.loc[keep, ["label", *MEASURES]].assign(sample=d.name)
        f["group"] = samples.loc[d.name, "group"]
        f["animal"] = samples.loc[d.name, "animal"]
        rows.append(f)
    if not rows:
        raise SystemExit(f"no sections with objects in {run / 'sections' / variant}")
    return pd.concat(rows, ignore_index=True)


def describe(df: pd.DataFrame, by: list[str], cohort: Cohort) -> pd.DataFrame:
    def stats(g: pd.DataFrame) -> pd.Series:
        out = {"n_fibers": len(g)}
        for m in MEASURES:
            v = g[m]
            out.update({f"{m}_median": v.median(), f"{m}_q25": v.quantile(0.25), f"{m}_q75": v.quantile(0.75)})
        out["min_feret_variance_coefficient"] = 1000 * g["min_feret_um"].std() / g["min_feret_um"].mean()
        return pd.Series(out)

    out = df.groupby(by, sort=False).apply(stats, include_groups=False).reset_index()
    return out.sort_values(by, key=cohort.sort_key(), kind="stable", ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cohort", required=True, help="the dataset's cohort in cohorts.local.toml")
    ap.add_argument("--variant", default="base")
    args = ap.parse_args()
    cohort = load_cohort(args.cohort)
    args.out.mkdir(parents=True, exist_ok=True)
    df = fibers(args.run, args.variant)
    per_section = describe(df, ["group", "animal", "sample"], cohort)
    per_group = describe(df, ["group"], cohort)
    per_group.insert(1, "genotype", per_group["group"].map(cohort.labels))
    per_section.round(2).to_csv(args.out / "fiber_size_sections.csv", index=False)
    per_group.round(2).to_csv(args.out / "fiber_size_groups.csv", index=False)
    print(per_section.round(1).to_string(index=False))
    print(per_group.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
