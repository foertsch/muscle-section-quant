"""Controls and treated of one muscle side by side, on one brightness scale.

One section per animal: the one whose median membrane dystrophin is the middle of that animal's
sections. From each, the report shows a 360 µm tissue region (the densely packed, in-focus window whose
local median is closest to the section median) and single fibers at the 10th, 50th and 90th percentile
of membrane dystrophin. Below come histograms of per-fiber membrane dystrophin, fiber scores against the
threshold and GFP pixels. Images are the GFP channel minus each section's off-tissue background, and every
sample uses the same linear display range.

Usage:
    python -m msq.dystrophin.compare_report output/full_20x --manifest manifest_20x.csv --muscle QUAD
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from msq.core.fiber import full_res_mask
from msq.core.report import render
from msq.core.report.hist import hist_pct, shared_edges  # noqa: F401  (the dystrophin reports and plots import them from here)
from msq.core.tissue import off_tissue
from msq.dystrophin import measure as dq
from msq.dystrophin import paths
from msq.dystrophin import report_utils as ru

QUANTILES = (0.10, 0.50, 0.90)
CLOSEUP_UM = 110.0
FIBER_AREA_UM2 = (800.0, 3500.0)  # close-ups use mid-sized fibers, so shape does not distract
LEVELS = ("50", "60", "70")
BIN_AU = 25.0
PIXEL_LO = -200.0


# ---------------------------------------------------------------- choices


def typical_sections(sections: pd.DataFrame, muscle: str) -> pd.DataFrame:
    """One row per animal: the section with the middle median membrane dystrophin. Controls first."""
    s = sections[sections["muscle"] == muscle].sort_values("median_dys_membrane_specific")
    out = pd.DataFrame([g.iloc[(len(g) - 1) // 2] for _, g in s.groupby("animal", sort=False)])
    out["_treated"] = out["group"] != "ctrl"
    return out.sort_values(["_treated", "animal"]).drop(columns="_treated").reset_index(drop=True)


def matched_fibers(fib: pd.DataFrame, shape: tuple[int, int], px_um: float, quantiles=QUANTILES,
                   area=FIBER_AREA_UM2, window_um: float = CLOSEUP_UM) -> list[dict]:
    """Per quantile of membrane dystrophin over included fibers: the mid-sized fiber closest to that value
    whose close-up window fits inside the image."""
    inc = fib[fib["include"]]
    half = window_um / px_um / 2
    ok = inc[inc["area_um2"].between(*area)
             & inc["centroid_y_px"].between(half, shape[0] - half)
             & inc["centroid_x_px"].between(half, shape[1] - half)]
    out = []
    for q in quantiles:
        target = float(inc["dys_membrane_specific"].quantile(q))
        i = (ok["dys_membrane_specific"] - target).abs().idxmin()
        out.append({"q": q, "label": int(ok.at[i, "label"]), "target": target,
                    "value": float(ok.at[i, "dys_membrane_specific"])})
    return out


def typical_window(cands: pd.DataFrame, section_median: float) -> dict:
    """The candidate window whose local median membrane dystrophin is closest to the section median."""
    c = cands.loc[(cands["local_median"] - section_median).abs().idxmin()]
    return {"y0": int(c["y0"]), "x0": int(c["x0"]), "local_median": float(c["local_median"])}


def display_ranges(fibers: list[pd.DataFrame]) -> dict[str, tuple[float, float]]:
    """Two linear ranges on background-subtracted GFP, shared by every image.

    full: black at background, white at the 99th percentile of membrane ridge over all included fibers.
    contrast: black at the 25th percentile of fiber interior (cytoplasm) GFP, white at the 95th
    percentile of membrane ridge, so membrane above the interior stands out.
    """
    inc = pd.concat([f[f["include"]] for f in fibers])
    r10 = lambda v: float(np.round(v, -1))  # noqa: E731
    return {"full": (0.0, r10(np.percentile(inc["dys_membrane_peak"], 99))),
            "contrast": (r10(np.percentile(inc["dys_cytoplasm"], 25)),
                         r10(np.percentile(inc["dys_membrane_peak"], 95)))}


# ---------------------------------------------------------------- histograms


def pixel_hists(gfp: np.ndarray, bg: float, tissue_ds: np.ndarray, off_ds: np.ndarray, f: int,
                edges: np.ndarray) -> dict:
    """Background-subtracted GFP pixel histograms, tissue vs off-tissue.

    One native pixel per f x f block (no averaging, so pixel noise is kept) on the grid of the masks.
    Zero pixels are blank padding in edited ND2s and are skipped. sd_off is the robust SD of off-tissue pixels.
    """
    g = dq.trim_to_multiple(gfp, f)[::f, ::f]
    assert g.shape == tissue_ds.shape == off_ds.shape
    valid = g > 0
    sub = g.astype(np.float32) - bg
    t, o = sub[tissue_ds & valid], sub[off_ds & valid]
    return {"tissue": hist_pct(t, edges), "off": hist_pct(o, edges),
            "median_tissue": float(np.median(t)), "sd_off": robust_sd(o)}


def robust_sd(v: np.ndarray) -> float:
    """1.4826 x median absolute deviation: the SD of the noise, unaffected by debris on the slide."""
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


# ---------------------------------------------------------------- images and outlines


def outline(labels: np.ndarray, f: int, r, y0: int, x0: int, margin: int = 2) -> str:
    """SVG path of one fiber's outline at full resolution, relative to a window at (y0, x0)."""
    ly0, ly1 = max(r.by0 // f - margin, 0), min(r.by1 // f + margin, labels.shape[0])
    lx0, lx1 = max(r.bx0 // f - margin, 0), min(r.bx1 // f + margin, labels.shape[1])
    m = full_res_mask(labels[ly0:ly1, lx0:lx1] == r.label, f)
    return ru.outline_path(m, ly0 * f - y0, lx0 * f - x0)


def fiber_base(r) -> dict:
    """What both report pages show of a fiber (a fibers.csv row): label, status, membrane dystrophin and area."""
    return {"label": int(r.label),
            "status": "included" if r.include else "flagged" if r.flag_bright_cytoplasm else "excluded",
            "dys": None if pd.isna(r.dys_membrane_specific) else round(float(r.dys_membrane_specific), 1),
            "area": round(float(r.area_um2))}


def fiber_info(r, calls: pd.DataFrame) -> dict:
    """Compact per-fiber record for the page: status, values and the call at each level."""
    d = fiber_base(r)
    if r.label in calls.index:
        for k in LEVELS:
            s = calls.at[r.label, f"score_{k}"]
            d[f"s{k}"] = None if pd.isna(s) else round(float(s), 1)
            d[f"c{k}"] = str(calls.at[r.label, f"call_{k}"])
    return d


def grey_images(crop: np.ndarray, bg: float, ranges: dict) -> dict[str, str]:
    sub = crop.astype(np.float32) - bg
    return {k: ru.jpeg_b64(ru.to_uint8(sub, lo, hi), 90) for k, (lo, hi) in ranges.items()}


# ---------------------------------------------------------------- per sample


def pixel_masks(res_dir: Path, act: np.ndarray, f: int, px_um: float, p) -> tuple[np.ndarray, np.ndarray]:
    """The main section and the off-tissue mask on the segmentation grid: the section from the run's section_ds<f>.tif,
    the off-tissue ring grown from its tissue_ds<f>.tif (blank padding, zero in the actin plane, stays out). A run made
    before those files has both recomputed from the actin plane."""
    act_ds = dq.downsample(act, f)
    stored = [res_dir / f"{kind}_ds{f}.tif" for kind in ("tissue", "section")]
    if all(path.exists() for path in stored):
        tissue, section = (tifffile.imread(path) > 0 for path in stored)
        return section, off_tissue(tissue, act_ds > 0, p.tissue_margin_um, px_um * f)
    tissue, off = dq.tissue_mask(act_ds, px_um * f, p.tissue_margin_um)
    return dq.main_section(tissue, px_um * f), off


def extract(res_dir: Path, nd2_path: Path, calls: pd.DataFrame, ranges: dict, pixel_edges: np.ndarray,
            read=None) -> dict:
    """Region, matched fibers and pixel histograms for one section. calls: indexed by label."""
    info = json.loads((res_dir / "run_info.json").read_text(encoding="utf-8"))
    f, px_um, bg = info["params"]["downsample"], info["pixel_size_um"], info["background_gfp"]
    p = dq.params_for(px_um)
    df = pd.read_csv(res_dir / "fibers.csv")
    labels = tifffile.imread(res_dir / f"labels_ds{f}.tif")
    img = (read or dq.read_nd2)(nd2_path)
    gfp = dq.trim_to_multiple(img["gfp"], f)
    H, W = gfp.shape
    fib = ru.fiber_boxes(labels, f).merge(df, on="label").set_index("label", drop=False)

    section, off = pixel_masks(res_dir, img["act"], f, px_um, p)
    del img
    pix = pixel_hists(gfp, bg, section, off, f, pixel_edges)

    cands = ru.candidate_windows(fib.reset_index(drop=True), (H, W), px_um)
    win = typical_window(cands, float(df.loc[df["include"], "dys_membrane_specific"].median()))
    y0, x0, n = win["y0"], win["x0"], ru.CROP_PX
    inside = fib[ru.inside_window(fib, y0, x0, n)]
    region = {"y0": y0, "x0": x0, "size": n, "local_median": round(win["local_median"], 1),
              "img": grey_images(gfp[y0:y0 + n, x0:x0 + n], bg, ranges),
              "cells": [{**fiber_info(r, calls), "path": outline(labels, f, r, y0, x0)}
                        for r in inside.itertuples()]}

    m = int(round(CLOSEUP_UM / px_um))
    closeups = []
    for pick in matched_fibers(fib, (H, W), px_um):
        r = next(fib.loc[[pick["label"]]].itertuples())
        cy0 = int(np.clip(round(r.centroid_y_px - m / 2), 0, H - m))
        cx0 = int(np.clip(round(r.centroid_x_px - m / 2), 0, W - m))
        closeups.append({**pick, **fiber_info(r, calls), "size": m, "path": outline(labels, f, r, cy0, cx0),
                         "img": grey_images(gfp[cy0:cy0 + m, cx0:cx0 + m], bg, ranges)})
    print(f"[{res_dir.name}] region at ({y0}, {x0}), local median {win['local_median']:.0f}; fibers "
          + ", ".join(f"{int(100 * c['q'])}th {c['value']:.0f}" for c in closeups), flush=True)
    return {"bg": bg, "px_um": px_um, "region": region, "closeups": closeups, "pixel": pix,
            "fibers": df[df["include"]]}


# ---------------------------------------------------------------- assembly


def sample_stats(row: pd.Series, inc: pd.DataFrame, calls: pd.DataFrame) -> dict:
    """The page's numbers for one section. % positive counts the calls (calls.csv.gz), so a control animal's are its
    held-out calls; for a treated section that equals the share of called fibers whose score reaches threshold."""
    called = calls[calls["call_50"] != "not called"]
    v = inc["dys_membrane_specific"]
    return {"sample": row["sample"], "animal": str(row["animal"]), "group": row["group"],
            "section": int(row["section"]), "n_included": int(len(inc)), "n_called": int(len(called)),
            "dys_median": float(v.median()), "dys_p10": float(v.quantile(0.1)), "dys_p90": float(v.quantile(0.9)),
            "cyto_median": float(inc["dys_cytoplasm"].median()),
            "ridge_median": float(inc["dys_membrane_peak"].median()),
            **{f"score_median_{k}": float(called[f"score_{k}"].median()) for k in LEVELS},
            **{f"pct_positive_{k}": float(100 * (called[f"call_{k}"] == "positive").mean()) for k in LEVELS}}


def build_payload(rows: pd.DataFrame, extracted: dict, calls: dict, threshold: float, ranges: dict,
                  pixel_edges: np.ndarray, muscle: str) -> dict:
    samples = list(rows["sample"])
    inc = {s: extracted[s]["fibers"] for s in samples}
    fiber_edges = shared_edges([inc[s]["dys_membrane_specific"] for s in samples])
    called = {s: calls[s][calls[s]["call_50"] != "not called"] for s in samples}
    score_edges = shared_edges([called[s][f"score_{k}"] for s in samples for k in LEVELS])
    r1 = lambda a: [round(float(x), 3) for x in a]  # noqa: E731
    out = {
        "generated": date.today().isoformat(), "muscle": muscle, "threshold": threshold, "levels": list(LEVELS),
        "ranges": ranges, "region_um": round(ru.CROP_PX * extracted[samples[0]]["px_um"]),
        "closeup_um": CLOSEUP_UM, "quantiles": [int(100 * q) for q in QUANTILES],
        "hist": {"fiber": {"edges": r1(fiber_edges), "rows": {}},
                 "score": {"edges": r1(score_edges), "rows": {}},
                 "pixel": {"edges": r1(pixel_edges), "rows": {}}},
        "samples": [],
    }
    for _, row in rows.iterrows():
        s, e = row["sample"], extracted[row["sample"]]
        stats = sample_stats(row, inc[s], calls[s])
        out["hist"]["fiber"]["rows"][s] = r1(hist_pct(inc[s]["dys_membrane_specific"], fiber_edges))
        out["hist"]["score"]["rows"][s] = {k: r1(hist_pct(called[s][f"score_{k}"], score_edges)) for k in LEVELS}
        out["hist"]["pixel"]["rows"][s] = {"tissue": r1(e["pixel"]["tissue"]), "off": r1(e["pixel"]["off"])}
        out["samples"].append({**stats, "bg": e["bg"], "tissue_px_median": e["pixel"]["median_tissue"],
                               "off_px_sd": e["pixel"]["sd_off"], "region": e["region"],
                               "closeups": e["closeups"]})
    return out


def write_html(payload: dict, out: Path, template: str = "compare_template.html") -> None:
    render(Path(__file__).parent / template, out, data=payload)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path, help="full-dataset results folder (with summary/)")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--muscle", default="QUAD")
    ap.add_argument("--sections", nargs="*", help="sample names to use instead of the typical section per animal")
    ap.add_argument("--tables", type=Path, default=paths.project().tables, help="full_summary.py tables")
    ap.add_argument("--out", type=Path, default=paths.project().reports / "quad_ctrl_vs_treated.html")
    args = ap.parse_args()
    nd2_paths = pd.read_csv(args.manifest).set_index("sample")["path"].map(Path).to_dict()
    build(args.results, args.tables, nd2_paths, args.muscle, args.out, args.sections)


def build(results: Path, tables: Path, nd2_paths: dict, muscle: str, out: Path,
          sample_names: list[str] | None = None, read=None) -> None:
    """The report for one muscle from full_summary.py's tables: the typical section per animal, or sample_names."""
    sections = pd.read_csv(tables / "sections.csv", dtype={"animal": str})
    rows = (sections.set_index("sample").loc[sample_names].reset_index() if sample_names
            else typical_sections(sections, muscle))
    print("sections: " + ", ".join(rows["sample"]), flush=True)
    th = json.loads((tables / "thresholds.json").read_text(encoding="utf-8"))
    threshold = float(th["muscles"][muscle]["threshold"])
    all_calls = pd.read_csv(tables / "calls.csv.gz")
    calls = {s: all_calls[all_calls["sample"] == s].set_index("label") for s in rows["sample"]}

    fibers = [pd.read_csv(results / s / "fibers.csv") for s in rows["sample"]]
    ranges = display_ranges(fibers)
    pixel_edges = np.arange(PIXEL_LO, np.ceil(1.25 * ranges["full"][1] / 100) * 100 + BIN_AU / 2, BIN_AU)
    print(f"display ranges (background-subtracted a.u.): {ranges}", flush=True)

    extracted = {s: extract(results / s, nd2_paths[s], calls[s], ranges, pixel_edges, read) for s in rows["sample"]}
    payload = build_payload(rows, extracted, calls, threshold, ranges, pixel_edges, muscle)
    payload["threshold_percentile"] = round(100 * (1 - th["fpr"]))
    write_html(payload, out)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
