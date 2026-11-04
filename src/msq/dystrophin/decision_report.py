"""Threshold decision report: move the threshold and see calls, numbers, histograms and fibers update together.

Per muscle (QUAD, TA) the page holds every called fiber's score at 50 / 60 / 70% of the ring (sorted, from
deliverables/tables/calls.csv.gz). The threshold is the chosen percentile of pooled control fibers' 50% score and
is computed in the page with numpy's linear percentile, so every number is exact at any setting. Images
come from one typical section per animal (as in compare_report.py): the whole section with every called fiber's
position, so the page can mark the positive ones; a 360 µm region whose share of positive fibers at the summary's
threshold matches the section's, with outlines that recolour with the threshold; and a pool of fibers whose scores
sit across the range of possible thresholds, from which the page shows the three just below and the three just above
the current threshold.

Usage:
    python -m msq.dystrophin.decision_report output/full_20x --manifest manifest_20x.csv
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from skimage.measure import block_reduce

from msq.dystrophin import compare_report as cr
from msq.dystrophin import measure as dq
from msq.dystrophin import paths
from msq.dystrophin import report_utils as ru
from msq.dystrophin import settings

LEVELS = ("50", "60", "70")
P_RANGE = (90.0, 99.5)        # slider range: percentile of pooled control fibers' 50% score
CLOSEUP_UM = 80.0
CLOSEUP_BIN = 2               # close-ups are 2 x 2 block means (0.72 µm/px) to keep the page small
POOL_TARGETS = 16
POOL_AREA_UM2 = (500.0, 4000.0)
OV_MAX_PX = 1000              # whole-section overviews: longest side at most this many pixels


# ---------------------------------------------------------------- data for the page


def encode_sorted(values) -> list[int]:
    """Scores (0.1 a.u. grid) as sorted integers x10, delta-encoded: the page restores them by a running sum."""
    a = np.asarray(values, float)
    v = np.sort(np.round(a[np.isfinite(a)] * 10).astype(np.int64))
    return np.diff(v, prepend=0).tolist()


def threshold_range(ctrl_scores_50, p_range=P_RANGE) -> tuple[float, float]:
    """Thresholds at the two ends of the slider."""
    return float(np.percentile(ctrl_scores_50, p_range[0])), float(np.percentile(ctrl_scores_50, p_range[1]))


def borderline_pool(scores: pd.DataFrame, lo: float, hi: float, n_targets: int = POOL_TARGETS,
                    levels=LEVELS) -> list[int]:
    """Labels of fibers whose score at some level lies near one of n_targets + 2 threshold values spread from one
    spacing below lo to one spacing above hi. Targets with no fiber within one spacing are skipped.
    scores: eligible fibers indexed by label, with score_<level> columns."""
    spacing = (hi - lo) / (n_targets - 1)
    targets = np.linspace(lo - spacing, hi + spacing, n_targets + 2)
    chosen: list[int] = []
    for k in levels:
        s = scores[f"score_{k}"].dropna()
        for t in targets:
            cand = s.drop(chosen, errors="ignore")
            if cand.empty:
                break
            i = (cand - t).abs().idxmin()
            if abs(cand[i] - t) <= spacing:
                chosen.append(int(i))
    return chosen


def window_by_share(cands: pd.DataFrame, fib: pd.DataFrame, positive: pd.Series, target: float,
                    section_median: float, size: int = ru.CROP_PX) -> dict:
    """The candidate window whose share of positive fibers (among called fibers fully inside) is closest to
    target (%); ties go to the window whose local median membrane dystrophin is closest to the section median.
    positive: 1.0 / 0.0 per called fiber, indexed by label; fib: indexed by label with bounding boxes."""
    share = [100 * positive.reindex(fib.index[ru.inside_window(fib, c.y0, c.x0, size)]).dropna().mean()
             for c in cands.itertuples()]
    d = cands.assign(share=share).dropna(subset=["share"])
    d = d.assign(_a=(d["share"] - target).abs(), _b=(d["local_median"] - section_median).abs())
    c = d.sort_values(["_a", "_b"], kind="stable").iloc[0]
    return {"y0": int(c["y0"]), "x0": int(c["x0"]), "local_median": float(c["local_median"]), "share": float(c["share"])}


def overview_factor(shape: tuple[int, int], f: int, max_px: int = OV_MAX_PX) -> int:
    """Block size, on top of the analysis downsampling f, that brings the longest side to max_px or less."""
    return max(1, int(np.ceil(max(shape) / f / max_px)))


def fiber_dots(centroids: pd.DataFrame, calls: pd.DataFrame, scale: float) -> dict:
    """Called fibers on the overview: positions in overview pixels and scores x10 as integers (None if missing).
    centroids: indexed by label with centroid_y_px / centroid_x_px (full resolution); calls: called fibers."""
    c = centroids.join(calls[[f"score_{k}" for k in LEVELS]], how="inner")
    out = {"fx": np.round(c["centroid_x_px"] / scale).astype(int).tolist(),
           "fy": np.round(c["centroid_y_px"] / scale).astype(int).tolist()}
    for k in LEVELS:
        out[f"f{k}"] = [None if pd.isna(v) else int(round(v * 10)) for v in c[f"score_{k}"]]
    return out


def fiber_record(r, calls: pd.DataFrame) -> dict:
    """Status, values and the score at each level; the page decides the call from the current threshold."""
    d = cr.fiber_base(r)
    for k in LEVELS:
        v = calls.at[r.label, f"score_{k}"] if r.label in calls.index else np.nan
        d[f"s{k}"] = None if pd.isna(v) else round(float(v), 1)
    return d


# ---------------------------------------------------------------- images


def extract(res_dir: Path, nd2_path: Path, calls: pd.DataFrame, hi_full: float, lo: float, hi: float,
            threshold: float, read=None) -> dict:
    """Region (outlines + scores) and the borderline pool for one section. calls: called fibers by label;
    threshold: the summary's threshold, used to pick a region with the section's share of positive fibers at 50%."""
    info = json.loads((res_dir / "run_info.json").read_text(encoding="utf-8"))
    f, px_um, bg = info["params"]["downsample"], info["pixel_size_um"], info["background_gfp"]
    df = pd.read_csv(res_dir / "fibers.csv")
    labels = tifffile.imread(res_dir / f"labels_ds{f}.tif")
    gfp = dq.trim_to_multiple((read or dq.read_nd2)(nd2_path)["gfp"], f)
    H, W = gfp.shape
    fib = ru.fiber_boxes(labels, f).merge(df, on="label").set_index("label", drop=False)
    jpeg = lambda crop: ru.jpeg_b64(ru.to_uint8(crop.astype(np.float32) - bg, 0.0, hi_full), 90)  # noqa: E731

    cands = ru.candidate_windows(fib.reset_index(drop=True), (H, W), px_um)
    positive = (calls["score_50"] >= threshold).astype(float)
    win = window_by_share(cands, fib, positive, 100 * float(positive.mean()),
                          float(df.loc[df["include"], "dys_membrane_specific"].median()))
    y0, x0, n = win["y0"], win["x0"], ru.CROP_PX
    region = {"size": n, "img": jpeg(gfp[y0:y0 + n, x0:x0 + n]),
              "cells": [{**fiber_record(r, calls), "path": cr.outline(labels, f, r, y0, x0)}
                        for r in fib[ru.inside_window(fib, y0, x0, n)].itertuples()]}

    m = int(round(CLOSEUP_UM / px_um / CLOSEUP_BIN)) * CLOSEUP_BIN
    half = m / 2
    ok = fib[fib["include"] & fib["area_um2"].between(*POOL_AREA_UM2)
             & fib["centroid_y_px"].between(half, H - half) & fib["centroid_x_px"].between(half, W - half)]
    eligible = calls.loc[calls.index.intersection(ok.index), [f"score_{k}" for k in LEVELS]]
    pool = []
    for lab in borderline_pool(eligible, lo, hi):
        r = next(fib.loc[[lab]].itertuples())
        cy0, cx0 = int(round(r.centroid_y_px - half)), int(round(r.centroid_x_px - half))
        crop = block_reduce(gfp[cy0:cy0 + m, cx0:cx0 + m].astype(np.float32), (CLOSEUP_BIN, CLOSEUP_BIN), np.mean)
        pool.append({**fiber_record(r, calls), "size": m, "img": jpeg(crop),
                     "path": cr.outline(labels, f, r, cy0, cx0)})
    k = overview_factor((H, W), f)
    ov = block_reduce(dq.downsample(gfp, f), (k, k), np.mean) - bg
    scale = f * k
    overview = {"w": int(ov.shape[1]), "h": int(ov.shape[0]), "px_um": px_um * scale,
                "box": [round(x0 / scale, 1), round(y0 / scale, 1), round(n / scale, 1)],
                **fiber_dots(df.set_index("label")[["centroid_y_px", "centroid_x_px"]], calls, scale)}
    print(f"[{res_dir.name}] region {win['share']:.1f}% positive (section {100 * positive.mean():.1f}%), "
          f"local median {win['local_median']:.0f}, {len(region['cells'])} fibers; "
          f"borderline pool {len(pool)}", flush=True)
    return {"bg": bg, "px_um": px_um, "region": region, "pool": pool, "overview": overview, "ov": ov}


# ---------------------------------------------------------------- assembly


def muscle_payload(muscle: str, results: Path, sections: pd.DataFrame, calls_all: pd.DataFrame,
                   nd2_paths: dict, default_p: float, read=None) -> dict:
    sec = sections[sections["muscle"] == muscle].copy()
    sec["_t"] = sec["group"] != settings.GROUPS.control
    sec = sec.sort_values(["_t", "animal", "section"]).drop(columns="_t")
    called = {s: calls_all[(calls_all["sample"] == s) & (calls_all["call_50"] != "not called")].set_index("label")
              for s in sec["sample"]}
    ctrl50 = pd.concat([called[s]["score_50"] for s in sec.loc[sec["group"] == settings.GROUPS.control, "sample"]])
    lo, hi = threshold_range(ctrl50)
    threshold = float(np.percentile(ctrl50, default_p))

    rows = cr.typical_sections(sections, muscle)
    ranges = cr.display_ranges([pd.read_csv(results / s / "fibers.csv") for s in rows["sample"]])
    animals, ovs = [], []
    for _, row in rows.iterrows():
        s = row["sample"]
        e = extract(results / s, nd2_paths[s], called[s], ranges["full"][1], lo, hi, threshold, read)
        px_um = e["px_um"]
        animals.append({"animal": str(row["animal"]), "group": row["group"], "typical": s,
                        "section": int(row["section"]), "bg": e["bg"], "region": e["region"], "pool": e["pool"],
                        "overview": e["overview"]})
        ovs.append(e["ov"])
    # one range for the muscle's overviews: black at background, white at the 98th percentile of tissue pixels
    # (bright vessels and debris saturate rather than dim the tissue)
    ov_hi = float(np.round(np.percentile(np.concatenate([o[o > 50].ravel() for o in ovs]), 98), -1))
    for a, o in zip(animals, ovs):
        a["overview"]["img"] = ru.jpeg_b64(ru.to_uint8(o, 0.0, ov_hi), 88)

    by_animal = {a: [called[s] for s in g["sample"]] for a, g in sec.groupby("animal")}
    hist = {}
    for k in LEVELS:
        arrays = {a: pd.concat([c[f"score_{k}"] for c in cs]).to_numpy() for a, cs in by_animal.items()}
        edges = cr.shared_edges(list(arrays.values()))
        hist[k] = {"edges": edges.round(3).tolist(),
                   "rows": {a: cr.hist_pct(v, edges).round(3).tolist() for a, v in arrays.items()}}
    return {
        "muscle": muscle, "ranges": ranges, "overview_range": [0.0, ov_hi], "region_um": round(ru.CROP_PX * px_um),
        "closeup_um": CLOSEUP_UM,
        "threshold_range": [lo, hi],
        "sections": [{"sample": r["sample"], "animal": str(r["animal"]), "group": r["group"],
                      "section": int(r["section"]), "n": int(len(called[r["sample"]])),
                      **{f"d{k}": encode_sorted(called[r["sample"]][f"score_{k}"]) for k in LEVELS}}
                     for _, r in sec.iterrows()],
        "animals": animals, "hist": hist,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path, help="full-dataset results folder (with summary/)")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--muscles", nargs="+", default=["QUAD", "TA"])
    ap.add_argument("--tables", type=Path, default=paths.project().tables, help="full_summary.py tables")
    ap.add_argument("--out", type=Path, default=paths.project().reports / "threshold_decision.html")
    args = ap.parse_args()
    nd2_paths = pd.read_csv(args.manifest).set_index("sample")["path"].map(Path).to_dict()
    build(args.results, args.tables, nd2_paths, args.muscles, args.out)


def build(results: Path, tables: Path, nd2_paths: dict, muscles: list[str], out: Path, read=None) -> None:
    """The report for the given muscles, from full_summary.py's tables. Each muscle needs control sections."""
    sections = pd.read_csv(tables / "sections.csv", dtype={"animal": str})
    calls_all = pd.read_csv(tables / "calls.csv.gz")
    th = json.loads((tables / "thresholds.json").read_text(encoding="utf-8"))
    default_p = round(100 * (1 - th["fpr"]), 1)
    payload = {"generated": date.today().isoformat(), "levels": list(LEVELS), "p_range": list(P_RANGE),
               "default_p": default_p, "order": list(muscles),
               "muscles": {m: muscle_payload(m, results, sections, calls_all, nd2_paths, default_p, read)
                           for m in muscles}}
    cr.write_html(payload, out, template="decision_template.html")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
