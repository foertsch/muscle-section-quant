"""Per-fiber dystrophin quantification on whole muscle cross-sections (Nikon ND2).

Fibers are segmented with Cellpose-SAM on the actin channel (Cy5) after 4x block-mean
downsampling. Dystrophin (GFP) is then measured at full resolution in a band centred on
each fiber's boundary, and in the fiber core as a per-fiber local background.

Usage:
    python -m msq.dystrophin.measure SCAN.nd2 [SCAN.nd2 ...] --out OUT

A folder of scans, with resume, checks and calls: run_pipeline.py.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path, PureWindowsPath
from typing import Callable

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from skimage.measure import find_contours

from msq.core import nd2, runinfo  # noqa: F401  (dq.runinfo: the tests set its code_version)
from msq.core.fiber import edge_zones, full_res_mask, sector_index
from msq.core.geometry import min_feret_px
from msq.core.images import block_mean as downsample
from msq.core.images import trim_to_multiple
from msq.core.io import read_json
from msq.core.nd2 import channel_index
from msq.core.runner import Provenance  # noqa: F401  (moved to the runner; still dq.Provenance)
from msq.core.segment import Segmenter, cellpose_for
from msq.core.tissue import background_level, main_section, off_tissue, otsu_tissue
from msq.dystrophin import paths, settings

DYSTROPHIN = settings.DYSTROPHIN_CHANNEL
ACTIN = settings.ACTIN_CHANNEL


@dataclass
class Params:
    downsample: int = 4  # segmentation runs at this block-mean factor (20x: 1.44 um/px)
    band_um: float = 1.5  # membrane band half-width, centred on the mask edge
    core_um: float = 3.0  # cytoplasm = fiber pixels deeper than this (capped at half the max depth)
    n_sectors: int = 36  # angular sectors around the centroid for the ridge estimate
    sector_pct: float = 90.0  # percentile of the band within each sector
    smooth_px: float = 1.0  # Gaussian sigma applied to GFP before taking sector percentiles
    min_area_um2: float = 50.0
    max_area_um2: float = 15000.0
    hollow_ratio: float = 0.8  # core/edge actin below this -> vessel-like ring, excluded
    low_actin_frac: float = 0.3  # core actin below this x median -> not a fiber, excluded
    bright_cyto_mad: float = 5.0  # core GFP above median + k*MAD -> necrotic/IgG+ flag
    inverted_threshold: float = 150.0  # ring more than this (a.u.) below own cytoplasm -> artifact, excluded
    tissue_margin_um: float = 50.0  # off-tissue background is taken this far from tissue
    cp_flow_threshold: float = 0.4
    cp_cellprob_threshold: float = 0.0
    cp_batch_size: int = 16


# ---------------------------------------------------------------- I/O


def read_nd2(path: Path) -> dict:
    """Read the dystrophin and actin planes at full resolution (copied out of the memory map) plus pixel size."""
    with nd2.open_scan(path, require={"C", "Y", "X"}) as scan:
        gfp, act = (np.array(scan.frame[channel_index(scan.channels, c)]) for c in (DYSTROPHIN, ACTIN))
        return {"gfp": gfp, "act": act, "px_um": scan.px_um, "sat": 2**scan.bits - 1, "channels": scan.channels}


# ---------------------------------------------------------------- background


def tissue_mask(act_ds: np.ndarray, px_um: float, margin_um: float) -> tuple[np.ndarray, np.ndarray]:
    """Tissue mask on the downsampled actin image, and a mask of confidently off-tissue pixels.

    Zero pixels (padding in edited ND2s, blank areas of raw stitched ones) count as neither.
    """
    valid = act_ds > 0
    tissue = otsu_tissue(ndi.gaussian_filter(act_ds, 2), valid, np.log1p)
    return tissue, off_tissue(tissue, valid, margin_um, px_um)


# ---------------------------------------------------------------- per-fiber measurement


def sector_ridge(values: np.ndarray, yy: np.ndarray, xx: np.ndarray, cy: float, cx: float,
                 n_sectors: int, pct: float) -> np.ndarray:
    """Per angular sector around (cy, cx): the pct-th percentile of band values."""
    sec = sector_index(yy, xx, cy, cx, n_sectors)
    out = np.full(n_sectors, np.nan)
    for s in range(n_sectors):
        v = values[sec == s]
        if v.size >= 3:
            out[s] = np.percentile(v, pct)
    return out


def measure_fiber(mask: np.ndarray, gfp: np.ndarray, act: np.ndarray, px_um: float, p: Params,
                  bg_gfp: float, bg_act: float, sat: int) -> dict:
    """Measure one fiber. mask/gfp/act are same-shape full-res crops with a margin around the fiber."""
    valid = gfp > 0
    band, edge_in, core = edge_zones(mask, px_um, p.band_um, p.core_um, valid)

    g = gfp.astype(np.float32)
    a = act.astype(np.float32)
    gs = ndi.gaussian_filter(g, p.smooth_px)

    cy, cx = np.argwhere(mask).mean(0)
    yy, xx = np.nonzero(band)
    ridge = sector_ridge(gs[band], yy, xx, cy, cx, p.n_sectors, p.sector_pct)
    ridge_bg = ridge - bg_gfp

    mem_peak = float(np.nanmedian(ridge))
    cyto = float(g[core].mean())
    act_core = float(a[core].mean()) - bg_act
    act_edge = float(a[edge_in].mean()) - bg_act
    grad = ndi.gaussian_gradient_magnitude(a, 1.0)

    return {
        "_sectors": np.c_[ridge - bg_gfp, ridge - cyto],  # per-sector values, saved for calls
        "area_um2": float(mask.sum()) * px_um**2,
        "min_feret_um": min_feret_px(mask) * px_um,
        "dys_membrane_peak": mem_peak - bg_gfp,
        "dys_membrane_mean": float(g[band].mean()) - bg_gfp,
        "dys_cytoplasm": cyto - bg_gfp,
        "dys_membrane_specific": mem_peak - cyto,
        "dys_membrane_to_cyto": (mem_peak - bg_gfp) / (cyto - bg_gfp) if cyto > bg_gfp else np.nan,
        "dys_sector_cv": float(np.nanstd(ridge_bg) / np.nanmean(ridge_bg)) if np.nanmean(ridge_bg) > 0 else np.nan,
        "n_sectors_measured": int(np.isfinite(ridge).sum()),
        "actin_core": act_core,
        "actin_edge": act_edge,
        "actin_core_to_edge": act_core / act_edge if act_edge > 0 else np.nan,
        "actin_saturated_frac": float((act[mask] >= sat).mean()),
        "dys_saturated_frac": float((gfp[band] >= sat).mean()),
        "actin_edge_sharpness": float(grad[band].mean()) / act_core if act_core > 0 else np.nan,
        "dys_membrane_specific_per_actin": (mem_peak - cyto) / act_core if act_core > 0 else np.nan,
        # The outline only. Where just the outer band reaches zero padding, the band loses those pixels and the fiber
        # stays in (rare; METHOD.md, caveats).
        "touches_void": bool((~valid & mask).any()),
        "centroid_y_px": cy,
        "centroid_x_px": cx,
    }


def measure_all(labels_ds: np.ndarray, gfp: np.ndarray, act: np.ndarray, px_um: float, p: Params,
                bg_gfp: float, bg_act: float, sat: int, progress: bool = True,
                sectors_out: list | None = None) -> pd.DataFrame:
    """Per-fiber table. If sectors_out is a list, one (label, sector, ridge_bgsub, ridge_minus_cyto)
    array per fiber is appended to it: the same sector values whose median is dys_membrane_specific."""
    f = p.downsample
    gfp = trim_to_multiple(gfp, f)
    act = trim_to_multiple(act, f)
    margin = int(np.ceil((p.band_um + 2 * p.smooth_px * px_um) / (px_um * f))) + 1
    H, W = labels_ds.shape
    rows = []
    objs = ndi.find_objects(labels_ds)
    t0 = time.time()
    for lab, sl in enumerate(objs, start=1):
        if sl is None:
            continue
        y0, y1 = max(sl[0].start - margin, 0), min(sl[0].stop + margin, H)
        x0, x1 = max(sl[1].start - margin, 0), min(sl[1].stop + margin, W)
        m_ds = labels_ds[y0:y1, x0:x1] == lab
        m = full_res_mask(m_ds, f)
        if not m.any():
            continue
        r = measure_fiber(m, gfp[y0 * f:y1 * f, x0 * f:x1 * f], act[y0 * f:y1 * f, x0 * f:x1 * f],
                          px_um, p, bg_gfp, bg_act, sat)
        sec = r.pop("_sectors")
        if sectors_out is not None:
            sectors_out.append(np.c_[np.full(len(sec), lab), np.arange(len(sec)), sec])
        r["centroid_y_px"] += y0 * f
        r["centroid_x_px"] += x0 * f
        r["touches_border"] = sl[0].start == 0 or sl[1].start == 0 or sl[0].stop == H or sl[1].stop == W
        rows.append({"label": lab, **r})
        if progress and lab % 2000 == 0:
            print(f"  measured {lab}/{len(objs)} fibers ({time.time() - t0:.0f} s)", flush=True)
    return pd.DataFrame(rows)


def fiber_geometry(labels: np.ndarray, box: pd.Series, gfp: np.ndarray, f: int, px_um: float,
                   p: Params, pad: int = 24) -> dict:
    """Recompute, for display, exactly what measure_fiber uses: band, sectors, cytoplasm."""
    y0 = max(int(box.by0) - pad, 0) // f * f
    x0 = max(int(box.bx0) - pad, 0) // f * f
    y1 = min(int(box.by1) + pad, labels.shape[0] * f)
    x1 = min(int(box.bx1) + pad, labels.shape[1] * f)
    y1, x1 = y0 + (y1 - y0) // f * f, x0 + (x1 - x0) // f * f
    m_ds = labels[y0 // f:y1 // f, x0 // f:x1 // f] == box.label
    m = full_res_mask(m_ds, f)
    g = gfp[y0:y1, x0:x1].astype(np.float32)
    gs = ndi.gaussian_filter(g, p.smooth_px)
    band, _, core = edge_zones(m, px_um, p.band_um, p.core_um)
    cyto = float(g[core].mean())
    cy, cx = np.argwhere(m).mean(0)

    yy, xx = np.nonzero(band)
    sec = sector_ridge(gs[band], yy, xx, cy, cx, p.n_sectors, p.sector_pct) - cyto
    sec_img = np.full(m.shape, np.nan, np.float32)
    sec_img[yy, xx] = sec[sector_index(yy, xx, cy, cx, p.n_sectors)]
    outline = max(find_contours(np.pad(m, 1).astype(np.float32), 0.5), key=len) - 1
    return {"g": g, "mask": m, "core": core, "outline": outline, "cy": cy, "cx": cx,
            "reach": float(np.hypot(yy - cy, xx - cx).max()) + 4, "sec": sec, "sec_img": sec_img}


# ---------------------------------------------------------------- QC flags


def robust_upper(x: pd.Series, k: float) -> float:
    med = x.median()
    mad = 1.4826 * (x - med).abs().median()
    return med + k * mad


def add_flags(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    """Exclusions (not fibers / not measurable) and the bright-cytoplasm flag.

    `include` marks fibers used for summary statistics.
    """
    df = df.copy()
    df["excl_size"] = ~df["area_um2"].between(p.min_area_um2, p.max_area_um2)
    df["excl_edge"] = df["touches_border"] | df["touches_void"]
    df["excl_hollow"] = ~(df["actin_core_to_edge"] >= p.hollow_ratio)
    base = ~(df["excl_size"] | df["excl_edge"] | df["excl_hollow"])
    med_actin = df.loc[base, "actin_core"].median()
    df["excl_low_actin"] = df["actin_core"] < p.low_actin_frac * med_actin
    # A fiber without dystrophin reads ~0; an interior clearly brighter than the ring means haze,
    # necrotic/IgG+ content or glow from a neighbouring blob (checked visually at 20x).
    df["excl_inverted"] = df["dys_membrane_specific"] < -p.inverted_threshold
    df["excl_off_section"] = ~df["in_main_section"] if "in_main_section" in df else False
    excluded = (df["excl_size"] | df["excl_edge"] | df["excl_hollow"] | df["excl_low_actin"]
                | df["excl_inverted"] | df["excl_off_section"])
    cut = robust_upper(df.loc[~excluded, "dys_cytoplasm"], p.bright_cyto_mad)
    df["flag_bright_cytoplasm"] = ~excluded & (df["dys_cytoplasm"] > cut)
    df["include"] = ~excluded & ~df["flag_bright_cytoplasm"]
    df.attrs["bright_cytoplasm_threshold"] = float(cut)
    return df


# ---------------------------------------------------------------- driver


def sample_name(path: Path) -> str:
    return path.stem.replace("Edited ", "").replace(" ", "_")


def params_for(px_um: float) -> Params:
    """20x defaults below 1 um/px. Coarser scans (4x: 1.83 um/px, NA 0.2) segment at native
    resolution, and band, core, sectors and smoothing follow the 4x radial profile: the GFP ridge
    peaks at +/-1 px from the mask edge and its halo reaches 2 px inside. inverted_threshold is in
    4x camera counts: below -50 every fiber checked was an artifact, above it real ring-less
    fibers appear (4x intensities run ~0.2x of 20x, but -33 would remove those)."""
    if px_um <= 1.0:
        return Params()
    return Params(downsample=1, band_um=2.0, core_um=5.0, n_sectors=16, smooth_px=0.5,
                  inverted_threshold=50.0)


def run(path: Path, out_root: Path, p: Params | None = None, resegment: bool = False,
        name: str | None = None, segment: Segmenter | None = None,
        provenance: Provenance | None = None, read: Callable[[Path], dict] | None = None) -> pd.DataFrame:
    """Measure one scan into out_root/<name>/. The segmentation cached there (labels_ds<f>.tif) is used unless
    resegment; otherwise segment (default: Cellpose-SAM, cellpose_for) makes it from the downsampled actin image.

    With provenance, run_info.json records the code version that measured the section and the Cellpose weights that
    segmented it: this run's weights when it segments, the earlier record when it uses the cached segmentation
    (None when that is unknown). read: the scan reader (default read_nd2; scans.py)."""
    name = name or sample_name(path)
    out = out_root / name
    out.mkdir(parents=True, exist_ok=True)
    previous = read_json(out / "run_info.json")  # read before it is replaced: who made a cached segmentation
    t0 = time.time()
    print(f"[{name}] reading {path.name}", flush=True)
    img = (read or read_nd2)(path)
    p = p or params_for(img["px_um"])
    px_um, f = img["px_um"], p.downsample
    act_ds = downsample(img["act"], f)
    gfp_ds = downsample(img["gfp"], f)

    tissue, off = tissue_mask(act_ds, px_um * f, p.tissue_margin_um)
    bg_gfp = background_level(gfp_ds, off)
    bg_act = background_level(act_ds, off)
    print(f"[{name}] background GFP {bg_gfp:.0f}, actin {bg_act:.0f}; tissue {tissue.mean():.1%} of image", flush=True)

    lab_path = out / f"labels_ds{f}.tif"
    segmented_now = not (lab_path.exists() and not resegment)
    if not segmented_now:
        labels = tifffile.imread(lab_path)
        print(f"[{name}] loaded cached segmentation ({labels.max()} objects)", flush=True)
    else:
        print(f"[{name}] segmenting {act_ds.shape} with Cellpose-SAM", flush=True)
        labels = (segment or cellpose_for(p))(act_ds).astype(np.uint32)
        tifffile.imwrite(lab_path, labels, compression="zlib")
        print(f"[{name}] {labels.max()} objects ({time.time() - t0:.0f} s)", flush=True)

    sectors: list = []
    df = measure_all(labels, img["gfp"], img["act"], px_um, p, bg_gfp, bg_act, img["sat"], sectors_out=sectors)
    sec = pd.DataFrame(np.concatenate(sectors), columns=["label", "sector", "ridge_bgsub", "ridge_minus_cyto"])
    sec = sec.astype({"label": int, "sector": int}).round({"ridge_bgsub": 1, "ridge_minus_cyto": 1})
    sec.to_csv(out / "sectors.csv.gz", index=False)
    main = main_section(tissue, px_um * f)
    for kind, mask in (("tissue", tissue), ("section", main)):  # for the reports and the explainers, not compared
        tifffile.imwrite(out / f"{kind}_ds{f}.tif", mask.astype(np.uint8), compression="zlib")
    yi = np.clip((df["centroid_y_px"] // f).astype(int), 0, main.shape[0] - 1)
    xi = np.clip((df["centroid_x_px"] // f).astype(int), 0, main.shape[1] - 1)
    df["in_main_section"] = main[yi, xi]
    df = add_flags(df, p)
    df.insert(0, "sample", name)
    df.to_csv(out / "fibers.csv", index=False)

    meta = {
        "source": str(path),
        "source_size": path.stat().st_size,  # with the name, tells whether these results come from a given scan
        "channels": img["channels"],
        "pixel_size_um": px_um,
        "saturation_level": img["sat"],
        "background_gfp": bg_gfp,
        "background_actin": bg_act,
        "bright_cytoplasm_threshold_bgsub": df.attrs["bright_cytoplasm_threshold"],
        "n_objects": int(len(df)),
        "n_included": int(df["include"].sum()),
        "params": asdict(p),
        "runtime_s": round(time.time() - t0, 1),
    }
    if provenance is not None:
        meta["code_version"] = provenance.code_version
        meta["cellpose_weights"] = provenance.weights if segmented_now else previous.get("cellpose_weights")
    (out / "run_info.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[{name}] {meta['n_included']}/{meta['n_objects']} fibers included ({meta['runtime_s']} s)", flush=True)
    return df


def made_from(sample_dir: Path, path: Path) -> bool:
    """Whether the results in sample_dir come from the scan at path: same file name (so a scan folder that moved or
    another drive letter does not count as a change), and the same size where run_info.json recorded it. True when
    there is no run_info.json to say otherwise."""
    info_path = sample_dir / "run_info.json"
    if not info_path.exists():
        return True
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if "source" in info and PureWindowsPath(info["source"]).name != PureWindowsPath(path).name:  # either separator
        return False
    size = info.get("source_size")
    return size is None or not path.exists() or path.stat().st_size == size


def is_done(sample_dir: Path, path: Path | None = None) -> bool:
    """run() writes run_info.json last, so its presence marks a finished sample. With path, the results must also
    come from that scan (made_from), so a corrected path in the sample table or a replaced scan is measured again."""
    finished = all((sample_dir / f).exists() for f in ("fibers.csv", "sectors.csv.gz", "run_info.json"))
    return finished and (path is None or made_from(sample_dir, path))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("nd2", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=paths.project().output / "full_20x")
    ap.add_argument("--resegment", action="store_true", help="ignore cached labels and rerun Cellpose")
    args = ap.parse_args()
    provenance = Provenance(lambda: segment_weights_file())
    for path in args.nd2:
        run(path, args.out, resegment=args.resegment, provenance=provenance)


def segment_weights_file() -> Path:
    from msq.core.segment import DEFAULT_MODEL, weights_file

    return weights_file(DEFAULT_MODEL)


if __name__ == "__main__":
    main()
