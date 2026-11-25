"""README crops from one section of a finished run: the segmentation outlines, one fiber's ring sectors and, if asked,
the calls. Images only (no text, no numbers), written straight from arrays as PNG files without metadata.
With assay = "psr" in the config the crops come from a finished PSR run instead (see "PSR mode" below).

    uv run python demo/make_crops.py demo/crops.local.toml

Copy demo/crops.example.toml to demo/crops.local.toml (git ignores it) and set the run folder and the section. The
crops come from what the run measured: its outlines (labels_ds<f>.tif), the fibers' status (fibers.csv), the sector
values (sectors.csv.gz) and the calls (tables/calls.csv.gz); only the scan is read again, for the pixels.

Writes to the out folder:
    segmentation.png   a square window: actin (red) and dystrophin (green), every object outlined by its status:
                       white included, orange bright cytoplasm (flagged), grey excluded; scale bar 50 um
    ring_sectors.png   one included fiber twice: its outline (white) and cytoplasm core (amber) on dystrophin
                       (green), then the band around the outline in its 36 sectors, each coloured by its value
                       (darker = dimmer, scaled within this fiber); scale bar 10 um
    calls.png          with calls = true: the window's dystrophin, outlines coloured by the 50% call: blue positive,
                       light grey negative, dark grey not called or excluded

PSR mode (assay = "psr"): the same window of one section four ways, from the per-section outputs of msq.psr.section
(rgb_ds<f>.tif, classes_ds<f>.tif, labels_ds<f>.tif, section_ds<f>.tif in run/sections/<variant>/<sample>/), each
section grid pixel drawn 3 x 3. The scan is not read. Writes:
    psr_colour.png            the colour image divided by the glass colour; scale bar 50 um
    psr_classes.png           the same window with the pixel classes: red (collagen) and yellow (fiber) shown, the
                              others dimmed
    psr_outlines.png          the Cellpose-SAM outlines (white) on the colour image
    psr_collagen_outside.png  red-class pixels in the main section outside every outline, highlighted (magenta);
                              red inside an outline (dark red) does not count towards the readout
The window is the pick (lower, typical, higher) among windows mostly inside the main section, by the share of collagen
outside the outlines, or the one given by at.

The window is one the reports would pick (densely covered by in-focus included fibers, no excluded objects), at the
10th, 50th or 90th percentile of local dystrophin (pick = lower, typical, higher), or the one given by at. The file
names and the images carry no sample name, animal, group or date; only the console output names the section.
"""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from skimage.segmentation import find_boundaries

from msq.core.fiber import edge_zones, full_res_mask, sector_index
from msq.dystrophin import measure as dq
from msq.dystrophin import qc
from msq.dystrophin.report_utils import candidate_windows, fiber_boxes, inside_window, to_uint8
from msq.dystrophin.scans import ND2Scans, NpzScans

HERE = Path(__file__).resolve().parent
DEFAULTS = {"run": None, "sample": None, "scan": None, "pick": "typical", "at": None, "window_um": 360,
            "fiber": None, "calls": False, "out": HERE / "crops", "assay": "dystrophin", "variant": "base"}
ASSAYS = ("dystrophin", "psr")
PICKS = {"lower": 0.10, "typical": 0.50, "higher": 0.90}  # quantile of the candidate windows' local medians
WINDOW_UM = (300, 600)  # the README image rules
NAMES = ("segmentation.png", "ring_sectors.png", "calls.png")
PSR_NAMES = ("psr_colour.png", "psr_classes.png", "psr_outlines.png", "psr_collagen_outside.png")
PSR_ZOOM = 3  # PSR crops: each section grid pixel drawn as 3 x 3
PSR_MIN_SECTION = 0.7  # a PSR window must lie at least this much inside the main section
PSR_COLOURS = {"red": "#d6333a", "yellow": "#f5d142", "outside": "#ff2bd6", "inside": "#5a1418", "outline": "#ffffff"}

STATUS = {"included": "#ffffff", "flagged": qc.FLAGGED, "excluded": qc.MUTED}
CALL = {"positive": "#6da7ec", "negative": "#d9d8d2", "other": "#6b6a65"}  # as the reports draw them
CORE = "#fab219"
RAMP = LinearSegmentedColormap.from_list(
    "ring", ["#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"])  # dark -> light on black
ZOOM = 3  # ring_sectors.png: each scan pixel drawn as 3 x 3


def rgb(hex_: str) -> np.ndarray:
    return np.array([int(hex_[i:i + 2], 16) for i in (1, 3, 5)], np.uint8)


def read_config(path: Path) -> dict:
    cfg = {**DEFAULTS, **tomllib.loads(path.read_text(encoding="utf-8"))}
    unknown = set(cfg) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown keys in {path.name}: {', '.join(sorted(unknown))}")
    if cfg["assay"] not in ASSAYS:
        raise ValueError(f"assay must be one of {', '.join(ASSAYS)}, not {cfg['assay']!r}")
    for key in ("run", "sample"):
        if cfg[key] is None:
            raise ValueError(f"{path.name} needs {key} = ...")
    for key in ("run", "scan", "out"):
        if cfg[key] is not None:
            cfg[key] = Path(cfg[key]).expanduser()
    return cfg


# ---------------------------------------------------------------- the section


def reader_for(scan: Path):
    return NpzScans() if scan.suffix == ".npz" else ND2Scans()


def load_section(run: Path, sample: str, scan: Path | None = None) -> dict:
    """What the run measured for one section, and the scan's two channels on the segmentation grid."""
    samples = pd.read_csv(run / "samples.csv", dtype=str).set_index("sample")
    d = run / "sections" / sample
    if sample not in samples.index or not (d / "fibers.csv").is_file():
        raise ValueError(f"no section {sample!r} in {run / 'samples.csv'} with results in {run / 'sections'}")
    info = json.loads((d / "run_info.json").read_text(encoding="utf-8"))
    f, px_um = int(info["params"]["downsample"]), float(info["pixel_size_um"])
    scan = Path(scan or samples.loc[sample, "path"])
    img = reader_for(scan).read(scan)
    fibers = pd.read_csv(d / "fibers.csv")
    calls = run / "tables" / "calls.csv.gz"
    calls = pd.read_csv(calls).query("sample == @sample").set_index("label") if calls.is_file() else None
    return {"f": f, "px_um": px_um, "labels": tifffile.imread(d / f"labels_ds{f}.tif"), "fibers": fibers,
            "sectors": pd.read_csv(d / "sectors.csv.gz"), "calls": calls,
            "gfp": dq.trim_to_multiple(img["gfp"], f), "act": dq.trim_to_multiple(img["act"], f),
            "bg": {"gfp": float(info["background_gfp"]), "act": float(info["background_actin"])},
            "status": {int(r.label): "included" if r.include else "flagged" if r.flag_bright_cytoplasm else "excluded"
                       for r in fibers.itertuples()}}


def display_range(sec: dict, ch: str) -> tuple[float, float]:
    """Black = off-tissue background, white = the 99th percentile of the included fibers' ring (dystrophin) or
    core (actin), as in the reports."""
    col = {"gfp": "dys_membrane_peak", "act": "actin_core"}[ch]
    inc = sec["fibers"].loc[sec["fibers"]["include"], col]
    return sec["bg"][ch], sec["bg"][ch] + float(np.nanpercentile(inc, 99))


# ---------------------------------------------------------------- window and fiber


def boxes(sec: dict) -> pd.DataFrame:
    return fiber_boxes(sec["labels"], sec["f"]).merge(sec["fibers"], on="label")


def pick_window(sec: dict, size: int, pick: str) -> tuple[int, int]:
    f = sec["f"]
    cands = candidate_windows(boxes(sec), sec["gfp"].shape, sec["px_um"], size=size, stride=max(size // 5 // f, 1) * f)
    if cands.empty:
        raise ValueError(f"no window of {size} px here is densely covered by in-focus included fibers; "
                         "give one with at = [y, x]")
    target = cands["local_median"].quantile(PICKS[pick])
    w = cands.loc[(cands["local_median"] - target).abs().idxmin()]
    return int(w["y0"]), int(w["x0"])


def pick_fiber(fib: pd.DataFrame) -> int:
    """A round fiber with an even, fully measured ring, near the 85th percentile of ring brightness in the window."""
    c = fib[fib["include"]].copy()
    c["elong"] = c["area_um2"] / (np.pi * (c["min_feret_um"] / 2) ** 2)
    even = c[(c["elong"] < 1.35) & (c["dys_sector_cv"] < 0.2) & (c["area_um2"] > 900)
             & (c["n_sectors_measured"] == c["n_sectors_measured"].max())]
    c = even if len(even) else c
    q85 = c["dys_membrane_specific"].quantile(0.85)
    return int(c.loc[(c["dys_membrane_specific"] - q85).abs().idxmin(), "label"])


# ---------------------------------------------------------------- drawing


def inner_edge(labels: np.ndarray, width: int) -> np.ndarray:
    """The outermost width pixels inside each object (value = label), so the ring just outside stays visible."""
    rest, edge = labels.copy(), np.zeros_like(labels)
    for _ in range(width):
        e = find_boundaries(rest, mode="inner") & (rest > 0)
        edge[e], rest[e] = rest[e], 0
    return edge


def outlines(labels_crop: np.ndarray, f: int, width: int = 2) -> np.ndarray:
    """Full-resolution outline pixels (value = label) of every object, built the way the measurement builds masks."""
    up = np.zeros((labels_crop.shape[0] * f, labels_crop.shape[1] * f), np.int32)
    for lab in np.unique(labels_crop[labels_crop > 0]):
        up[full_res_mask(labels_crop == lab, f)] = lab
    return inner_edge(up, width)


def paint(img: np.ndarray, edges: np.ndarray, colour_of: dict[int, str], default: str) -> None:
    lut = np.tile(rgb(default), (int(edges.max()) + 1, 1))
    for lab, c in colour_of.items():
        if lab < len(lut):
            lut[lab] = rgb(c)
    on = edges > 0
    img[on] = lut[edges[on]]


def scale_bar(img: np.ndarray, length_px: float) -> None:
    h = img.shape[0]
    m, t = int(0.04 * h), max(3, h // 160)
    img[h - m - t:h - m, m:m + int(round(length_px))] = 255


def save_png(img: np.ndarray, path: Path) -> None:
    Image.fromarray(img).save(path, format="PNG")  # IHDR, IDAT and IEND only: no text, time or profile


def window_images(sec: dict, y0: int, x0: int, size: int) -> dict[str, np.ndarray]:
    f = sec["f"]
    sl = np.s_[y0:y0 + size, x0:x0 + size]
    g = to_uint8(sec["gfp"][sl], *display_range(sec, "gfp"))
    a = to_uint8(sec["act"][sl], *display_range(sec, "act"))
    edges = outlines(sec["labels"][y0 // f:(y0 + size) // f, x0 // f:(x0 + size) // f], f)

    seg = np.stack([a, g, np.zeros_like(g)], -1)
    paint(seg, edges, {k: STATUS[v] for k, v in sec["status"].items()}, STATUS["excluded"])
    out = {"segmentation.png": seg}
    if sec["calls"] is not None:
        calls = np.stack([g] * 3, -1)
        call = sec["calls"]["call_50"]
        paint(calls, edges, {int(k): CALL.get(v, CALL["other"]) for k, v in call.items()}, CALL["other"])
        out["calls.png"] = calls
    for img in out.values():
        scale_bar(img, 50 / sec["px_um"])
    return out


def fiber_rings(sec: dict, label: int, pad: int = 24) -> dict:
    """One fiber's outline, cytoplasm core and band in sectors, recomputed exactly as measure.measure_fiber draws
    them; values are the fiber's stored sector values (sectors.csv.gz, ring minus own cytoplasm) by sector."""
    f, px_um = sec["f"], sec["px_um"]
    p = dq.params_for(px_um)
    box = fiber_boxes(sec["labels"], f).set_index("label").loc[label]
    H, W = sec["gfp"].shape
    y0, x0 = max(int(box.by0) - pad, 0) // f * f, max(int(box.bx0) - pad, 0) // f * f
    y1, x1 = min(int(box.by1) + pad, H), min(int(box.bx1) + pad, W)
    y1, x1 = y0 + (y1 - y0) // f * f, x0 + (x1 - x0) // f * f
    m_ds = sec["labels"][y0 // f:y1 // f, x0 // f:x1 // f] == label
    mask = full_res_mask(m_ds, f)
    raw = sec["gfp"][y0:y1, x0:x1]
    band, _, core = edge_zones(mask, px_um, p.band_um, p.core_um, raw > 0)
    cy, cx = np.argwhere(mask).mean(0)
    yy, xx = np.mgrid[:mask.shape[0], :mask.shape[1]]
    sector = np.where(band, sector_index(yy, xx, cy, cx, p.n_sectors), -1)
    stored = sec["sectors"].query("label == @label").set_index("sector")["ridge_minus_cyto"]
    return {"gfp": raw.astype(np.float32), "mask": mask, "band": band, "core": core, "sector": sector, "cy": cy,
            "cx": cx, "values": stored.reindex(range(p.n_sectors)).to_numpy(float)}


def ring_image(sec: dict, label: int) -> np.ndarray:
    g = fiber_rings(sec, label)
    base = to_uint8(g["gfp"], *display_range(sec, "gfp"))
    left = np.stack([np.zeros_like(base), base, np.zeros_like(base)], -1)  # dystrophin green, as in segmentation.png
    left[inner_edge(g["mask"].astype(np.int32), 1) > 0] = rgb(STATUS["included"])
    left[inner_edge(g["core"].astype(np.int32), 1) > 0] = rgb(CORE)
    scale_bar(left, 10 / sec["px_um"])

    right = np.stack([(base * 0.45).astype(np.uint8)] * 3, -1)
    v = g["values"]
    hi = np.nanpercentile(v, 97) if np.isfinite(v).any() else 1.0
    colour = (RAMP(np.clip(np.nan_to_num(v) / (hi if hi > 0 else 1.0), 0, 1))[:, :3] * 255).astype(np.uint8)
    s = g["sector"]
    shown = (s >= 0) & np.isfinite(v)[np.maximum(s, 0)]
    right[shown] = colour[s[shown]]
    cut = np.zeros_like(shown)  # lines between neighbouring sectors
    cut[:, :-1] |= (s[:, :-1] != s[:, 1:]) & (s[:, :-1] >= 0) & (s[:, 1:] >= 0)
    cut[:-1] |= (s[:-1] != s[1:]) & (s[:-1] >= 0) & (s[1:] >= 0)
    right[cut] = (right[cut] * 0.35).astype(np.uint8)

    zoom = [np.kron(im, np.ones((ZOOM, ZOOM, 1), np.uint8)) for im in (left, right)]
    gap = np.zeros((zoom[0].shape[0], 12 * ZOOM, 4), np.uint8)  # transparent
    rgba = [np.dstack([im, np.full(im.shape[:2], 255, np.uint8)]) for im in zoom]
    return np.hstack([rgba[0], gap, rgba[1]])


# ---------------------------------------------------------------- PSR mode


def load_psr_section(run: Path, sample: str, variant: str = "base") -> dict:
    """The per-section outputs of msq.psr.section for one sample, all on the segmentation grid."""
    d = run / "sections" / variant / sample
    if not (d / "section.json").is_file():
        raise ValueError(f"no PSR section {sample!r} in {run / 'sections' / variant}")
    info = json.loads((d / "section.json").read_text(encoding="utf-8"))
    f = int(info["params"]["downsample"])
    rgb_ = tifffile.imread(d / f"rgb_ds{f}.tif")
    return {"f": f, "grid_um": float(info["grid_um"]), "rgb": rgb_, "labels": tifffile.imread(d / f"labels_ds{f}.tif"),
            "classes": tifffile.imread(d / f"classes_ds{f}.tif"), "section": tifffile.imread(d / f"section_ds{f}.tif") > 0}


def psr_red_outside(sec: dict) -> np.ndarray:
    """Section grid pixels whose main class is red, in the main section, inside no outline."""
    return (sec["classes"].argmax(-1) == 1) & sec["section"] & (sec["labels"] == 0)


def psr_window_share(sec: dict, sl: tuple) -> float:
    """Collagen outside the outlines in the window, as section.py defines it: red outside / (red + yellow), over the
    window's main-section pixels, on the section grid."""
    main = sec["classes"][sl].argmax(-1)[sec["section"][sl]]
    return float(psr_red_outside(sec)[sl].sum() / max(1, ((main == 1) | (main == 2)).sum()))


def psr_pick_window(sec: dict, size: int, pick: str) -> tuple[int, int]:
    """Top-left (section grid pixels) of the window at the pick's quantile of collagen outside the outlines among the
    windows that are mostly main section and hold outlines."""
    H, W = sec["labels"].shape
    stride = max(size // 5, 1)
    cands = []
    for y0 in range(0, H - size + 1, stride):
        for x0 in range(0, W - size + 1, stride):
            sl = np.s_[y0:y0 + size, x0:x0 + size]
            if sec["section"][sl].mean() >= PSR_MIN_SECTION and (sec["labels"][sl] > 0).mean() >= 0.05:
                cands.append((psr_window_share(sec, sl), y0, x0))
    if not cands:
        raise ValueError(f"no window of {size} grid pixels lies in the main section and holds outlines; "
                         "give one with at = [y, x]")
    cands.sort()
    _, y0, x0 = cands[min(int(PICKS[pick] * len(cands)), len(cands) - 1)]
    return y0, x0


def psr_images(sec: dict, y0: int, x0: int, size: int) -> dict[str, np.ndarray]:
    """The four PSR crops of one window (section grid y0, x0, size), drawn PSR_ZOOM times larger."""
    z = PSR_ZOOM
    sl = np.s_[y0:y0 + size, x0:x0 + size]

    def up(a: np.ndarray) -> np.ndarray:
        return np.kron(a, np.ones((z, z) + (1,) * (a.ndim - 2), a.dtype))

    colour = up(sec["rgb"][sl])
    dim = (colour * 0.35).astype(np.uint8)
    main = up(sec["classes"][sl].argmax(-1))
    classes = dim.copy()
    classes[main == 1], classes[main == 2] = rgb(PSR_COLOURS["red"]), rgb(PSR_COLOURS["yellow"])
    edges = outlines(sec["labels"][sl], z)
    outl = colour.copy()
    paint(outl, edges, {}, PSR_COLOURS["outline"])
    outside = dim.copy()
    red_in = (main == 1) & ~up(psr_red_outside(sec)[sl])
    outside[red_in] = rgb(PSR_COLOURS["inside"])
    outside[up(psr_red_outside(sec)[sl])] = rgb(PSR_COLOURS["outside"])
    paint(outside, edges, {}, PSR_COLOURS["outline"])
    out = dict(zip(PSR_NAMES, (colour, classes, outl, outside)))
    for img in out.values():
        scale_bar(img, 50 / sec["grid_um"] * z)
    return out


def make_psr_crops(cfg: dict) -> dict:
    if cfg["scan"] is not None:
        raise ValueError("scan is not used in PSR mode (the crops come from the run's downsampled outputs)")
    lo, hi = WINDOW_UM
    if not lo <= cfg["window_um"] <= hi:
        raise ValueError(f"window_um must be {lo} to {hi} (the README image rules), not {cfg['window_um']}")
    sec = load_psr_section(Path(cfg["run"]).expanduser(), cfg["sample"], cfg["variant"])
    f, H, W = sec["f"], *sec["labels"].shape
    size = int(round(cfg["window_um"] / sec["grid_um"]))
    if size > min(H, W):
        raise ValueError(f"the section image is smaller than a {cfg['window_um']} um window")
    if cfg["at"] is not None:
        y0, x0 = (min(max(int(v) // f, 0), lim - size) for v, lim in zip(cfg["at"], (H, W)))
    else:
        y0, x0 = psr_pick_window(sec, size, cfg["pick"])
    out = Path(cfg["out"]).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    for name in NAMES + PSR_NAMES:
        (out / name).unlink(missing_ok=True)
    images = psr_images(sec, y0, x0, size)
    for name, img in images.items():
        save_png(img, out / name)
    return {"y0": y0 * f, "x0": x0 * f, "size": size * f, "um": size * sec["grid_um"], "written": sorted(images),
            "collagen_outside": psr_window_share(sec, np.s_[y0:y0 + size, x0:x0 + size])}


# ---------------------------------------------------------------- main


def make_crops(cfg: dict) -> dict:
    """Write the crops for one section; returns the choices made (window, size, fiber, local median)."""
    cfg = {**DEFAULTS, **cfg}
    if cfg["assay"] == "psr":
        return make_psr_crops(cfg)
    lo, hi = WINDOW_UM
    if not lo <= cfg["window_um"] <= hi:
        raise ValueError(f"window_um must be {lo} to {hi} (the README image rules), not {cfg['window_um']}")
    sec = load_section(Path(cfg["run"]).expanduser(), cfg["sample"],
                       Path(cfg["scan"]).expanduser() if cfg["scan"] else None)
    f = sec["f"]
    size = int(round(cfg["window_um"] / sec["px_um"] / f)) * f
    if cfg["at"] is not None:
        H, W = sec["gfp"].shape
        y0, x0 = (min(max(int(v), 0), lim - size) // f * f for v, lim in zip(cfg["at"], (H, W)))
    else:
        y0, x0 = pick_window(sec, size, cfg["pick"])
    fib = boxes(sec)
    inside = fib[inside_window(fib, y0, x0, size)]
    if cfg["fiber"] is not None:
        label = int(cfg["fiber"])
        if sec["status"].get(label) != "included":
            raise ValueError(f"fiber {label} is not an included fiber of this section")
    else:
        label = pick_fiber(inside if inside["include"].any() else fib)

    out = Path(cfg["out"]).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        (out / name).unlink(missing_ok=True)
    images = window_images(sec, y0, x0, size)
    if not cfg["calls"]:
        images.pop("calls.png", None)
    images["ring_sectors.png"] = ring_image(sec, label)
    for name, img in images.items():
        save_png(img, out / name)
    return {"y0": y0, "x0": x0, "size": size, "um": size * sec["px_um"], "fiber": label, "written": sorted(images),
            "local_median": float(inside.loc[inside["include"], "dys_membrane_specific"].median())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path, help="e.g. demo/crops.local.toml (copy demo/crops.example.toml)")
    args = ap.parse_args()
    try:
        cfg = read_config(args.config)
        got = make_crops(cfg)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    where = f"{cfg['sample']}: window at (y={got['y0']}, x={got['x0']}) px, {got['um']:.0f} um wide; "
    print(where + (f"collagen outside the outlines {100 * got['collagen_outside']:.1f}% of red plus yellow"
                   if cfg["assay"] == "psr" else f"ring fiber {got['fiber']}; local median {got['local_median']:.0f}"))
    print(f"wrote {', '.join(got['written'])} to {Path(cfg['out']).expanduser()}")
    print(f'caption: "crop from one of the lab\'s mouse muscle sections; {got["um"]:.0f} µm wide, scale bar 50 µm"')


if __name__ == "__main__":
    main()
