"""Overlay report: what "collagen outside the fibers" counts, shown on the tissue with layers to switch on and off.

For each example section: the whole section on the 1.3 um grid, and windows of WINDOW_UM at half the scan's
resolution (0.43 um/px), with four layers:
    fiber outlines                      Cellpose-SAM outlines
    collagen counted                    red pixels outside every outline (the numerator of the readout)
    red inside outlines, not counted    red pixels inside an outline
    unstained                           glass, freezing-artifact holes, tears (left out of numerator and denominator)
Each view lists its own numbers: collagen outside the fibers, red inside outlines and total red, all as a share of
red + yellow pixels.

Writes one self-contained HTML file (open it in a browser).

Usage:
    python -m msq.psr.overlay_report RUN --samples S1 S2 ... --out REPORT.html --cohort NAME [--windows 2]
        [--scans-root DIR]
RUN holds sections/base/<sample>/ (section.py) and samples.csv; --scans-root replaces the scans folder recorded in
samples.csv when the report is built on another machine.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
from skimage.segmentation import find_boundaries

from msq.core import report
from msq.core.objects import upsample_labels
from msq.core.report import images
from msq.core.report.images import glass_normalised, png_uri, rgba  # noqa: F401  (png_uri and rgba re-exported)
from msq.psr import checks, click_windows, paths
from msq.psr import section as sec
from msq.psr.settings import load_cohort

TEMPLATE = paths.CODE / "overlay_report_template.html"
WINDOW_UM = 500.0
DISPLAY_STEP = 2          # windows are shown at every 2nd full-resolution pixel (0.43 um/px)
OVERVIEW_MAX_PX = 1600
LAYERS = {  # name: (label, RGBA)
    "outlines": ("Fiber outlines", (0, 229, 255, 255)),
    "counted": ("Collagen counted (outside fibers)", (47, 107, 255, 190)),
    "inside": ("Red inside outlines (not counted)", (0, 210, 106, 220)),
    "unstained": ("Unstained (holes, tears)", (40, 40, 40, 170)),
}


def jpeg_uri(rgb: np.ndarray, quality: int = 86) -> str:
    return images.jpeg_uri(rgb, quality, optimize=False)


def layer_masks(classes: np.ndarray, labels: np.ndarray, inside_section: np.ndarray) -> dict[str, np.ndarray]:
    """Boolean mask per layer, all on the same grid."""
    red = (classes == sec.RED) & inside_section
    return {"outlines": find_boundaries(labels, mode="inner"),
            "counted": red & (labels == 0),
            "inside": red & (labels > 0),
            "unstained": (classes == sec.BG) & inside_section}


def shares(classes: np.ndarray, labels: np.ndarray, inside_section: np.ndarray) -> dict[str, float]:
    red = (classes == sec.RED) & inside_section
    yellow = (classes == sec.YELLOW) & inside_section
    denom = max(1, int(red.sum() + yellow.sum()))
    return {"collagen_outside": float((red & (labels == 0)).sum() / denom),
            "red_inside": float((red & (labels > 0)).sum() / denom),
            "red_total": float(red.sum() / denom)}


def window_view(frame: np.ndarray, info: dict, labels: np.ndarray, section_mask: np.ndarray, corner: tuple[int, int],
                k: int, title: str) -> dict:
    """One window: full-resolution classes, labels upsampled from the grid, shown at every DISPLAY_STEP-th pixel."""
    f = info["params"]["downsample"]
    gy, gx = corner
    rgb = np.array(frame[gy * f:(gy + k) * f, gx * f:(gx + k) * f])[..., ::-1]
    bg = np.array(info["background_rgb"], np.float32)
    classes = sec.classify(rgb, bg, sec.Params(**info["params"]))
    lab = upsample_labels(labels[gy:gy + k, gx:gx + k], f)
    inside = upsample_labels(section_mask[gy:gy + k, gx:gx + k], f)
    stats = shares(classes, lab, inside)
    stats["fibers"] = int(len(np.unique(lab[lab > 0])))
    s = DISPLAY_STEP
    shown = glass_normalised(rgb[::s, ::s], bg)
    masks = layer_masks(classes[::s, ::s], lab[::s, ::s], inside[::s, ::s])
    return {"title": title, "kind": "window", "size_um": round(k * info["grid_um"], 1), "base": jpeg_uri(shown),
            "layers": {n: png_uri(rgba(m, LAYERS[n][1])) for n, m in masks.items()}, "stats": stats, "boxes": []}


def overview_view(folder: Path, info: dict, labels: np.ndarray, section_mask: np.ndarray, boxes: list[dict],
                  title: str) -> dict:
    """The whole section on the segmentation grid: a pixel takes the class that covers at least half of it."""
    f = info["params"]["downsample"]
    rgb = tifffile.imread(folder / f"rgb_ds{f}.tif")
    frac = tifffile.imread(folder / f"classes_ds{f}.tif")
    classes = np.full(labels.shape, sec.OTHER, np.uint8)
    for c in (sec.BG, sec.RED, sec.YELLOW):
        classes[frac[..., c] >= 50] = c
    step = max(1, int(np.ceil(max(labels.shape) / OVERVIEW_MAX_PX)))
    masks = layer_masks(classes, labels, section_mask)
    masks["outlines"] = find_boundaries(labels, mode="inner")
    stats = {"collagen_outside": info["collagen_outside_fraction"], "red_total": info["red_fraction"],
             "red_inside": info["red_fraction"] - info["collagen_outside_fraction"],
             "fibers": int(info.get("n_objects_in_section", info["n_objects"]))}
    return {"title": title, "kind": "overview", "size_um": None, "base": jpeg_uri(rgb[::step, ::step]),
            "layers": {n: png_uri(rgba(m[::step, ::step], LAYERS[n][1])) for n, m in masks.items()},
            "stats": stats, "boxes": boxes}


def section_views(folder: Path, scan: Path, label: str, n_windows: int, rng: np.random.Generator) -> list[dict]:
    info = json.loads((folder / "section.json").read_text(encoding="utf-8"))
    f, grid_um = info["params"]["downsample"], info["grid_um"]
    labels = tifffile.imread(folder / f"labels_ds{f}.tif")
    classes_frac = tifffile.imread(folder / f"classes_ds{f}.tif")
    section_mask = tifffile.imread(folder / f"section_ds{f}.tif") > 0
    k = int(round(WINDOW_UM / grid_um))
    corners = []
    for _ in range(n_windows * 10):
        c = click_windows.pick_window(section_mask, classes_frac, grid_um, rng, size_um=WINDOW_UM)
        if c and all(abs(c[0] - a) >= k or abs(c[1] - b) >= k for a, b in corners):
            corners.append(c)
        if len(corners) == n_windows:
            break
    letters = "ABCDEFGH"
    h, w = labels.shape
    boxes = [{"label": letters[i], "x": c[1] / w, "y": c[0] / h, "w": k / w, "h": k / h} for i, c in enumerate(corners)]
    views = [overview_view(folder, info, labels, section_mask, boxes, f"{label}: whole section")]
    reader, frame, _ = sec.open_scan(scan)
    try:
        for i, c in enumerate(corners):
            views.append(window_view(frame, info, labels, section_mask, c, k, f"{label}: window {letters[i]}"))
    finally:
        del frame
        reader.close()
    return views


SUBTITLE = ("What \"collagen outside the fibers\" counts, on three example sections (one per genotype, the animal "
            "closest to its genotype's mean). Switch the layers to compare them with the tissue.")


def render(views: list[dict], out: Path, subtitle: str = SUBTITLE, title: str = "Collagen Overlay Check",
           layers: dict | None = None) -> Path:
    """The page; views may carry their own "rows" for the numbers table, and layers their own names and colours."""
    layers = [{"id": n, "label": lab, "rgba": list(c)} for n, (lab, c) in (layers or LAYERS).items()]
    return report.render(TEMPLATE, out, data={"views": views, "layers": layers},
                         slots={"TITLE": title, "SUBTITLE": subtitle})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--samples", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cohort", required=True, help="the dataset's cohort in cohorts.local.toml")
    ap.add_argument("--windows", type=int, default=2)
    ap.add_argument("--scans-root", type=Path, help="local scan folder replacing the one in samples.csv")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--subtitle", default=SUBTITLE, help="line under the title: which sections and why")
    args = ap.parse_args()
    cohort = load_cohort(args.cohort)
    samples = checks.read_samples(args.run / "samples.csv").set_index("sample")
    rng = np.random.default_rng(args.seed)
    views = []
    for s in args.samples:
        row = samples.loc[s]
        scan = Path(row["path"])
        if args.scans_root:
            parts = scan.parts
            group_idx = next(i for i, p in enumerate(parts) if any(k in p.lower() for k in cohort.folders))
            scan = args.scans_root.joinpath(*parts[group_idx:])
        label = f"{cohort.labels[row['group']]} {row['animal']} sec {int(row['section'])}"
        views += section_views(args.run / "sections" / "base" / s, scan, label, args.windows, rng)
        print(s, "done", flush=True)
    out = render(views, args.out, args.subtitle)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
