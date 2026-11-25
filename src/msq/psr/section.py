"""One whole PSR section: normalise, classify pixels, segment fibers, and count what lies inside and around each fiber.

Steps:
1. Read the stitched RGB scan through the reader's memory map, in strips, flipped from the file's BGR order;
   block-mean downsample for segmentation (6x: 0.2167 -> 1.30 um/px).
2. Background colour = median of empty glass well away from the tissue. Dividing by it gives transmittance. This
   corrects exposure, gain, lamp and white balance differences between scans only if the camera responds linearly to
   light. That is assumed here; it was checked on synthetic images only.
3. Pixel classes at full resolution with Hildyard et al.'s rule (protocols.io 81wgb6j1olpk) on the normalised
   colour, as ImageJ computes it: saturation 0-40 = background, hue < 23 or >= 234 = red (collagen),
   23-50 = yellow (fiber), the rest = other.
4. Fibers: Cellpose-SAM on the fiber signal (3 x green - 2 x red transmittance, large white areas set dark) of the
   downsampled image.
5. Per fiber, three zones on the segmentation grid: the core (the mask eroded by edge_um), the edge band (mask minus
   core) and the ring just outside (edge_um wide, each ring pixel owned by the nearest fiber). Class counts per zone
   come from the full-resolution classes, so "red inside a fiber" can be told apart from "outline slightly over the
   collagen".

Readouts, both over the main section only (the largest tissue piece, msq.core.tissue.main_section; other pieces and
the glass are not measured):
- collagen_outside_fraction, the primary readout, "collagen outside the fiber outlines": red pixels in the main
  section outside every Cellpose-SAM outline, divided by the red plus yellow pixels in the main section. All red
  outside the outlines counts, whatever it is (perimysium, epimysium, tendon, vessel walls). Red inside any outline
  does not count, including outlines of objects that are not fibers, so this readout depends on the segmentation.
- red_fraction, a secondary readout: all red pixels in the main section (inside or outside outlines), divided by the
  red plus yellow pixels in the main section.

Nothing is masked. Folds, dust, edge staining and tendon stay in the counts, and collagen is counted over the whole
main section. Sections with freezing artifacts are not excluded: hole_survey.py flags the affected tiles, and the R
figures mark sections where more than 0.3 of the fiber area lies in such tiles.

Usage:
    python -m msq.psr.section SCAN.nd2 --out DIR --sample NAME [--variant base]

Writes DIR/ (built as DIR.partial and renamed when complete, by msq.core.section): objects.csv (one row per
segmented object), section.json (background, totals, coverage, settings, run time, peak memory, code version, Cellpose
weights), labels_ds<f>.tif, rgb_ds<f>.tif (normalised colour), classes_ds<f>.tif (percent of each
class per downsampled pixel: background, red, yellow, other), section_ds<f>.tif (main-section mask),
hue_sat_hist.npz (main-section pixels counted by hue and saturation byte).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.measure import block_reduce

from msq.core import images, nd2
from msq.core.objects import joint_counts, region_rows, upsample_labels
from msq.core.section import Layout, Outputs, Prepared, run_section
from msq.core.segment import Segmenter, cellpose_for
from msq.core.tissue import main_section, off_tissue, otsu_tissue
from msq.core.zones import zones
from msq.psr import settings

BG, RED, YELLOW, OTHER = 0, 1, 2, 3
CLASS_NAMES = ("bg", "red", "yellow", "other")


@dataclass(frozen=True)
class Params:
    downsample: int = 6              # segmentation grid: 0.2167 -> 1.30 um/px
    seg_input: str = "fiber"         # see segmentation_input: "fiber", "fiber_raw", "yellow", "green_t", "green_od"
    white_t: float = 0.8             # "fiber": white = every channel passes more than this share of the light ...
    white_max_um2: float = 150.0     # ... and white areas larger than this (glass, tears, lumens) are set dark
    cp_flow_threshold: float = 0.4
    cp_cellprob_threshold: float = 0.0
    cp_batch_size: int = 16
    edge_um: float = 1.5             # width of the edge band inside and of the ring outside each fiber
    bg_margin_um: float = 200.0      # background glass is taken at least this far from the tissue
    sat_background: int = 40         # Hildyard: saturation 0-40 is unstained
    hue_red_below: int = 23          # red: hue < 23 ...
    hue_red_from: int = 234          # ... or hue >= 234
    hue_yellow_to: int = 50          # yellow: 23 <= hue <= 50
    section_close_um: float = 50.0   # gaps up to this size are bridged when finding the main section


VARIANTS = {
    "base": {},
    "green": {"seg_input": "green_t"},   # the first review's input, for comparison
    "yellow": {"seg_input": "yellow"},   # the second review's input, for comparison
    "fiber_raw": {"seg_input": "fiber_raw"},  # 3 green - 2 red without darkening large white areas
    "od": {"seg_input": "green_od"},
    "ds4": {"downsample": 4},
    "ds8": {"downsample": 8},
    "flow06": {"cp_flow_threshold": 0.6},
}


def params_for(variant: str) -> Params:
    return replace(Params(), **VARIANTS[variant])


# ---------------------------------------------------------------- reading


def iter_strips(frame: np.ndarray, f: int, rows: int = images.STRIP_ROWS_RGB) -> Iterator[tuple[int, int, np.ndarray]]:
    """(y0, y1, RGB uint8 strip) over the image trimmed to a multiple of f; frame is in the file's order (BGR)."""
    return images.iter_strips(frame, f, rows, flip=settings.FILE_ORDER == "BGR")


def downsample_rgb(frame: np.ndarray, f: int, rows: int = images.STRIP_ROWS_RGB) -> np.ndarray:
    """Block-mean RGB (float32) on the segmentation grid."""
    return images.block_mean_rgb(frame, f, rows, flip=settings.FILE_ORDER == "BGR")


def open_scan(path: Path):
    """The ND2 reader and its memory-mapped frame (Y, X, 3, file order) plus the pixel size in um."""
    scan = nd2.open_scan(path)
    return scan.reader, scan.frame.reshape(scan.sizes["Y"], scan.sizes["X"], 3), scan.px_um


# ---------------------------------------------------------------- background and tissue


def saturation(rgb: np.ndarray) -> np.ndarray:
    mx = rgb.max(-1)
    return np.where(mx > 0, (mx - rgb.min(-1)) / np.maximum(mx, 1e-6), 0.0)


def background_colour(small: np.ndarray, px_um: float, margin_um: float) -> tuple[np.ndarray, np.ndarray]:
    """Background RGB from glass far from the tissue, and the tissue mask (both on the downsampled grid).

    Tissue is told from glass by saturation: the glass is grey to slightly blue, stained tissue is far more
    saturated. Pixels that are all zero (unscanned) count as neither."""
    valid = small.max(-1) > 0
    tissue = otsu_tissue(ndi.gaussian_filter(saturation(small), 2), valid)
    off = off_tissue(tissue, valid, margin_um, px_um)
    if off.sum() < 100:
        raise ValueError("too few glass pixels away from the tissue to estimate the background colour")
    return np.median(small[off], axis=0), tissue


# ---------------------------------------------------------------- colour classes


def hue_sat_bytes(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hue and saturation as ImageJ's HSB stack gives them (java.awt.Color.RGBtoHSB, truncated to 0-255)."""
    rgb = rgb.astype(np.float64)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx, mn = rgb.max(-1), rgb.min(-1)
    d = mx - mn
    safe = np.where(d > 0, d, 1.0)
    h = np.where(mx == r, (g - b) / safe, np.where(mx == g, 2.0 + (b - r) / safe, 4.0 + (r - g) / safe))
    h = np.where(d > 0, (h / 6.0) % 1.0, 0.0)
    s = np.where(mx > 0, d / np.where(mx > 0, mx, 1.0), 0.0)
    return (h * 255).astype(np.uint8), (s * 255).astype(np.uint8)


def classify(rgb: np.ndarray, bg: np.ndarray, p: Params, return_hue_sat: bool = False):
    """Class per pixel of an RGB uint8 image, after normalising to the background colour (glass = 255).
    With return_hue_sat, also the hue and saturation bytes the classes were cut from."""
    norm = np.clip(np.rint(rgb.astype(np.float32) / bg.astype(np.float32) * 255), 0, 255)
    h, s = hue_sat_bytes(norm)
    out = np.full(h.shape, OTHER, np.uint8)
    out[(h < p.hue_red_below) | (h >= p.hue_red_from)] = RED
    out[(h >= p.hue_red_below) & (h <= p.hue_yellow_to)] = YELLOW
    out[s <= p.sat_background] = BG
    return (out, h, s) if return_hue_sat else out


# ---------------------------------------------------------------- segmentation


def dark_large_white(img: np.ndarray, t: np.ndarray, grid_um: float, white_t: float, max_um2: float) -> np.ndarray:
    """Set unstained white areas larger than max_um2 (glass, tears, lumens) to 0, keeping smaller ones (the freezing
    holes inside fibers, a few um across) as they are. White = every channel passes more than white_t of the light."""
    white = t.min(-1) > white_t
    comp, n = ndi.label(white)
    if n == 0:
        return img
    big = 1 + np.flatnonzero(ndi.sum(white, comp, range(1, n + 1)) * grid_um ** 2 > max_um2)
    out = img.copy()
    out[np.isin(comp, big)] = 0
    return out


def segmentation_input(small: np.ndarray, bg: np.ndarray, kind: str, grid_um: float = 1.3,
                       p: Params | None = None) -> np.ndarray:
    """The image Cellpose sees, from transmittance T (glass = 1).

    "fiber" (default) = 3 T_green - 2 T_red, with large white areas set to 0. Red collagen absorbs green but passes
    red, so it goes dark (pale pink collagen partly); yellow cytoplasm and white holes inside fibers stay bright; glass,
    tears and lumens (white areas over white_max_um2) go dark so fibers next to them keep an edge. Compared on crops of
    sections:
      - "green_t" (T_green, the first input) put outlines through the middle of the collagen between fibers, so the
        outlines held a large share of collagen;
      - "yellow" (T_green - T_blue) gave the tightest outlines but turns the white freezing-artifact holes inside
        fibers black, so whole areas of holey fibers went unsegmented;
      - 3 T_green - 2 T_red alone handled the holes but lost fibers next to glass; with large white set to 0 both
        problems were small."""
    p = p or Params()
    t = np.clip(small / bg, 0, 1).astype(np.float32)
    if kind == "fiber":
        img = np.clip(3 * t[..., 1] - 2 * t[..., 0], 0, 1)
        return dark_large_white(img, t, grid_um, p.white_t, p.white_max_um2)
    if kind == "fiber_raw":
        return np.clip(3 * t[..., 1] - 2 * t[..., 0], 0, 1)
    if kind == "yellow":
        return np.clip(t[..., 1] - t[..., 2], 0, 1)
    if kind == "green_t":
        return t[..., 1]
    if kind == "green_od":
        return (-np.log10(np.clip(t[..., 1], 1 / 255, 1))).astype(np.float32)
    raise ValueError(f"unknown segmentation input {kind!r}")


# ---------------------------------------------------------------- counts


def add_counts(counts: np.ndarray, labels_ds: np.ndarray, classes: np.ndarray, f: int, y0: int) -> None:
    """Add the class counts of a full-resolution strip (starting at row y0) to counts[label, class]."""
    rows = labels_ds[y0 // f:(y0 + classes.shape[0]) // f]
    counts += joint_counts(upsample_labels(rows, f, classes.shape), classes, *counts.shape)


def class_fractions(classes: np.ndarray, f: int) -> np.ndarray:
    """Percent of each class per f x f block (uint8, last axis = BG, RED, YELLOW, OTHER)."""
    return np.stack([np.round(100 * block_reduce((classes == c).astype(np.float32), (f, f), np.mean))
                     for c in (BG, RED, YELLOW, OTHER)], -1).astype(np.uint8)


# ---------------------------------------------------------------- per object


def object_table(labels: np.ndarray, section: np.ndarray, counts: dict[str, np.ndarray], px_ds: float,
                 f: int) -> pd.DataFrame:
    df = region_rows(labels, px_ds, f,
                     lambda rp: {"in_section": bool(section[int(rp.centroid[0]), int(rp.centroid[1])])})
    if df.empty:
        return df
    lab = df["label"].to_numpy()
    for zone, c in counts.items():
        sub = c[lab]
        for i, name in enumerate(CLASS_NAMES):
            df[f"{zone}_{name}"] = sub[:, i]
        total = sub.sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            df[f"red_frac_{zone}"] = np.where(total > 0, sub[:, RED] / total, np.nan)
            df[f"yellow_frac_{zone}"] = np.where(total > 0, sub[:, YELLOW] / total, np.nan)
    return df


# ---------------------------------------------------------------- one section

LAYOUT = Layout(seconds_in="seconds")  # one variant per folder, section.json holds the run time as seconds.total


@dataclass(frozen=True)
class PsrSection:
    """The section driver's view of one PSR scan (msq.core.section.run_section): its settings and names. The scan
    it is run on is (frame in file order, pixel size in um)."""

    p: Params
    sample: str
    source: str

    def params(self, variant: str) -> Params:
        return self.p

    def prepare(self, scan, variants) -> Prepared:
        frame, px_um = scan
        p, t0 = self.p, time.time()
        px_ds = px_um * p.downsample
        small = downsample_rgb(frame, p.downsample)
        bg, tissue = background_colour(small, px_ds, p.bg_margin_um)
        section = main_section(tissue, px_ds, p.section_close_um)
        return Prepared(SimpleNamespace(frame=frame, px_um=px_um, px_ds=px_ds, small=small, bg=bg, section=section,
                                        read_s=round(time.time() - t0, 1)))

    def run_variant(self, prepared: Prepared, variant: str, segment: Segmenter) -> Outputs:
        p, g = self.p, prepared.state
        f, px_ds, px_um, frame, small, bg, section = p.downsample, g.px_ds, g.px_um, g.frame, g.small, g.bg, g.section
        t_read = time.time()
        labels = segment(segmentation_input(small, bg, p.seg_input, px_ds, p)).astype(np.uint32)
        t_seg = time.time()

        k = max(1, int(round(p.edge_um / px_ds)))
        core, band, ring = zones(labels, k)
        n = int(labels.max())
        unsegmented = (section & (labels == 0)).astype(np.uint32)
        maps = {"core": core, "band": band, "ring": ring, "mask": labels, "section": section.astype(np.uint32),
                "unsegmented": unsegmented}
        counts = {name: np.zeros((n + 1 if name not in ("section", "unsegmented") else 2, 4), np.int64)
                  for name in maps}
        fractions = np.zeros(small.shape[:2] + (4,), np.uint8)
        hue_sat = np.zeros(256 * 256, np.int64)  # pixels of the main section by (hue, saturation)
        for y0, y1, strip in iter_strips(frame, f):
            classes, h, s = classify(strip, bg, p, return_hue_sat=True)
            for name, m in maps.items():
                add_counts(counts[name], m, classes, f, y0)
            fractions[y0 // f:y1 // f] = class_fractions(classes, f)
            inside = upsample_labels(section[y0 // f:y1 // f], f, strip.shape[:2])
            hue_sat += np.bincount(h[inside].astype(np.int64) * 256 + s[inside], minlength=256 * 256)
        t_count = time.time()

        objects = object_table(labels, section, {z: counts[z] for z in ("core", "band", "ring")}, px_ds, f)
        # All fractions are over the main section; "inside" means inside any outline, "outside" inside no outline.
        # collagen_outside_fraction is the primary readout (red outside every outline / (red + yellow)); red_fraction
        # is the secondary one (all red / (red + yellow)). Outlines of non-fiber objects also hide red from the first.
        sec_counts, uns_counts = counts["section"][1], counts["unsegmented"][1]
        red_yellow = max(1, sec_counts[RED] + sec_counts[YELLOW])
        in_section = objects["label"][objects["in_section"]].to_numpy() if len(objects) else np.array([], int)
        info = {
            "sample": self.sample, "source": self.source, "variant": variant, "params": asdict(p),
            "pixel_size_um": px_um, "grid_um": px_ds, "edge_px": k, "image_px": list(frame.shape[:2]),
            "background_rgb": [round(float(v), 2) for v in bg],
            "section_area_mm2": float(section.sum() * px_ds ** 2 / 1e6),
            "n_objects": n, "n_objects_in_section": int(len(in_section)),
            "section_class_px": dict(zip(CLASS_NAMES, map(int, sec_counts))),
            "red_fraction": float(sec_counts[RED] / red_yellow),             # all red / (red + yellow)
            "collagen_outside_fraction": float(uns_counts[RED] / red_yellow),  # red outside outlines / (red + yellow)
            "coverage_yellow": float(1 - uns_counts[YELLOW] / max(1, sec_counts[YELLOW])),
            "red_in_masks": float(1 - uns_counts[RED] / max(1, sec_counts[RED])),
            "unsegmented_class_px": dict(zip(CLASS_NAMES, map(int, uns_counts))),
            "seconds": {"read": g.read_s, "segment": round(t_seg - t_read, 1),
                        "classify_count": round(t_count - t_seg, 1)},
        }
        norm = np.clip(np.rint(small / bg * 255), 0, 255).astype(np.uint8)
        return Outputs(info, {
            "objects.csv": objects, f"labels_ds{f}.tif": labels, f"rgb_ds{f}.tif": (norm, {"photometric": "rgb"}),
            f"classes_ds{f}.tif": (fractions, {"photometric": "minisblack", "planarconfig": "contig"}),
            f"section_ds{f}.tif": section.astype(np.uint8),
            "hue_sat_hist.npz": {"counts": hue_sat.reshape(256, 256)}})  # [hue, saturation]


def run_frame(frame: np.ndarray, px_um: float, out_dir: Path, p: Params, sample: str, source: str,
              segment: Segmenter | None = None, variant: str = "") -> Path:
    """Process one stitched scan (frame in file order) and write its folder; returns out_dir. segment turns the
    segmentation input into labels (cached labels or a stand-in in tests); by default Cellpose-SAM with p's settings."""
    return run_section(PsrSection(p, sample, source), (frame, px_um), out_dir, [variant],
                       segment or cellpose_for(p), LAYOUT)


def run(path: Path, out_dir: Path, sample: str, variant: str = "base", source: str | None = None,
        segment: Segmenter | None = None) -> Path:
    reader, frame, px_um = open_scan(path)
    try:
        return run_frame(frame, px_um, out_dir, params_for(variant), sample, source or str(path), segment=segment,
                         variant=variant)
    finally:
        del frame
        reader.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scan", type=Path)
    ap.add_argument("--out", type=Path, required=True, help="section folder to write")
    ap.add_argument("--sample", required=True)
    ap.add_argument("--variant", default="base", choices=list(VARIANTS))
    ap.add_argument("--source", help="path to record as the scan's origin (default: the scan path)")
    args = ap.parse_args()
    out = run(args.scan, args.out, args.sample, args.variant, args.source)
    info = json.loads((out / "section.json").read_text())
    print(f"{args.sample} [{args.variant}]: {info['n_objects']} objects, coverage {info['coverage_yellow']:.3f}, "
          f"red fraction {info['red_fraction']:.3f}, outside outlines {info['collagen_outside_fraction']:.3f}, "
          f"{info['seconds']}, peak {info['peak_memory_gb']} GB")


if __name__ == "__main__":
    main()
