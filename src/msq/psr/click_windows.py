"""Windows for the click review: one random tissue window per section, with the outlines as a separate overlay.

For each section folder in RUN/sections/<variant>/ it picks a window of SIZE_UM inside the main section (at least
EDGE_UM from its edge, at least MIN_TISSUE stained), and writes to OUT:
    wNN.jpg            the window, normalised to the glass colour
    wNN_outlines.png   the fiber outlines on a transparent background (same size)
    wNN_labels.tif     the outline labels at full resolution, for scoring the clicks
    windows_key.csv    which section each window comes from (keep it away from the reviewers: the page is blind)
    windows.json       what the page needs: window ids, size, pixel size, number of outlines

Windows are numbered in a shuffled order, so genotype cannot be read from the sequence.

The review is done by eye, by one reviewer. The reviewer's results are not stored in the repository.

Usage:
    python -m msq.psr.click_windows RUN --out OUT [--variant base] [--seed 7]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from scipy import ndimage as ndi
from skimage.segmentation import find_boundaries

from msq.core.objects import upsample_labels
from msq.core.report.images import glass_normalised
from msq.psr import checks
from msq.psr import section as sec

SIZE_UM = 250.0
EDGE_UM = 150.0
MIN_TISSUE = 0.8
OUTLINE_RGBA = (0, 229, 255, 255)


def pick_window(sect: np.ndarray, classes: np.ndarray, grid_um: float, rng: np.random.Generator,
                size_um: float = SIZE_UM, edge_um: float = EDGE_UM, min_tissue: float = MIN_TISSUE,
                tries: int = 2000) -> tuple[int, int] | None:
    """Top-left corner (grid pixels) of a random window inside the section, away from its edge and mostly stained."""
    k = int(round(size_um / grid_um))
    inner = ndi.binary_erosion(sect, iterations=max(1, int(round(edge_um / grid_um))))
    stained = (classes[..., sec.RED].astype(float) + classes[..., sec.YELLOW]) / 100
    ys, xs = np.nonzero(inner)
    if ys.size == 0:
        return None
    for _ in range(tries):
        i = rng.integers(ys.size)
        y0, x0 = ys[i] - k // 2, xs[i] - k // 2
        if y0 < 0 or x0 < 0 or y0 + k > sect.shape[0] or x0 + k > sect.shape[1]:
            continue
        if inner[y0:y0 + k, x0:x0 + k].all() and stained[y0:y0 + k, x0:x0 + k].mean() >= min_tissue:
            return int(y0), int(x0)
    return None


def outline_overlay(labels_full: np.ndarray) -> np.ndarray:
    edges = find_boundaries(labels_full, mode="inner")
    edges = ndi.binary_dilation(edges) & (labels_full > 0)
    out = np.zeros(labels_full.shape + (4,), np.uint8)
    out[edges] = OUTLINE_RGBA
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--variant", default="base")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    samples = checks.read_samples(args.run / "samples.csv").set_index("sample")
    folders = sorted(d for d in (args.run / "sections" / args.variant).iterdir()
                     if not d.name.endswith(".partial") and (d / "section.json").exists())
    order = rng.permutation(len(folders))
    key, windows = [], []
    for number, idx in enumerate(order, start=1):
        d = folders[idx]
        info = json.loads((d / "section.json").read_text())
        f, grid_um = info["params"]["downsample"], info["grid_um"]
        labels = tifffile.imread(d / f"labels_ds{f}.tif")
        classes = tifffile.imread(d / f"classes_ds{f}.tif")
        sect = tifffile.imread(d / f"section_ds{f}.tif") > 0
        corner = pick_window(sect, classes, grid_um, rng)
        if corner is None:
            print(f"{d.name}: no window found, skipped")
            continue
        gy, gx = corner
        k = int(round(SIZE_UM / grid_um))
        reader, frame, px_um = sec.open_scan(Path(samples.loc[d.name, "path"]))
        try:
            rgb = np.array(frame[gy * f:(gy + k) * f, gx * f:(gx + k) * f])[..., ::-1]  # copy before the file closes
        finally:
            del frame
            reader.close()
        bg = np.array(info["background_rgb"], np.float32)
        img = glass_normalised(rgb, bg)
        lab = upsample_labels(labels[gy:gy + k, gx:gx + k], f)
        wid = f"w{number:02d}"
        Image.fromarray(img).save(args.out / f"{wid}.jpg", quality=88)
        Image.fromarray(outline_overlay(lab)).save(args.out / f"{wid}_outlines.png", optimize=True)
        tifffile.imwrite(args.out / f"{wid}_labels.tif", lab, compression="zlib")
        n_outlines = int(len(np.unique(lab[lab > 0])))
        windows.append({"id": wid, "size_px": int(img.shape[0]), "size_um": round(img.shape[0] * px_um, 1),
                        "px_um": px_um, "n_outlines": n_outlines})
        key.append({"window": wid, "sample": d.name, "group": samples.loc[d.name, "group"],
                    "animal": samples.loc[d.name, "animal"], "y0_full": gy * f, "x0_full": gx * f,
                    "n_outlines": n_outlines})
        print(wid, d.name, n_outlines, "outlines")
    pd.DataFrame(key).to_csv(args.out / "windows_key.csv", index=False)
    (args.out / "windows.json").write_text(json.dumps(windows, indent=1))


if __name__ == "__main__":
    main()
