"""Freezing artifacts per scan: how much of the fiber area is riddled with small white holes.

Freezing artefacts show up in PSR as small white (unstained) spots inside yellow fibers. They matter for
segmentation: inputs that turn white dark make holey fibers look speckled and Cellpose misses them. This survey needs
no segmentation, so it can screen every scan before a run.

The survey only flags. Tiles with freezing artifacts are not masked out of the collagen readout, folds, dust, edge
staining and tendon are not masked either, and no section is excluded. summary.py carries holey_area into
sections.csv as freezing_artifact_area, and the R figures mark sections above a 0.3 share.

Per scan, on a 3x downsampled grid (0.65 um/px):
  white      = every channel passes more than WHITE_T of the glass light
  hole       = a white spot of HOLE_UM2 in area whose 2 px ring is at least RING_YELLOW yellow (inside a fiber)
  hole_frac  = hole area / (yellow + hole area) in the main section
  holey_area = share of the fiber area that lies in 200 um tiles where holes take more than TILE_HOLEY of it

Usage:
    python -m msq.psr.hole_survey SCAN.nd2 --sample NAME --out OUT.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from msq.core.objects import upsample_labels
from msq.core.tissue import main_section
from msq.psr import section as sec

F = 3
WHITE_T = 0.8
HOLE_UM2 = (3.0, 150.0)
RING_YELLOW = 0.7
TILE_UM = 200.0
TILE_HOLEY = 0.02


def hole_masks(t: np.ndarray, yellow: np.ndarray, grid_um: float) -> np.ndarray:
    """Holes inside fibers: small white spots mostly surrounded by yellow. t: transmittance (H, W, 3)."""
    white = t.min(-1) > WHITE_T
    comp, n = ndi.label(white)
    if n == 0:
        return np.zeros_like(white)
    ids = np.arange(1, n + 1)
    area = ndi.sum(white, comp, ids) * grid_um ** 2
    small = ids[(area >= HOLE_UM2[0]) & (area <= HOLE_UM2[1])]
    cand = np.where(np.isin(comp, small), comp, 0)
    grown = ndi.grey_dilation(cand, size=5)
    ring = (grown > 0) & (cand == 0) & ~white
    share = ndi.mean(yellow[ring], grown[ring], small) if small.size else np.array([])
    keep = small[np.nan_to_num(share) >= RING_YELLOW]
    return np.isin(comp, keep)


def tile_share(holes: np.ndarray, yellow: np.ndarray, grid_um: float) -> float:
    """Share of the fiber area in tiles where holes are more than TILE_HOLEY of (yellow + holes)."""
    k = max(1, int(round(TILE_UM / grid_um)))
    h, w = (holes.shape[0] // k) * k, (holes.shape[1] // k) * k
    hs = holes[:h, :w].reshape(h // k, k, w // k, k).sum((1, 3)).astype(float)
    ys = yellow[:h, :w].reshape(h // k, k, w // k, k).sum((1, 3)).astype(float)
    tissue = hs + ys
    frac = np.divide(hs, tissue, out=np.zeros_like(hs), where=tissue > 0)
    return float(tissue[frac > TILE_HOLEY].sum() / max(1.0, tissue.sum()))


def survey(frame: np.ndarray, px_um: float) -> tuple[dict, np.ndarray]:
    p = sec.Params()
    small6 = sec.downsample_rgb(frame, 6)
    bg, tissue = sec.background_colour(small6, px_um * 6, p.bg_margin_um)
    section6 = main_section(tissue, px_um * 6, p.section_close_um)
    small = sec.downsample_rgb(frame, F)
    grid_um = px_um * F
    sect = upsample_labels(section6, 2, small.shape[:2])  # the 6x grid can be one 3x row or column short
    t = np.clip(small / bg, 0, 1)
    yellow = (sec.classify(np.clip(np.rint(small), 0, 255).astype(np.uint8), bg, p) == sec.YELLOW) & sect
    holes = hole_masks(t, yellow, grid_um) & sect
    n_holes = ndi.label(holes)[1]
    fiber_px = yellow.sum() + holes.sum()
    stats = {"background_rgb": [round(float(v), 2) for v in bg], "n_holes": int(n_holes),
             "holes_per_mm2_fiber": float(n_holes / max(1e-9, fiber_px * grid_um ** 2 / 1e6)),
             "hole_frac": float(holes.sum() / max(1, fiber_px)), "holey_area": tile_share(holes, yellow, grid_um)}
    return stats, holes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scan", type=Path)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    reader, frame, px_um = sec.open_scan(args.scan)
    try:
        stats, _ = survey(frame, px_um)
    finally:
        del frame
        reader.close()
    args.out.write_text(json.dumps({"sample": args.sample, **stats}))
    print(args.sample, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in stats.items() if k != "background_rgb"})


if __name__ == "__main__":
    main()
