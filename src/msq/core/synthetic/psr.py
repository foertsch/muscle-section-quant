"""Synthetic PSR sections and H&E fields, and the fake ND2 header the psr tests use."""

from __future__ import annotations

import numpy as np
import pandas as pd

from msq.core.synthetic import ellipse, hex_grid, to_labels

DESCRIPTION = """Camera Name: Color Camera Nikon DS-Fi3
 Camera Settings:
Camera Type: Nikon DS-Fi3
Exposure: 11 ms
Gain: 1.0x
Sharpness: Medium
WB Red: 2.43
WB Blue: 3.16
 Microscope Settings:   Microscope: Ti2 Microscope
  Nikon Ti2, Illuminator(DIA) Iris intensity: 63.0
"""

IMAGING = {"exposure_ms": 11.0, "gain": 1.0, "wb_red": 2.43, "wb_blue": 3.16, "light": 63.0, "sharpness": "Medium",
           "acquired": "2026-03-30 14:49"}

HEADER = {"sizes": {"Y": 100, "X": 120, "S": 3}, "channels": ["Brightfield"], "px_um": 0.2167, "bits": 8,
          "imaging": IMAGING}


def header(**imaging) -> dict:
    """HEADER with some imaging settings changed."""
    return {**HEADER, "imaging": {**IMAGING, **imaging}}


GLASS = np.array([200.0, 190.0, 210.0])     # empty glass under a slightly blue lamp
PSR_YELLOW = np.array([190.0, 175.0, 40.0])  # PSR muscle fiber
PSR_RED = np.array([170.0, 30.0, 70.0])      # PSR collagen
PSR_BLUE = np.array([60.0, 60.0, 190.0])     # a fold: neither red nor yellow
HE_GLASS = np.array([238.0, 236.0, 240.0])
HE_GAP = np.array([205.0, 185.0, 200.0])     # pale space between fibers, still darker than glass
HE_FIBER = np.array([200.0, 110.0, 160.0])   # eosin
HE_NUCLEUS = np.array([80.0, 55.0, 125.0])   # hematoxylin


def synthetic_psr_section(seed: int, fibrosis: str = "low", size: int = 3600, f: int = 6) -> dict:
    """A seeded whole-section PSR scan as the reader returns it (BGR uint8), with the fibers' labels on the
    segmentation grid.

    Piecewise constant colours: a round blob of tissue on glass, yellow elliptical fibers on a jittered grid with red
    collagen between them ("high" fibrosis: smaller fibers further apart), freezing holes (small glass-coloured
    spots) in a fifth of the fibers, one tear (a large white gap), one blue fold, a satellite fragment with two fibers
    far from the main section, and 200 um of glass around everything so the default background margin finds glass.
    Returns frame (BGR), labels_ds (uint32), px_um and truth."""
    rng = np.random.default_rng(seed)
    px_um = 0.2167
    rgb = np.tile(GLASS, (size, size, 1)).astype(np.float32)
    yy, xx = np.mgrid[:size, :size]
    c0 = size / 2
    r_blob = 700.0
    ang = np.arctan2(yy - c0, xx - c0)
    rad = np.hypot(yy - c0, xx - c0)
    blob = rad <= r_blob * (1 + 0.06 * np.sin(4 * ang + rng.uniform(0, 6)))
    frag_c = (c0 - 0.32 * size, c0 + 0.25 * size)  # beyond what main_section bridges (size 3600: 1150 px up)
    frag = np.hypot(yy - frag_c[0], xx - frag_c[1]) <= 160
    for m in (blob, frag):  # collagen everywhere in the tissue, in patches of different depth
        patch = ((yy // 400) + (xx // 400)) % 3
        for k, dark in enumerate((1.0, 0.9, 0.8)):
            rgb[m & (patch == k)] = PSR_RED * dark
    pitch, r_lo, r_hi = (150, 45, 75) if fibrosis == "low" else (165, 38, 62)
    masks, rows = [], []
    for cy, cx in hex_grid(rng, size, pitch, 8):
        r = rng.uniform(r_lo, r_hi)
        yi, xi = int(np.clip(cy, 0, size - 1)), int(np.clip(cx, 0, size - 1))
        if rad[yi, xi] + r + 20 > r_blob * 0.94:
            continue
        masks.append(ellipse((size, size), cy, cx, r, r * rng.uniform(0.75, 1.0), rng.uniform(0, np.pi)))
        rows.append({"kind": "fiber", "r_px": round(r, 1)})
    for dy, dx in ((-55, -30), (60, 40)):
        masks.append(ellipse((size, size), frag_c[0] + dy, frag_c[1] + dx, 50, 42, 0.5))
        rows.append({"kind": "fragment", "r_px": 50})
    for lab, m in enumerate(masks, start=1):
        rgb[m] = PSR_YELLOW * rng.uniform(0.85, 1.0)
        if rng.random() < 0.2:  # freezing holes: small, glass-coloured, inside the fiber
            cy, cx = np.argwhere(m).mean(0)
            for _ in range(rng.integers(1, 4)):
                hole = ellipse((size, size), cy + rng.uniform(-15, 15), cx + rng.uniform(-15, 15),
                               rng.uniform(8, 14), rng.uniform(8, 14), 0.0)
                rgb[hole & m] = GLASS
                rows[lab - 1]["holes"] = rows[lab - 1].get("holes", 0) + 1
    tear = blob & (np.abs(yy - (c0 + 250)) < 30) & (xx > c0 - 300) & (xx < c0 + 100)  # a large white gap
    rgb[tear] = GLASS
    fold = blob & (np.abs((xx - c0) * 0.6 - (yy - c0) + 420) < 12)
    rgb[fold] = PSR_BLUE
    frame = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)[..., ::-1].copy()  # file order BGR
    truth = pd.DataFrame(rows)
    truth.insert(0, "label", np.arange(1, len(rows) + 1))
    return {"frame": frame, "labels_ds": to_labels(masks, f), "px_um": px_um, "truth": truth}


def synthetic_he_field(seed: int, size: int = 1200, grid: int = 2) -> dict:
    """A seeded H&E field (RGB uint8) with fiber labels on the 2x grid, for the H&E step with saved outlines.

    A round piece of tissue (pale gaps) on glass holds eosin-pink elliptical fibers; purple nuclei of 2 to 4 um sit
    under the membrane of most fibers, at the centre of a third of them, half way in for a few, between the fibers,
    and in one dense cluster. One fiber crosses the tissue edge. Returns rgb, fiber_labels (on the grid), px_um and
    truth (per fiber: central nucleus or not)."""
    rng = np.random.default_rng(seed)
    px_um = 0.23
    rgb = np.tile(HE_GLASS, (size, size, 1)).astype(np.float32)
    yy, xx = np.mgrid[:size, :size]
    c0 = size / 2
    rad = np.hypot(yy - c0, xx - c0)
    tissue = rad <= 0.45 * size
    rgb[tissue] = HE_GAP
    masks, rows, nuclei = [], [], []
    for cy, cx in hex_grid(rng, size, 150, 8):
        r = rng.uniform(45, 70)
        yi, xi = int(np.clip(cy, 0, size - 1)), int(np.clip(cx, 0, size - 1))
        if rad[yi, xi] + r + 12 > 0.45 * size:
            continue
        angle = rng.uniform(0, np.pi)
        ry, rx = r, r * rng.uniform(0.8, 1.0)
        masks.append(ellipse((size, size), cy, cx, ry, rx, angle))
        central = rng.random() < 0.33
        rows.append({"kind": "fiber", "central": central, "r_px": round(r, 1)})
        for _ in range(rng.integers(1, 3)):  # peripheral nuclei just inside the outline
            t = rng.uniform(0, 2 * np.pi)
            nuclei.append((cy + 0.8 * ry * np.sin(t), cx + 0.8 * rx * np.cos(t), rng.uniform(7, 9)))
        if central:
            nuclei.append((cy + rng.uniform(-4, 4), cx + rng.uniform(-4, 4), rng.uniform(7, 9)))
        elif rng.random() < 0.15:  # half way in: splits the depth sweep
            t = rng.uniform(0, 2 * np.pi)
            nuclei.append((cy + 0.45 * ry * np.sin(t), cx + 0.45 * rx * np.cos(t), rng.uniform(7, 9)))
    masks.append(ellipse((size, size), c0 + 0.45 * size - 10, c0, 55, 50, 0.2))  # crosses the tissue edge
    rows.append({"kind": "edge", "central": False, "r_px": 55})
    for lab, m in enumerate(masks, start=1):
        rgb[m] = HE_FIBER * rng.uniform(0.9, 1.0)
    in_fiber = np.zeros((size, size), bool)
    for m in masks:
        in_fiber |= m
    gap = tissue & ~in_fiber
    ys, xs = np.nonzero(gap)
    for i in rng.choice(len(ys), 40, replace=False):  # interstitial nuclei
        nuclei.append((ys[i], xs[i], rng.uniform(7, 9)))
    cy, cx = c0 - 0.3 * size, c0  # a dense cluster in one 100 um tile
    for _ in range(25):
        y, x = cy + rng.uniform(-45, 45), cx + rng.uniform(-45, 45)
        if gap[int(y), int(x)]:
            nuclei.append((y, x, rng.uniform(7, 9)))
    for y, x, r in nuclei:
        rgb[ellipse((size, size), y, x, r, r * rng.uniform(0.85, 1.0), rng.uniform(0, np.pi))] = HE_NUCLEUS
    truth = pd.DataFrame(rows)
    truth.insert(0, "label", np.arange(1, len(rows) + 1))
    return {"rgb": np.clip(np.rint(rgb), 0, 255).astype(np.uint8), "fiber_labels": to_labels(masks, grid),
            "px_um": px_um, "truth": truth, "n_nuclei": len(nuclei)}
