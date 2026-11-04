"""Synthetic laminin + necrosis sections, and the fake ND2 headers and description the necrosis tests use."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from msq.core.objects import upsample_labels
from msq.core.synthetic import ellipse, hex_grid, to_labels

# The 'Plane #n:' blocks as NIS Elements writes them (Nikon DS-Qi2).
DESCRIPTION = """Camera Name: Camera Nikon DS-Qi2
Plane #1:
 Name: DAPI
 Camera Settings:
  Camera Type: Nikon DS-Qi2
  Binning: 1.0x1.0
  Exposure: 300 ms
  Gain: 1.0x
 Microscope Settings:
  Nikon Ti2, FilterChanger(Turret-Lo): 2 (DAPI-5060C (DAPI))
  Nikon Ti2, Illuminator(DIA) Iris intensity: 23.8
  Sola II, Illuminator(Sola) Iris intensity: 10.0
Plane #2:
 Name: GFP
 Camera Settings:
  Camera Type: Nikon DS-Qi2
  Binning: 1.0x1.0
  Exposure: 500 ms
  Gain: 1.0x
 Microscope Settings:
  Nikon Ti2, FilterChanger(Turret-Lo): 3 (96372 - GFP-4050B (GFP))
  Nikon Ti2, Illuminator(DIA) Iris intensity: 23.8
  Sola II, Illuminator(Sola) Iris intensity: 10.0
Plane #3:
 Name: Cy5
 Camera Settings:
  Camera Type: Nikon DS-Qi2
  Binning: 1.0x1.0
  Exposure: 2 s
  Gain: 1.0x
 Microscope Settings:
  Nikon Ti2, FilterChanger(Turret-Lo): 5 (Cy5-4040C (Cy5))
  Nikon Ti2, Illuminator(DIA) Iris intensity: 23.8
  Sola II, Illuminator(Sola) Iris intensity: 10.0
"""

IMAGING = {"lam": {"exposure_ms": 500.0, "gain": 1.0, "light": 10.0},
           "nec": {"exposure_ms": 2000.0, "gain": 1.0, "light": 10.0}}

# A 20x whole section (whole sections are 11,000 to 14,000 px across).
HEADER = {"sizes": {"C": 3, "Y": 12000, "X": 12500}, "channels": ["DAPI", "GFP", "Cy5"], "px_um": 0.3606,
          "bits": 14, "imaging": IMAGING, "acquired": "2026-07-21 00:15:06"}
OVERVIEW = {**HEADER, "sizes": {"C": 3, "Y": 2424, "X": 2424}, "px_um": 1.8337}
FIELD = {**HEADER, "sizes": {"C": 3, "Y": 1024, "X": 1440}}


def header(acquired: str = "2026-07-21 00:15:06", **nec) -> dict:
    """HEADER with another acquisition time and some necrosis-channel settings changed."""
    return {**HEADER, "acquired": acquired, "imaging": {**IMAGING, "nec": {**IMAGING["nec"], **nec}}}


GLASS_LAM, INTERSTITIUM_LAM, INTERIOR_LAM, MEMBRANE_LAM = 300.0, 1200.0, 600.0, 3000.0
GLASS_NEC, TISSUE_NEC = 400.0, 600.0
CHANNELS = ["DAPI", "GFP", "Cy5"]  # laminin on GFP, necrosis on Cy5, as in the 20x scans


def synthetic_necrosis_section(seed: int, group: str = "group_a", size_ds: int = 320, f: int = 4,
                               px_um: float = 0.3606) -> dict:
    """A seeded laminin + necrosis scan (C, Y, X uint16, full resolution) with the fibers' labels on the 4x grid.

    Designed on the 4x grid and repeated f times, so the block means equal the design exactly. A round blob of
    tissue on glass: elliptical fiber interiors (dark on laminin) behind 2 px membranes (bright), interstitium
    between them; necrosis signal low in healthy cores with a per-fiber level, high in necrotic ones (1% of a
    reference section's fibers, 10% of a group_a section's), a few intermediate cores, one bright patch of
    interstitial debris. Artifacts: a fiber cut by the section edge, two specks of debris, a torn-off fragment with
    two fibers beyond what main_section bridges. Returns frame, labels_ds (uint32), px_um, channels and truth."""
    rng = np.random.default_rng(seed)
    lam = np.full((size_ds, size_ds), GLASS_LAM, np.float32)
    nec = np.full((size_ds, size_ds), GLASS_NEC, np.float32)
    yy, xx = np.mgrid[:size_ds, :size_ds]
    c0 = size_ds / 2
    r_blob = 0.35 * size_ds
    ang = np.arctan2(yy - c0, xx - c0)
    rad = np.hypot(yy - c0, xx - c0)
    phase = rng.uniform(0, 6)
    blob = rad <= r_blob * (1 + 0.07 * np.sin(3 * ang + phase))
    frag_c = (c0 - 0.42 * size_ds, c0 + 0.40 * size_ds)
    frag = np.hypot(yy - frag_c[0], xx - frag_c[1]) <= 0.07 * size_ds
    for m in (blob, frag):
        lam[m] = INTERSTITIUM_LAM
        nec[m] = TISSUE_NEC
    masks, rows = [], []
    for cy, cx in hex_grid(rng, size_ds, 26, 2):
        r = rng.uniform(7, 13)
        yi, xi = int(np.clip(cy, 0, size_ds - 1)), int(np.clip(cx, 0, size_ds - 1))
        if rad[yi, xi] + r + 3 > r_blob * 0.93:
            continue
        masks.append(ellipse((size_ds, size_ds), cy, cx, r, r * rng.uniform(0.75, 1.0), rng.uniform(0, np.pi)))
        rows.append({"kind": "fiber"})
    # a fiber cut by the section's edge, two specks of debris, two fibers on the fragment
    t = rng.uniform(0, 2 * np.pi)
    r_edge = r_blob * (1 + 0.07 * np.sin(3 * t + phase)) - 4  # the interior sticks 5 px out of the tissue
    masks.append(ellipse((size_ds, size_ds), c0 + r_edge * np.sin(t), c0 + r_edge * np.cos(t), 9, 9, 0))
    rows.append({"kind": "section_edge"})
    for _ in range(2):
        while True:
            cy, cx = rng.uniform(c0 - 0.6 * r_blob, c0 + 0.6 * r_blob, 2)
            if not any(m[int(cy), int(cx)] for m in masks):
                break
        masks.append(ellipse((size_ds, size_ds), cy, cx, 1.6, 1.6, 0))
        rows.append({"kind": "debris"})
    for dy, dx in ((-6, -4), (7, 5)):
        masks.append(ellipse((size_ds, size_ds), frag_c[0] + dy, frag_c[1] + dx, 6, 5, 0.4))
        rows.append({"kind": "fragment"})
    necrotic_share = 0.01 if group == "reference" else 0.10
    for lab, m in enumerate(masks, start=1):
        membrane = ndi.binary_dilation(m, iterations=2) & ~m & (blob | frag)  # no membrane on the glass
        lam[membrane] = MEMBRANE_LAM * rng.uniform(0.8, 1.0)
        lam[m] = INTERIOR_LAM * rng.uniform(0.8, 1.2)
        u = rng.random()
        if u < necrotic_share:
            level, kind = rng.uniform(2000, 4500), "necrotic"
        elif u < necrotic_share + 0.04:
            level, kind = rng.uniform(800, 1500), "intermediate"
        else:
            level, kind = rng.uniform(450, 700), "healthy"
        nec[m] = level
        rows[lab - 1].update(necrosis=kind, core_level=round(float(level), 1))
    debris = blob & (np.hypot(yy - (c0 + 0.5 * r_blob), xx - (c0 - 0.3 * r_blob)) <= 4)
    debris &= ~np.any(masks, axis=0)
    nec[debris] = 3000.0  # necrotic debris between fibers
    dapi = np.full_like(lam, 1000.0)
    frame = np.stack([upsample_labels(a, f) for a in (dapi, lam, nec)]).astype(np.uint16)
    truth = pd.DataFrame(rows)
    truth.insert(0, "label", np.arange(1, len(rows) + 1))
    return {"frame": frame, "labels_ds": to_labels(masks, 1), "px_um": px_um, "channels": list(CHANNELS),
            "truth": truth}
