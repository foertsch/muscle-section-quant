"""Synthetic dystrophin sections and the fake section results the dystrophin tests use."""

from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from msq.core.synthetic import ellipse_box

HEADER = {"sizes": {"C": 3, "Y": 100, "X": 100}, "channels": ["DAPI", "GFP", "Cy5"], "px_um": 0.3606,
          "gfp": {"exposure": "2 s", "gain": "1.0x", "light": "10.0"}}


def write_section(out_root: Path, name: str, n_fibers: int = 300, positive_frac: float | None = None,
                  source: Path | None = None, few_sectors: int = 0) -> Path:
    """fibers.csv, sectors.csv.gz and run_info.json for one section, in the format the measurement writes.
    Negatives' sectors ~N(100, 120), positives' ~N(900, 150); controls (name ends in _ctrl) get 1% positives,
    treated 40%, unless positive_frac is given. source: the scan it was "measured" from. few_sectors: that many
    fibers at the end have only 20 of 36 sectors measured (too few to be called), all very bright (5000)."""
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    frac = positive_frac if positive_frac is not None else (0.01 if name.endswith("_ctrl") else 0.4)
    labels = np.arange(1, n_fibers + 1)
    pos = rng.random(n_fibers) < frac
    short = labels > n_fibers - few_sectors
    rows = []
    for lab, p, few in zip(labels, pos, short):
        v = rng.normal(900, 150, 36) if p else rng.normal(100, 120, 36)
        if few:
            v = np.where(np.arange(36) < 20, 5000.0, np.nan)
        rows += [(lab, i, round(float(x), 1)) for i, x in enumerate(v)]
    d = out_root / name
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["label", "sector", "ridge_minus_cyto"]).to_csv(d / "sectors.csv.gz", index=False)
    pd.DataFrame({"sample": name, "label": labels, "include": True, "n_sectors_measured": np.where(short, 20, 36),
                  "dys_membrane_specific": np.where(pos, 900.0, 100.0), "area_um2": 1500.0,
                  "min_feret_um": rng.normal(35, 8, n_fibers).round(1), "flag_bright_cytoplasm": False}
                 ).to_csv(d / "fibers.csv", index=False)
    info = {"params": {"n_sectors": 36, "downsample": 4}, "pixel_size_um": 0.3606}
    if source is not None:
        info.update(source=str(source), source_size=source.stat().st_size)
    (d / "run_info.json").write_text(json.dumps(info))
    return d


def synthetic_section(seed: int, group: str = "treated", size: int = 768, f: int = 4, px_um: float = 0.3606,
                      pitch: int = 54, tissue_radius: float = 0.30, interior_artifacts: bool = True,
                      fiber_radius: tuple[float, float] = (16, 24)) -> dict:
    """A seeded whole-section scan as the dystrophin reader returns it, with the fibers' own labels on the
    segmentation grid.

    Piecewise constant images (no pixel noise): a round tissue blob on dark glass holds elliptical fibers on a
    jittered grid, each with its own actin level, cytoplasmic GFP and a GFP ring 3 px wide inside its edge. Treated
    fibers carry a bright ring over an angular fraction of the edge drawn from U(0, 1), so the 50% call splits them;
    control fibers carry a dim ring over such a fraction, and 2% of them a full bright one (revertant-like).
    Artifacts: one hollow (vessel-like) fiber, one with bright cytoplasm, two specks of debris, a fiber on the image
    border, a torn-off fragment with two fibers far from the main blob, and zero padding along the top as in edited
    scans.

    tissue_radius: the tissue disc's radius as a share of size (the demo uses a larger one, so the reports' 1000 px
    windows fit inside the tissue). interior_artifacts: False leaves out the hollow object and the debris inside the
    tissue (the reports pick windows free of excluded objects; the demo needs some). Each fiber is kept and drawn as
    its own bounding box, so large sections stay fast. fiber_radius: the range of fiber radii in pixels (the default
    gives fibers of about 150 um2, small for muscle but quick; the demo uses real sizes, about 700 to 1,200 um2).

    Returns gfp and act (uint16, size x size), labels_ds (uint32, size/f square), px_um, sat, channels, and truth
    (one row per object)."""
    rng = np.random.default_rng(seed)
    gfp = np.full((size, size), 500.0, np.float32)
    act = np.full((size, size), 500.0, np.float32)
    yy, xx = np.mgrid[:size, :size]
    cy0 = cx0 = size / 2
    r_blob = tissue_radius * size
    ang = np.arctan2(yy - cy0, xx - cx0)
    rad = np.hypot(yy - cy0, xx - cx0)
    phase = rng.uniform(0, 2 * np.pi)
    blob = rad <= r_blob * (1 + 0.08 * np.sin(3 * ang + phase))
    frag_c = (int(0.14 * size), int(0.84 * size))  # a torn-off fragment, further than main_section bridges
    frag = np.hypot(yy - frag_c[0], xx - frag_c[1]) <= 0.055 * size
    for m in (blob, frag):
        act[m] = 1500.0
        gfp[m] = 900.0

    objects = []  # [(box, mask inside the box), kind, coverage]

    def add_fiber(cy, cx, ry, rx, angle, kind="fiber"):
        found = ellipse_box((size, size), cy, cx, ry, rx, angle)
        if found is None:
            return
        objects.append([found, kind, np.nan])

    def covers(obj, y: int, x: int) -> bool:
        (by, bx), m = obj[0]
        return by.start <= y < by.stop and bx.start <= x < bx.stop and bool(m[y - by.start, x - bx.start])

    # fibers on a jittered hexagonal grid inside the blob
    rows = np.arange(-size // 2, size // 2, int(pitch * 0.866))
    for i, dy in enumerate(rows):
        for dx in np.arange(-size // 2, size // 2, pitch) + (pitch / 2 if i % 2 else 0):
            cy, cx = cy0 + dy + rng.uniform(-6, 6), cx0 + dx + rng.uniform(-6, 6)
            r = rng.uniform(*fiber_radius)
            yi, xi = int(np.clip(cy, 0, size - 1)), int(np.clip(cx, 0, size - 1))
            if rad[yi, xi] + r + 6 > r_blob * (1 + 0.08 * np.sin(3 * ang[yi, xi] + phase)):
                continue
            add_fiber(cy, cx, r, r * rng.uniform(0.75, 1.0), rng.uniform(0, np.pi))
    n_grid = len(objects)
    # artifacts
    if interior_artifacts:
        objects[rng.integers(n_grid)][1] = "hollow"
    objects[rng.integers(n_grid)][1] = "bright_cytoplasm"
    for _ in range(2 if interior_artifacts else 0):  # debris specks on the interstitium
        while True:
            cy, cx = rng.uniform(cy0 - r_blob * 0.7, cy0 + r_blob * 0.7, 2)
            if blob[int(cy), int(cx)] and not any(covers(o, int(cy), int(cx)) for o in objects):
                break
        add_fiber(cy, cx, 4, 4, 0.0, "debris")
    add_fiber(size - 8, cx0, 20, 20, 0.0, "border")  # cut by the image edge, off the tissue
    add_fiber(frag_c[0] - 14, frag_c[1], 13, 11, 0.3, "fragment")
    add_fiber(frag_c[0] + 14, frag_c[1] + 4, 12, 12, 1.1, "fragment")

    rows_out = []
    for lab, ((box, mask), kind, _) in enumerate(objects, start=1):
        # work inside the fiber's bounding box: the mask is empty outside it, and erosion treats the box's edge
        # like the image's (outside counts as empty), so every value equals the whole-canvas computation
        ys, xs = np.nonzero(mask)
        ys, xs = ys + box[0].start, xs + box[1].start  # canvas coordinates, so the centre is computed as before
        g, a = gfp[box], act[box]
        a_in = rng.uniform(4000, 7000)
        cyto = rng.uniform(1100, 1700)
        edge = mask & ~ndi.binary_erosion(mask, iterations=3)
        a[mask] = a_in
        g[mask] = cyto
        if kind == "hollow":
            a[ndi.binary_erosion(mask, iterations=8)] = 500.0
        if kind == "bright_cytoplasm":
            g[mask] = cyto = 6000.0
        coverage = float(rng.uniform(0, 1))
        start = rng.uniform(0, 2 * np.pi)
        mcy, mcx = ys.mean(), xs.mean()
        theta = (np.arctan2(yy[box] - mcy, xx[box] - mcx) - start) % (2 * np.pi)
        covered = edge & (theta <= coverage * 2 * np.pi)
        if group == "ctrl":
            bright = rng.random() < 0.02
            ring = rng.uniform(2500, 4000) if bright else cyto + rng.uniform(100, 400)
            if bright:
                coverage, covered = 1.0, edge
        else:
            ring = rng.uniform(2500, 4000)
        g[edge] = cyto + rng.uniform(0, 150)
        g[covered] = ring
        rows_out.append({"label": lab, "kind": kind, "coverage": round(coverage, 3), "ring": round(float(ring), 1),
                         "cytoplasm": round(float(cyto), 1)})
    gfp[:32] = 0  # zero padding along the top, as in edited scans
    act[:32] = 0

    labels = np.zeros((size // f, size // f), np.uint32)
    for lab, ((box, mask), _, _) in enumerate(objects, start=1):
        y0, x0 = box[0].start // f * f, box[1].start // f * f  # the box grown to whole f x f blocks
        y1, x1 = -(-box[0].stop // f) * f, -(-box[1].stop // f) * f
        block = np.zeros((y1 - y0, x1 - x0), bool)
        block[box[0].start - y0:box[0].stop - y0, box[1].start - x0:box[1].stop - x0] = mask
        ds = block.reshape((y1 - y0) // f, f, (x1 - x0) // f, f).mean((1, 3)) > 0.5
        sub = labels[y0 // f:y1 // f, x0 // f:x1 // f]
        sub[ds & (sub == 0)] = lab
    return {"gfp": gfp.astype(np.uint16), "act": act.astype(np.uint16), "labels_ds": labels, "px_um": px_um,
            "sat": 16383, "channels": ["DAPI", "GFP", "Cy5"], "truth": pd.DataFrame(rows_out)}
