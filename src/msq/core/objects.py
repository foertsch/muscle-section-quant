"""Per-object bookkeeping on the segmentation grid, shared by the assays that count pixels in zones around each
fiber (psr, necrosis): the shape and position of every object, pixel counts and means per zone, and labels repeated
onto a finer grid."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
from skimage.measure import regionprops

from msq.core.geometry import feret_px


def region_rows(labels: np.ndarray, px_um: float, f: int,
                extra: Callable[[Any], dict] | None = None) -> pd.DataFrame:
    """One row per object, in label order: size and shape on the grid of pixel size px_um, centroid and bounding box
    in full-resolution pixels (the grid is f times coarser), and whether it touches the image border. extra(region)
    gives more columns, added after those."""
    h, w = labels.shape
    rows = []
    for rp in regionprops(labels):
        y0, x0, y1, x1 = rp.bbox
        cy, cx = rp.centroid
        fmin, fmax = feret_px(rp.image)
        rows.append({
            "label": rp.label, "area_um2": rp.area * px_um ** 2, "min_feret_um": fmin * px_um,
            "max_feret_um": fmax * px_um, "solidity": rp.solidity, "eccentricity": rp.eccentricity,
            "centroid_y": (cy + 0.5) * f, "centroid_x": (cx + 0.5) * f,  # full-resolution pixels
            "bbox_y0": y0 * f, "bbox_x0": x0 * f, "bbox_y1": y1 * f, "bbox_x1": x1 * f,
            "touches_border": y0 == 0 or x0 == 0 or y1 == h or x1 == w, **(extra(rp) if extra else {})})
    return pd.DataFrame(rows)


def joint_counts(zone: np.ndarray, values: np.ndarray, n_rows: int, n_values: int) -> np.ndarray:
    """counts[label, value] (int64, n_rows x n_values): the pixels of each label of a zone image that take each integer
    value 0 .. n_values - 1 of values (a class, a histogram bin)."""
    idx = zone.ravel().astype(np.int64) * n_values + values.ravel()
    return np.bincount(idx, minlength=n_rows * n_values)[:n_rows * n_values].reshape(n_rows, n_values)


def zone_mean_sd(zone: np.ndarray, img: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pixel count, mean and SD of img per label of a zone image (arrays of length n + 1; NaN where empty)."""
    idx = zone.ravel().astype(np.int64)
    v = img.ravel().astype(np.float64)
    cnt = np.bincount(idx, minlength=n + 1)[:n + 1]
    s = np.bincount(idx, weights=v, minlength=n + 1)[:n + 1]
    s2 = np.bincount(idx, weights=v * v, minlength=n + 1)[:n + 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(cnt > 0, s / cnt, np.nan)
        sd = np.sqrt(np.maximum(np.where(cnt > 0, s2 / cnt, np.nan) - mean ** 2, 0))
    return cnt, mean, sd


def upsample_labels(labels: np.ndarray, f: int, shape: tuple[int, int] | None = None, origin: tuple[int, int] = (0, 0),
                    clip: bool = False) -> np.ndarray:
    """Labels (or any grid of values) repeated f times along both axes: the window of this shape (default: all of it)
    whose top-left pixel is at origin on the finer grid. Where the window runs past the grid it is 0, or with clip the
    edge value."""
    (oy, ox), (h, w) = origin, shape or (labels.shape[0] * f, labels.shape[1] * f)
    if clip:
        ys, xs = np.arange(oy, oy + h) // f, np.arange(ox, ox + w) // f
        return labels[np.ix_(ys.clip(0, labels.shape[0] - 1), xs.clip(0, labels.shape[1] - 1))]
    y0, x0 = oy // f, ox // f
    up = np.repeat(np.repeat(labels[y0:(oy + h - 1) // f + 1, x0:(ox + w - 1) // f + 1], f, 0), f, 1)
    up = up[oy - y0 * f:oy - y0 * f + h, ox - x0 * f:ox - x0 * f + w]
    if up.shape != (h, w):
        up = np.pad(up, ((0, h - up.shape[0]), (0, w - up.shape[1])))
    return up
