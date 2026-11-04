"""Fiber outlines as SVG paths: the longest 0.5-level contour of a mask, simplified."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import approximate_polygon, find_contours


def outline_path(mask: np.ndarray, oy: float, ox: float, scale: float = 1.0, tol: float = 1.2,
                 centre: bool = False) -> str:
    """SVG path of the longest 0.5-level contour of a boolean mask, offset by (oy, ox), in units of `scale` pixels
    (the mask's pixels are scale apart). centre puts the path on pixel centres, not corners; tol is the polygon
    simplification in pixels."""
    cs = find_contours(np.pad(mask, 1).astype(np.float32), 0.5)
    if not cs:
        return ""
    c = approximate_polygon(max(cs, key=len), tol) - 1
    if centre:
        c = c + 0.5
    c = c * scale + np.array([oy, ox])
    return "M" + "L".join(f"{x:.1f},{y:.1f}" for y, x in c) + "Z"


def label_paths(labels: np.ndarray, wanted: np.ndarray, oy: float = 0.0, ox: float = 0.0, scale: float = 1.0,
                tol: float = 1.2, centre: bool = False) -> dict[int, str]:
    """Outline path per wanted label of a label image (coordinates: label-image pixels x scale, offset)."""
    out = {}
    wanted_set = set(int(w) for w in wanted)
    for lab, sl in enumerate(ndi.find_objects(labels), start=1):
        if sl is None or lab not in wanted_set:
            continue
        p = outline_path(labels[sl] == lab, (sl[0].start * scale) + oy, (sl[1].start * scale) + ox, scale, tol, centre)
        if p:
            out[lab] = p
    return out
