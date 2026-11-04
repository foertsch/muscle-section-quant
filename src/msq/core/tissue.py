"""Tissue on glass: the pieces worth keeping, the main section, and the off-tissue background level.

Each assay finds its tissue mask its own way (actin, saturation, laminin); what follows is shared: small pieces are
dropped, the largest piece after bridging small gaps is the main section, and the background is the median of the
pixels well away from the tissue.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu


def keep_large_pieces(mask: np.ndarray, min_share: float = 0.01) -> np.ndarray:
    """The connected pieces of a mask that reach min_share of the largest piece's area (one piece: unchanged)."""
    lab, n = ndi.label(mask)
    if n > 1:
        sizes = ndi.sum(mask, lab, range(1, n + 1))
        mask = np.isin(lab, 1 + np.flatnonzero(sizes >= min_share * sizes.max()))
    return mask


def otsu_tissue(sm: np.ndarray, valid: np.ndarray, f: Callable[[np.ndarray], np.ndarray] = lambda a: a) -> np.ndarray:
    """Tissue from a smoothed image: valid pixels above the Otsu threshold of the valid ones, holes filled, small pieces
    dropped (keep_large_pieces). f changes the values first (np.log1p for intensities, so dim tissue is told from glass
    on a log scale)."""
    thr = threshold_otsu(f(sm[valid]))
    return keep_large_pieces(ndi.binary_fill_holes((f(sm) > thr) & valid))


def off_tissue(tissue: np.ndarray, valid: np.ndarray, margin_um: float, px_um: float,
               euclidean: bool = False) -> np.ndarray:
    """Valid pixels at least margin_um from the tissue. The default grows the tissue by int(round(margin_um / px_um))
    pixels, a diamond: margin_um along the axes, 0.7x diagonally; euclidean takes the true distance."""
    if euclidean:
        return valid & (ndi.distance_transform_edt(~tissue) > margin_um / px_um)
    r = max(1, int(round(margin_um / px_um)))
    return valid & ~ndi.binary_dilation(tissue, iterations=r)


def main_section(tissue: np.ndarray, px_um: float, close_um: float = 50.0) -> np.ndarray:
    """Largest connected tissue piece after bridging gaps up to close_um along the image axes (0.7x that
    diagonally: the closing grows a diamond), holes filled.

    Whole-slide fields also capture neighbouring sections on the slide; only the main section is analysed.
    Torn-off fragments further than close_um from it are dropped as well."""
    r = max(1, int(round(close_um / px_um / 2)))
    closed = ndi.binary_closing(np.pad(tissue, r), iterations=r)[r:-r, r:-r] | tissue
    lab, n = ndi.label(closed)
    if n == 0:
        return closed
    sizes = ndi.sum(closed, lab, range(1, n + 1))
    return ndi.binary_fill_holes(lab == 1 + int(np.argmax(sizes)))


def background_level(img: np.ndarray, off_tissue: np.ndarray) -> float:
    """Median of off-tissue, non-zero pixels: camera offset plus slide and mountant signal."""
    vals = img[off_tissue & (img > 0)]
    if vals.size < 100:
        raise ValueError("too few pixels off the tissue to estimate the background")
    return float(np.median(vals))
