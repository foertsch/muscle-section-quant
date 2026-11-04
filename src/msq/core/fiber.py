"""One fiber at full resolution: its mask (bilinear from the segmentation grid), the membrane band and cytoplasm core
around its boundary, and the angular sector of each pixel. The dystrophin measurement, its reports and the demo crops
all draw them from here, so what is shown is what is measured."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.transform import resize


def full_res_mask(m_ds: np.ndarray, f: int) -> np.ndarray:
    """A fiber's boolean mask on the f-times finer grid: linear interpolation of the mask, cut at 0.5."""
    return resize(m_ds.astype(np.float32), (m_ds.shape[0] * f, m_ds.shape[1] * f), order=1) > 0.5


def edge_zones(mask: np.ndarray, px_um: float, band_um: float, core_um: float,
               valid: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(band, edge_in, core) of a full-resolution fiber mask: the membrane band (within band_um of the boundary, both
    sides; only valid pixels when valid is given), its inner half, and the cytoplasm core (deeper than core_um, capped
    at half the fiber's greatest depth)."""
    din = ndi.distance_transform_edt(mask) * px_um
    dout = ndi.distance_transform_edt(~mask) * px_um
    edge_in = mask & (din <= band_um)
    band = edge_in | (~mask & (dout <= band_um))
    if valid is not None:
        band = band & valid
    return band, edge_in, mask & (din > min(core_um, 0.5 * din.max()))


def sector_index(yy: np.ndarray, xx: np.ndarray, cy: float, cx: float, n_sectors: int) -> np.ndarray:
    """The angular sector (0 .. n_sectors - 1) of pixels at (yy, xx) around the centre (cy, cx)."""
    ang = np.arctan2(yy - cy, xx - cx)
    return np.minimum(((ang + np.pi) / (2 * np.pi) * n_sectors).astype(int), n_sectors - 1)
