"""Three zones per segmented fiber on the segmentation grid: the core (the mask eroded by k pixels), the edge band
(mask minus core) and the ring just outside (k pixels wide, each ring pixel owned by the nearest fiber)."""

from __future__ import annotations

import numpy as np
from skimage.segmentation import expand_labels


def zones(labels: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Core (each mask eroded by k pixels, touching masks kept apart), edge band, and the ring k pixels outside."""
    core = labels.copy()
    for _ in range(k):
        keep = core > 0
        padded = np.pad(core, 1)
        for nb in (padded[:-2, 1:-1], padded[2:, 1:-1], padded[1:-1, :-2], padded[1:-1, 2:]):
            keep &= nb == core
        core = np.where(keep, core, 0)
    band = np.where((labels > 0) & (core == 0), labels, 0).astype(labels.dtype)
    ring = expand_labels(labels, k)
    ring[labels > 0] = 0
    return core, band, ring.astype(labels.dtype)
