"""Shape measures of a fiber mask."""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull


def feret_px(mask: np.ndarray) -> tuple[float, float]:
    """Minimum and maximum Feret (caliper) diameter in pixels, using pixel corners of the mask. The minimum is the
    narrowest caliper over the hull's edge directions, the maximum the longest distance between two hull corners."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return 0.0, 0.0
    pts = np.concatenate([np.stack([ys + dy, xs + dx], 1) for dy in (0, 1) for dx in (0, 1)]).astype(float)
    try:
        hull = pts[ConvexHull(pts).vertices]
    except Exception:  # degenerate (collinear) masks
        span = np.ptp(pts, axis=0)
        return float(span.min()), float(np.hypot(*span))
    edges = np.roll(hull, -1, 0) - hull
    normals = np.stack([-edges[:, 1], edges[:, 0]], 1)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    smallest = min(np.ptp(hull @ nrm) for nrm in normals)
    largest = np.sqrt(((hull[:, None, :] - hull[None, :, :]) ** 2).sum(-1).max())
    return float(smallest), float(largest)


def min_feret_px(mask: np.ndarray) -> float:
    """Minimum Feret (caliper) diameter in pixels."""
    return feret_px(mask)[0]
