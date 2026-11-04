"""Histogram bins shared by the pages' rows of histograms."""

from __future__ import annotations

import numpy as np


def shared_edges(arrays, width: float = 25.0, lo_pct: float = 0.5, hi_pct: float = 99.5) -> np.ndarray:
    """Bin edges on multiples of width from the lowest lo_pct to the highest hi_pct percentile of any
    array, so no sample has more than (100 - hi_pct)% of its values in the open-ended last bin."""
    finite = [v[np.isfinite(v)] for v in (np.asarray(a, float) for a in arrays)]
    finite = [v for v in finite if v.size]
    lo = np.floor(min(np.percentile(v, lo_pct) for v in finite) / width) * width
    hi = np.ceil(max(np.percentile(v, hi_pct) for v in finite) / width) * width
    return np.arange(lo, hi + width / 2, width)


def hist_pct(values, edges: np.ndarray) -> np.ndarray:
    """Percent of finite values per bin; values beyond the edges count in the first or last bin."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.zeros(len(edges) - 1)
    counts, _ = np.histogram(np.clip(v, edges[0], edges[-1]), edges)
    return 100 * counts / v.size
