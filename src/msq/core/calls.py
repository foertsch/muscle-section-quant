"""Thresholds and calls the assays share: a threshold set on the control fibers, the share of values that reach it,
and the per-animal and per-group tables. What is one assay's own stays there: dystrophin's ring score, levels,
held-out and borrowed thresholds; necrosis's correction and percentile grid and its pixel shares.

Each function is the expression the assays used before, so moving onto it changes no number."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd


def threshold_from_controls(values, pct: float) -> float:
    """The pct-th percentile of the control values (numpy's default, linear), NaN left out; a ValueError without a
    value. Dystrophin keeps its own np.nanpercentile, which gives NaN then."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        raise ValueError("no control values to set the threshold on")
    return float(np.percentile(v, pct))


def rate_at_or_above(values, threshold: float) -> float:
    """The share of values at or above threshold; NaN counts as below. NaN when there are no values."""
    s = values if isinstance(values, pd.Series) else pd.Series(values, dtype=float)
    return float((s >= threshold).mean())


def animal_means(sections: pd.DataFrame, keys: Sequence[str], means: Sequence[str], sums: Sequence[str] = (), *,
                 order: Callable | None = None) -> pd.DataFrame:
    """One row per animal (keys end with it): n_sections, the sums and the means of its sections, in that column
    order. Rows in sorted key order; with order (a sort key such as Groups.sort_key()) in that order instead, stable,
    ties in order of first appearance."""
    t = (sections.groupby(list(keys), sort=order is None)
         .agg(n_sections=("sample", "size"), **{c: (c, "sum") for c in sums}, **{c: (c, "mean") for c in means})
         .reset_index())
    if order is not None:
        t = t.sort_values(list(keys), key=order, kind="stable").reset_index(drop=True)
    return t


def group_stats(animals: pd.DataFrame, keys: Sequence[str], cols: Sequence[str], sums: Sequence[str] = (), *,
                sd: str = "sd", sort: bool = False) -> pd.DataFrame:
    """Per group (keys, in order of first appearance, or sorted): n_animals, the sums, then per column its mean and
    sample SD (ddof 1) over the animals, named <col>_mean and <col>_<sd>."""
    g = animals.groupby(list(keys), sort=sort)
    t = g.agg(n_animals=("animal", "size"), **{c: (c, "sum") for c in sums})
    for c in cols:
        t[f"{c}_mean"] = g[c].mean()
        t[f"{c}_{sd}"] = g[c].std()
    return t.reset_index()
