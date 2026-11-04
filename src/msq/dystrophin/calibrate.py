"""Building blocks for positive / negative fiber calls from the 36 ring sectors (used by full_summary.py).

Each fiber's ring is measured in 36 angular sectors (sectors.csv.gz; sector value = 90th percentile
of the +/-1.5 um band minus the fiber's own cytoplasm). These are the same values whose median is
dys_membrane_specific. A fiber's coverage is the fraction of its measured sectors at or above a sector
threshold; fibers with fewer than 24 of 36 sectors measured are not called.

The calibrated rule (used by full_summary.py, see GLOSSARY.md): one threshold per muscle = the 98th percentile of
pooled negative-control fibers' score at 50% coverage (coverage_score, threshold_from_controls). A fiber is
positive at stringency 50 / 60 / 70% when its score at that level reaches the same threshold, i.e. when at least
18 / 22 / 26 of its 36 sectors do.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_results(results: Path) -> dict[str, dict]:
    """Included fibers and their sector values, per sample."""
    out = {}
    for d in sorted(p.parent for p in results.glob("*/sectors.csv.gz")):
        n_sectors = json.loads((d / "run_info.json").read_text(encoding="utf-8"))["params"]["n_sectors"]
        fib = pd.read_csv(d / "fibers.csv")
        fib = fib[fib["include"]].set_index("label")
        fib["sector_measured_frac"] = fib["n_sectors_measured"] / n_sectors
        sec = pd.read_csv(d / "sectors.csv.gz")
        out[d.name] = {"fibers": fib, "sectors": sec[sec["label"].isin(fib.index)]}
    return out


def coverage_score(sectors: pd.DataFrame, coverage: float, min_sectors: int = 24) -> pd.Series:
    """Per fiber: the highest sector threshold it still passes at this coverage.

    That is the k-th brightest of its measured sector values, k = ceil(coverage x n_measured), so
    `score >= threshold` is exactly "the share of the fiber's measured sectors at or above threshold is >= coverage".
    At coverage 0.5 with 36 sectors it is the 18th brightest sector, about the fiber's median. Fibers with fewer
    than min_sectors measured sectors get no score."""
    wide = sectors.pivot(index="label", columns="sector", values="ridge_minus_cyto")
    v = -np.sort(-wide.to_numpy(dtype=float), axis=1)  # descending; NaN sorts last
    n = np.isfinite(v).sum(axis=1)
    ok = n >= min_sectors
    k = np.ceil(coverage * n).astype(int)
    score = np.full(len(v), np.nan)
    score[ok] = v[ok, k[ok] - 1]
    return pd.Series(score, index=wide.index, name=f"score_{coverage:g}").dropna()


def threshold_from_controls(control_scores, fpr: float = 0.02) -> float:
    """Sector threshold that only `fpr` of the pooled control fibers reach: the (1 - fpr) percentile of
    their scores at the chosen coverage."""
    return float(np.nanpercentile(np.asarray(control_scores, dtype=float), 100 * (1 - fpr)))
