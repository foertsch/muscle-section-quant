
import numpy as np
import pandas as pd
import pytest

from msq.dystrophin import calibrate as cal  # noqa: E402


def sectors_table(per_fiber: dict[int, list[float]]) -> pd.DataFrame:
    rows = [(lab, i, v) for lab, vals in per_fiber.items() for i, v in enumerate(vals)]
    return pd.DataFrame(rows, columns=["label", "sector", "ridge_minus_cyto"])


def fiber_coverage(sectors: pd.DataFrame, measured_frac: pd.Series, sector_threshold: float,
                   min_measured: float = 2 / 3) -> pd.Series:
    """The rule coverage_score encodes, stated directly: the share of each fiber's measured sectors at or above
    sector_threshold; NaN when too few are measured."""
    s = sectors.dropna(subset=["ridge_minus_cyto"])
    cov = (s["ridge_minus_cyto"] >= sector_threshold).groupby(s["label"]).mean()
    cov = cov.reindex(measured_frac.index)
    cov[measured_frac < min_measured] = np.nan
    return cov


def test_coverage_score_matches_coverage_rule():
    rng = np.random.default_rng(3)
    per_fiber = {lab: list(rng.normal(300, 400, 36)) for lab in range(1, 41)}
    per_fiber[41] = list(np.arange(1, 37, dtype=float))  # sectors 1..36
    per_fiber[42] = [500.0] * 10 + [np.nan] * 26  # too few measured
    sec = sectors_table(per_fiber)
    measured = pd.Series({lab: np.isfinite(v).mean() for lab, v in per_fiber.items()})
    for c in (0.5, 0.6, 0.7):
        score = cal.coverage_score(sec, c)
        assert 42 not in score.index
        for cut in (0, 250, 600):
            via_cov = fiber_coverage(sec, measured, cut).dropna() >= c
            assert (via_cov.loc[score.index] == (score >= cut)).all()
    assert cal.coverage_score(sec, 0.5)[41] == 19  # 18th brightest of 1..36


def test_threshold_from_controls_lets_through_the_target_share():
    scores = np.arange(1000, dtype=float)
    cut = cal.threshold_from_controls(scores, fpr=0.02)
    assert cut == pytest.approx(979.02)
    assert (scores >= cut).mean() == pytest.approx(0.02)
