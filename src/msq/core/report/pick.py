"""Which section a page shows."""

from __future__ import annotations

import pandas as pd


def representative_section(share: pd.Series) -> str:
    """The section of one animal whose value is closest to the animal's mean (the first one on a tie).
    share: the animal's sections' values, indexed by section and in section order."""
    return (share - share.mean()).abs().idxmin()
