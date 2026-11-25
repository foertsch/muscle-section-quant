"""Folder layouts, so every script agrees on where results live.

The code lives in this package; the scans and results live elsewhere. A run writes one self-contained folder:

    samples.csv               sample table with each scan's imaging settings (checks.py)
    checks.txt                the problems found before anything slow runs
    sections/<sample>/        per-section output of the section step
    tables/ figures/ reports/ results to look at or share
"""

from pathlib import Path

CODE = Path(__file__).resolve().parent  # the package folder, where the HTML templates are
