"""The segmentation review that psr and necrosis share: how well two outlinings of a section agree (iou_pairs,
best_iou), outlines drawn in colour on an image (draw_outlines), the scans opened for the pictures (open_scans) and the
command (main). What an assay brings is a Review: its extra arguments, how it loads a run, its summary and agreement
rows and its pictures."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import pandas as pd
from skimage.segmentation import find_boundaries

GREY, HIGHLIGHT = (120, 120, 120), (0, 255, 255)


def iou_pairs(a: np.ndarray, b: np.ndarray) -> pd.DataFrame:
    """The overlapping pairs of objects of two label images on one grid, sorted by a, then b: the labels (a, b), the
    pixels they share (inter), their IoU and cover_a, the share of a's area that b covers."""
    both = (a > 0) & (b > 0)
    key, inter = np.unique(a[both].astype(np.uint64) << np.uint64(32) | b[both].astype(np.uint64), return_counts=True)
    ia, ib = (key >> np.uint64(32)).astype(np.int64), (key & np.uint64(0xFFFFFFFF)).astype(np.int64)
    area_a = np.bincount(a.ravel(), minlength=int(a.max()) + 1)
    area_b = np.bincount(b.ravel(), minlength=int(b.max()) + 1)
    return pd.DataFrame({"a": ia, "b": ib, "inter": inter, "iou": inter / (area_a[ia] + area_b[ib] - inter),
                         "cover_a": inter / area_a[ia]})


def best_iou(pairs: pd.DataFrame, side: str, labels) -> pd.Series:
    """The best IoU of each of the labels on one side ("a" or "b") of iou_pairs; 0 for an object that overlaps none."""
    return pairs.groupby(side)["iou"].max().reindex(labels, fill_value=0.0)


def draw_outlines(rgb: np.ndarray, labels: np.ndarray, colours: dict[int, tuple] | None = None, default=GREY,
                  width: int = 1, highlight: int | None = None) -> np.ndarray:
    """A copy of rgb with the inner boundary of every object drawn in its colour (colours: label -> RGB; default for
    the rest), width pixels thick; the object highlight is outlined on top in HIGHLIGHT."""
    out = rgb.copy()
    edges = find_boundaries(labels, mode="inner")
    for _ in range(width - 1):
        edges |= find_boundaries(np.where(edges, labels, 0), mode="outer") & (labels > 0)
    colours = colours or {}
    lut = np.empty((max(int(labels.max()), max(colours, default=0)) + 1, 3), np.uint8)  # colours cover the section
    lut[:] = default
    for lab, col in colours.items():
        lut[lab] = col
    out[edges] = lut[labels[edges]]
    if highlight is not None:
        e = find_boundaries(labels == highlight, mode="inner")
        e |= find_boundaries(np.where(e, 1, 0), mode="outer") & (labels == highlight)
        out[e] = HIGHLIGHT
    return out


@contextmanager
def open_scans(paths: dict[str, Path], open_frame: Callable) -> Iterator[dict]:
    """The scans of a review opened for its pictures: sample -> (frame, ...what open_frame returns after the reader).
    Each reader is closed on leaving. open_frame(path) -> (reader, frame, ...)."""
    opened = {k: open_frame(p) for k, p in paths.items()}
    readers, frames = [v[0] for v in opened.values()], {k: v[1:] for k, v in opened.items()}
    del opened
    try:
        yield frames
    finally:
        frames.clear()
        for reader in readers:
            reader.close()


@dataclass(frozen=True)
class Review:
    """What an assay brings to main: doc is the command's help; load(run, args) reads the run folder into whatever
    the others take; summary(loaded) and pairs(loaded) give the rows of summary.csv and agreement.csv; render(loaded,
    args) writes everything else into args.out; arguments(parser) adds the assay's own options. wide prints the summary
    with one row per column."""
    doc: str
    load: Callable[[Path, argparse.Namespace], Any]
    summary: Callable[[Any], list[dict]]
    pairs: Callable[[Any], list[dict]]
    render: Callable[[Any, argparse.Namespace], None]
    arguments: Callable[[argparse.ArgumentParser], None] = lambda parser: None
    wide: bool = False


def main(review: Review, argv: list[str] | None = None) -> None:
    """RUN --out OUT [the assay's options]: summary.csv and agreement.csv, then the assay's pictures."""
    ap = argparse.ArgumentParser(description=review.doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    review.arguments(ap)
    args = ap.parse_args(argv)
    loaded = review.load(args.run, args)
    args.out.mkdir(parents=True, exist_ok=True)
    summary, pairs = pd.DataFrame(review.summary(loaded)), pd.DataFrame(review.pairs(loaded))
    summary.to_csv(args.out / "summary.csv", index=False)
    pairs.to_csv(args.out / "agreement.csv", index=False)
    review.render(loaded, args)
    print(summary.T.to_string() if review.wide else summary.to_string(index=False))
    if len(pairs):
        print(pairs.to_string(index=False))
