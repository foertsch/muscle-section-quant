"""Segmentation as something passed in: a callable from an image to a label image.

The pipelines call Cellpose-SAM through CellposeSegmenter (named weights, loaded once per process). Tests, goldens
and the real-data checks pass FixedLabels or labels_from_file instead, so nothing needs the weights or a GPU.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import tifffile

Segmenter = Callable[[np.ndarray], np.ndarray]  # image on the segmentation grid -> labels (uint32, 0 = background)
DEFAULT_MODEL = "cpsam_v2"  # cellpose 4.2's default weights; named, so the checksum in run info means something

_MODELS: dict[tuple[str, bool], object] = {}


@dataclass(frozen=True)
class CellposeSettings:
    flow_threshold: float = 0.4
    cellprob_threshold: float = 0.0
    batch_size: int = 16


def cellpose_model(model: str = DEFAULT_MODEL, gpu: bool = True):
    """One CellposeModel per process and model name (loading the 1.2 GB weights takes seconds)."""
    key = (model, gpu)
    if key not in _MODELS:
        from cellpose import models

        _MODELS[key] = models.CellposeModel(gpu=gpu, pretrained_model=model)
    return _MODELS[key]


def weights_file(model: str = DEFAULT_MODEL) -> Path:
    """Where Cellpose keeps (or will download) the weights of a named model."""
    from cellpose.models import MODEL_DIR

    return Path(MODEL_DIR) / model


class CellposeSegmenter:
    """Cellpose-SAM on one image, with the settings the assays share."""

    def __init__(self, settings: CellposeSettings = CellposeSettings(), model: str = DEFAULT_MODEL, gpu: bool = True):
        self.settings, self.model, self.gpu = settings, model, gpu

    def __call__(self, img: np.ndarray) -> np.ndarray:
        s = self.settings
        masks, _, _ = cellpose_model(self.model, self.gpu).eval(
            img, batch_size=s.batch_size, flow_threshold=s.flow_threshold, cellprob_threshold=s.cellprob_threshold,
            normalize=True)
        return masks.astype(np.uint32)


def cellpose_for(p: Any) -> Segmenter:
    """Cellpose-SAM with the named weights and the thresholds and batch size of an assay's Params (cp_flow_threshold,
    cp_cellprob_threshold, cp_batch_size)."""
    return CellposeSegmenter(CellposeSettings(p.cp_flow_threshold, p.cp_cellprob_threshold, p.cp_batch_size))


def segmenter_for(segment: Segmenter | Mapping[str, Segmenter] | None, variant: str, p: Any) -> Segmenter:
    """The segmenter of one variant: Cellpose-SAM by default, one segmenter for every variant, or one per variant
    (a mapping, e.g. each variant's cached labels)."""
    if segment is None:
        return cellpose_for(p)
    if isinstance(segment, Mapping):
        return segment[variant]
    return segment


class FixedLabels:
    """A segmenter that returns known labels, whatever image it is given (goldens, tests, cached segmentation)."""

    def __init__(self, labels: np.ndarray):
        self.labels = np.asarray(labels).astype(np.uint32)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if img.shape[:2] != self.labels.shape:
            raise ValueError(f"the labels are {self.labels.shape} but the image is {img.shape[:2]}")
        return self.labels


def labels_from_file(path: Path) -> FixedLabels:
    """The segmentation a run cached as labels_ds<f>.tif, as a segmenter."""
    return FixedLabels(tifffile.imread(path))
