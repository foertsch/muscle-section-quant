"""Seeded synthetic sections for tests, goldens and the demo: one builder per assay (dystrophin, psr with an H&E
field, necrosis), plus the fake ND2 headers and descriptions the checks read. Piecewise constant images, so a whole
section compresses to a few kB and every pipeline can run on them without Cellpose (the builders return the labels a
perfect segmentation would give).
"""

from __future__ import annotations

import numpy as np


def ellipse_box(shape: tuple[int, int], cy: float, cx: float, ry: float, rx: float,
                angle: float) -> tuple[tuple[slice, slice], np.ndarray] | None:
    """A rotated ellipse as (box, mask): its tight bounding box on a canvas of shape, and the mask inside it; None
    when no pixel of it is on the canvas. ellipse() is the same on the whole canvas."""
    r = max(ry, rx) + 1
    y0, y1 = max(0, int(cy - r)), min(shape[0], int(cy + r) + 2)
    x0, x1 = max(0, int(cx - r)), min(shape[1], int(cx + r) + 2)
    if y1 <= y0 or x1 <= x0:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dy, dx = yy - cy, xx - cx
    c, s = np.cos(angle), np.sin(angle)
    u, v = c * dx + s * dy, -s * dx + c * dy
    m = (u / rx) ** 2 + (v / ry) ** 2 <= 1.0
    ys, xs = np.nonzero(m)
    if ys.size == 0:
        return None
    a, b, c_, d = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return (slice(y0 + a, y0 + b), slice(x0 + c_, x0 + d)), m[a:b, c_:d]


def ellipse(shape: tuple[int, int], cy: float, cx: float, ry: float, rx: float, angle: float) -> np.ndarray:
    """Boolean mask of a rotated ellipse (filled inside its bounding box only, so large canvases stay fast)."""
    out = np.zeros(shape, bool)
    found = ellipse_box(shape, cy, cx, ry, rx, angle)
    if found is not None:
        out[found[0]] = found[1]
    return out


def to_labels(masks: list[np.ndarray], f: int) -> np.ndarray:
    """One label per mask on the f-times coarser grid (block mean over 0.5), as a perfect segmentation would give."""
    h, w = masks[0].shape[0] // f, masks[0].shape[1] // f
    labels = np.zeros((h, w), np.uint32)
    for lab, m in enumerate(masks, start=1):
        ds = m[: h * f, : w * f].reshape(h, f, w, f).mean((1, 3)) > 0.5
        labels[ds & (labels == 0)] = lab
    return labels


def hex_grid(rng: np.random.Generator, size: int, pitch: float, jitter: float):
    """Centres on a jittered hexagonal grid over a size x size canvas."""
    rows = np.arange(-size / 2, size / 2, pitch * 0.866)
    for i, dy in enumerate(rows):
        for dx in np.arange(-size / 2, size / 2, pitch) + (pitch / 2 if i % 2 else 0):
            yield size / 2 + dy + rng.uniform(-jitter, jitter), size / 2 + dx + rng.uniform(-jitter, jitter)
