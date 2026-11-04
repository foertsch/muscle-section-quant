"""Block means and strips: the segmentation grid is a block-mean downsampling of the full-resolution scan, read in
strips of rows so a whole section never sits in memory at once. Full-resolution and downsampled grids stay aligned
because the image is trimmed to a multiple of the factor first.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
from skimage.measure import block_reduce

STRIP_ROWS_RGB = 510  # full-resolution rows per strip of an RGB uint8 scan (about 45 MB for the widest scan)
STRIP_ROWS_PLANES = 1024  # rows per strip of a 16-bit multichannel scan (about 85 MB for three channels)


def trim_to_multiple(img: np.ndarray, f: int) -> np.ndarray:
    """Crop so both axes are divisible by f, keeping full-res and downsampled grids aligned."""
    return img[: img.shape[0] // f * f, : img.shape[1] // f * f]


def block_mean(img: np.ndarray, f: int) -> np.ndarray:
    """Block mean of one plane (float32) on the f-times coarser grid."""
    return block_reduce(trim_to_multiple(img, f).astype(np.float32), (f, f), np.mean)


def iter_strips(frame: np.ndarray, f: int, rows: int = STRIP_ROWS_RGB,
                flip: bool = False) -> Iterator[tuple[int, int, np.ndarray]]:
    """(y0, y1, strip) over a (Y, X, 3) frame trimmed to a multiple of f. flip reverses the colour axis (a file
    stored as BGR comes out RGB)."""
    rows = max(f, rows // f * f)
    yt, xt = frame.shape[0] // f * f, frame.shape[1] // f * f
    sl = slice(None, None, -1) if flip else slice(None)
    for y0 in range(0, yt, rows):
        y1 = min(y0 + rows, yt)
        yield y0, y1, np.ascontiguousarray(frame[y0:y1, :xt, sl])


def block_mean_rgb(frame: np.ndarray, f: int, rows: int = STRIP_ROWS_RGB, flip: bool = False) -> np.ndarray:
    """Block-mean RGB (float32) of a (Y, X, 3) frame on the f-times coarser grid, read in strips."""
    out = np.empty((frame.shape[0] // f, frame.shape[1] // f, 3), np.float32)
    for y0, y1, strip in iter_strips(frame, f, rows, flip):
        out[y0 // f:y1 // f] = block_reduce(strip.astype(np.float32), (f, f, 1), np.mean)
    return out


def block_mean_planes(frame: np.ndarray, channels: list[int], f: int,
                      rows: int = STRIP_ROWS_PLANES) -> list[np.ndarray]:
    """Block means (float32) of the given planes of a (C, Y, X) frame, read in strips of full-resolution rows; the
    image is trimmed to a multiple of f so both grids stay aligned."""
    yt, xt = frame.shape[1] // f * f, frame.shape[2] // f * f
    out = [np.empty((yt // f, xt // f), np.float32) for _ in channels]
    rows = max(f, rows // f * f)
    for y0 in range(0, yt, rows):
        y1 = min(y0 + rows, yt)
        strip = np.asarray(frame[:, y0:y1, :xt])
        for o, c in zip(out, channels):
            o[y0 // f:y1 // f] = block_reduce(strip[c].astype(np.float32), (f, f), np.mean)
    return out
