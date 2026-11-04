"""Images for HTML pages: arrays to data URIs, display scaling and the glass-normalised colour image."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image


def to_uint8(a: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (np.clip((a.astype(np.float32) - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def jpeg_uri(img: np.ndarray, quality: int = 88, optimize: bool = True) -> str:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality, optimize=optimize)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def png_uri(rgba: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgba).save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def data_uri(path: Path, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def rgba(mask: np.ndarray, colour: tuple) -> np.ndarray:
    """A transparent RGBA layer, painted with colour where mask is set."""
    out = np.zeros(mask.shape + (4,), np.uint8)
    out[mask] = colour
    return out


def glass_normalised(rgb: np.ndarray, glass: np.ndarray) -> np.ndarray:
    """RGB uint8 with the colour of empty glass mapped to white, so staining colours read the same on every slide."""
    return np.clip(np.rint(rgb.astype(np.float32) / glass * 255), 0, 255).astype(np.uint8)
