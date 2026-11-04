"""Image and geometry helpers shared by the reports (compare_report.py, decision_report.py): fiber bounding boxes,
candidate windows, SVG outlines and JPEG images for HTML pages. The outline and image code lives in msq.core.report;
the names stay here for the reports, the demo crops and the explain scripts."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from msq.core.report import outlines
from msq.core.report.images import jpeg_uri as jpeg_b64, to_uint8  # noqa: F401

CROP_PX = 1000  # 360 um at 0.36 um/px
STRIDE_PX = 200


def fiber_boxes(labels_ds: np.ndarray, f: int) -> pd.DataFrame:
    """Full-resolution bounding box per label (y0, y1, x0, x1; end-exclusive)."""
    rows = [(lab, sl[0].start * f, sl[0].stop * f, sl[1].start * f, sl[1].stop * f)
            for lab, sl in enumerate(ndi.find_objects(labels_ds), start=1) if sl is not None]
    return pd.DataFrame(rows, columns=["label", "by0", "by1", "bx0", "bx1"])


def inside_window(fib: pd.DataFrame, y0: int, x0: int, size: int) -> np.ndarray:
    return ((fib["by0"] >= y0) & (fib["by1"] <= y0 + size)
            & (fib["bx0"] >= x0) & (fib["bx1"] <= x0 + size)).to_numpy()


def candidate_windows(fib: pd.DataFrame, shape: tuple[int, int], px_um: float, size: int = CROP_PX,
                      stride: int = STRIDE_PX, min_cover: float = 0.55, min_n: int = 25) -> pd.DataFrame:
    """Windows densely covered by included fibers, with no excluded objects and decent focus."""
    sharp_floor = fib.loc[fib["include"], "actin_edge_sharpness"].quantile(0.25)
    rows = []
    for y0 in range(0, shape[0] - size + 1, stride):
        for x0 in range(0, shape[1] - size + 1, stride):
            m = inside_window(fib, y0, x0, size)
            sub = fib[m]
            inc = sub[sub["include"]]
            if len(inc) < min_n or (~sub["include"] & ~sub["flag_bright_cytoplasm"]).any():
                continue
            cover = sub["area_um2"].sum() / (size * px_um) ** 2
            if cover < min_cover or inc["actin_edge_sharpness"].median() < sharp_floor:
                continue
            rows.append({"y0": y0, "x0": x0, "n": len(sub), "cover": cover,
                         "local_median": inc["dys_membrane_specific"].median()})
    return pd.DataFrame(rows)


def outline_path(mask: np.ndarray, oy: float, ox: float, tol: float = 1.2) -> str:
    """SVG path of the longest 0.5-level contour of a boolean mask, offset by (oy, ox)."""
    return outlines.outline_path(mask, oy, ox, tol=tol)
