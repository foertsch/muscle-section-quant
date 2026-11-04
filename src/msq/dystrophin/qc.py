"""Plot style and palette for every figure of the pipeline, and the per-section maps the runner draws.

Importing this module sets matplotlib's style (fonts, colours, grid) for the whole process, so the figures of
full_summary.py, plot_fiber_size.py and the scripts look alike.

Per section (sections/<sample>/ in a run folder, output/<run>/<sample>/ in a project folder):
    map_dys_membrane_specific.png  fibers coloured by membrane-specific dystrophin (dystrophin_maps)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tifffile  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from skimage.segmentation import find_boundaries  # noqa: E402

from msq.dystrophin import settings  # noqa: E402

SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
GROUP_COLORS = settings.GROUPS.colours
SEQ = LinearSegmentedColormap.from_list(
    "blue_seq", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
EXCLUDED = "#c3c2b7"
FLAGGED = "#ec835a"

METRICS = {
    "dys_membrane_specific": "Membrane dystrophin, specific\n(ridge − own cytoplasm, a.u.)",
    "dys_membrane_peak": "Membrane dystrophin ridge\n(background-subtracted, a.u.)",
    "dys_cytoplasm": "Cytoplasmic GFP\n(background-subtracted, a.u.)",
    "dys_membrane_to_cyto": "Membrane : cytoplasm ratio",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False,
})


def stretch(a: np.ndarray, lo=1.0, hi=99.7) -> np.ndarray:
    """Intensities scaled to 0..1 between the lo-th and hi-th percentile, for showing a crop."""
    l, h = np.percentile(a, [lo, hi])
    return np.clip((a.astype(np.float32) - l) / max(h - l, 1), 0, 1)


def save(fig, out: Path, **kw) -> None:
    """Write fig to out (kw: savefig's) and close it."""
    fig.savefig(out, **kw)
    plt.close(fig)


# ---------------------------------------------------------------- per-sample images


def fiber_map(labels: np.ndarray, df: pd.DataFrame, col: str, title: str, cbar_label: str,
              px_um: float, out: Path, vrange: tuple[float, float] | None = None) -> None:
    """Section map: included fibers on a sequential ramp, excluded grey, bright-cytoplasm flagged.

    vrange fixes the colour scale (e.g. shared across sections); default = this section's 2-98th pct."""
    inc = df[df["include"]]
    lo, hi = vrange if vrange is not None else np.nanpercentile(inc[col], [2, 98])
    norm = Normalize(lo, hi, clip=True)
    lut = np.tile(np.array(to_rgb(SURFACE)), (int(labels.max()) + 1, 1))
    lut[df["label"].to_numpy()] = to_rgb(EXCLUDED)
    lut[df.loc[df["flag_bright_cytoplasm"], "label"].to_numpy()] = to_rgb(FLAGGED)
    lut[inc["label"].to_numpy()] = SEQ(norm(inc[col].to_numpy()))[:, :3]
    rgb = lut[labels]
    rgb[find_boundaries(labels, mode="inner")] = to_rgb(SURFACE)

    h, w = labels.shape
    fig, ax = plt.subplots(figsize=(9, 9 * h / w + 0.6))
    ax.imshow(rgb, interpolation="antialiased")
    ax.set_axis_off()
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    bar_um = 1000
    ax.plot([w * 0.03, w * 0.03 + bar_um / px_um], [h * 0.97] * 2, color=INK, lw=2, solid_capstyle="butt")
    ax.text(w * 0.03, h * 0.955, "1 mm", color=INK, fontsize=8)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=SEQ)
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.01, extend="both")
    cb.set_label(cbar_label, color=INK2)
    cb.outline.set_visible(False)
    ax.legend(handles=[Patch(color=EXCLUDED, label="excluded (size / edge / vessel / low actin / ring < cytoplasm / other section)"),
                       Patch(color=FLAGGED, label="flagged: bright cytoplasm (necrotic / IgG+?)")],
              loc="upper left", bbox_to_anchor=(0, -0.0), ncol=2, fontsize=8, handlelength=1)
    save(fig, out, dpi=220, bbox_inches="tight")


def dystrophin_maps(dirs: list[Path]) -> None:
    """map_dys_membrane_specific.png in each section folder, on one colour scale shared by all of them."""
    allv = pd.concat([pd.read_csv(d / "fibers.csv").query("include")["dys_membrane_specific"] for d in dirs])
    shared = tuple(np.nanpercentile(allv, [2, 98]))
    for d in dirs:
        info = json.loads((d / "run_info.json").read_text(encoding="utf-8"))
        f, px_um = info["params"]["downsample"], info["pixel_size_um"]
        labels = tifffile.imread(d / f"labels_ds{f}.tif")
        print(f"[{d.name}] maps", flush=True)
        fiber_map(labels, pd.read_csv(d / "fibers.csv"), "dys_membrane_specific",
                  f"{d.name}: membrane dystrophin per fiber", METRICS["dys_membrane_specific"].replace("\n", " "),
                  px_um * f, d / "map_dys_membrane_specific.png", vrange=shared)
