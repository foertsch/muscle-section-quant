"""Segmentation review: what Cellpose-SAM outlined, how much red ("double positive") each object holds, and how the
segmentation changes with the settings.

Reads a run folder with sections/<variant>/<sample>/ (section.py) and samples.csv (checks.py, paths to the scans),
and writes to OUT:
    summary.csv               one row per section and variant: objects, coverage of yellow, red inside outlines,
                              objects per core-red bin, merge candidates, run time
    agreement.csv             each variant against base: share of base objects matched (IoU >= 0.5), mean IoU,
                              merges and splits
    objects_all.csv           every object of every section (base variant), for plots and the filter decision
    overview_<sample>.png     the section with outlines coloured by core-red bin; yellow that no outline covers in
                              magenta
    crops_<sample>.png        random 500 um windows: image with outlines | pixel classes with outlines
    gallery_<bin>.png         objects from every section in one core-red bin (rows: sections)
    gallery_merge.png         the least convex large objects (possible merged fibers)
    variants_<sample>.png     the same windows outlined by each variant
    distributions.png         core, edge-band and ring red fractions per section; core red against size

Usage:
    python -m msq.psr.review_segmentation RUN --out RUN/review
"""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tifffile  # noqa: E402
from skimage.transform import resize  # noqa: E402

from msq.core import review  # noqa: E402
from msq.core.objects import upsample_labels  # noqa: E402
from msq.core.report.images import glass_normalised  # noqa: E402
from msq.psr import checks, settings  # noqa: E402
from msq.psr import section as sec  # noqa: E402

BINS = [("clean", 0.0, 0.05), ("low", 0.05, 0.15), ("mid", 0.15, 0.30), ("high", 0.30, 0.50), ("red", 0.50, 1.01)]
BIN_COLOURS = {"clean": (0, 200, 80), "low": (255, 230, 0), "mid": (255, 140, 0), "high": (255, 0, 0),
               "red": (180, 0, 255), "no_core": (120, 120, 120)}
CLASS_RGB = np.array([[255, 255, 255], [220, 20, 60], [255, 215, 0], [70, 130, 180]], np.uint8)  # bg red yellow other
MERGE_SOLIDITY = 0.85
WHITE = (255, 255, 255)


def core_bin(red_frac_core: pd.Series) -> pd.Series:
    out = pd.Series("no_core", index=red_frac_core.index, dtype=object)
    for name, lo, hi in BINS:
        out[(red_frac_core >= lo) & (red_frac_core < hi)] = name
    return out


# ---------------------------------------------------------------- loading


class Section:
    def __init__(self, folder: Path):
        self.folder = folder
        self.info = json.loads((folder / "section.json").read_text())
        self.f = self.info["params"]["downsample"]
        self.objects = pd.read_csv(folder / "objects.csv")
        if len(self.objects):
            self.objects["bin"] = core_bin(self.objects["red_frac_core"])
        self._cache: dict[str, np.ndarray] = {}

    def image(self, name: str) -> np.ndarray:
        if name not in self._cache:
            self._cache[name] = tifffile.imread(self.folder / f"{name}_ds{self.f}.tif")
        return self._cache[name]

    @property
    def params(self) -> sec.Params:
        return sec.Params(**self.info["params"])


def full_res_crop(frame: np.ndarray, y0: int, x0: int, h: int, w: int) -> np.ndarray:
    """RGB uint8 crop of the scan (file order BGR flipped), clipped to the image."""
    y0, x0 = max(0, y0), max(0, x0)
    crop = np.asarray(frame[y0:y0 + h, x0:x0 + w])
    return crop[..., ::-1] if settings.FILE_ORDER == "BGR" else crop


def labels_full(s: Section, y0: int, x0: int, h: int, w: int) -> np.ndarray:
    """The segmentation grid's labels covering a full-resolution window, upsampled by nearest neighbour."""
    return upsample_labels(s.image("labels"), s.f, (h, w), (y0, x0), clip=True)


def window_panels(s: Section, frame: np.ndarray, y0: int, x0: int, size: int, colours=None, highlight=None,
                  step: int = 1) -> tuple[np.ndarray, np.ndarray]:
    rgb = full_res_crop(frame, y0, x0, size, size)
    lab = labels_full(s, max(0, y0), max(0, x0), *rgb.shape[:2])
    bg = np.array(s.info["background_rgb"])
    classes = sec.classify(rgb, bg, s.params)
    img = review.draw_outlines(glass_normalised(rgb, bg.astype(np.float32)), lab, colours, WHITE, 2, highlight)
    cls = review.draw_outlines(CLASS_RGB[classes], lab, colours, (0, 0, 0), 2, highlight)
    return img[::step, ::step], cls[::step, ::step]


# ---------------------------------------------------------------- numbers


def coverage_away_from_edge(s: Section, edge_um: float = 100.0) -> float:
    """Share of yellow inside outlines, leaving out the outer edge_um of the section, where folds and torn tissue
    sit (a stand-in until there is a fold mask)."""
    from scipy import ndimage as ndi

    inner = ndi.binary_erosion(s.image("section").astype(bool), iterations=max(1, int(edge_um / s.info["grid_um"])))
    yellow = s.image("classes")[..., sec.YELLOW].astype(np.float64)
    total = yellow[inner].sum()
    return float(yellow[inner & (s.image("labels") > 0)].sum() / total) if total else np.nan


def summary_row(s: Section, variant: str) -> dict:
    o, info = s.objects, s.info
    use = o[o["in_section"]] if len(o) else o
    row = {"sample": info["sample"], "variant": variant, "n_objects": info["n_objects"],
           "n_in_section": info["n_objects_in_section"], "coverage_yellow": info["coverage_yellow"],
           "red_in_masks": info["red_in_masks"], "red_fraction": info["red_fraction"],
           "median_area_um2": use["area_um2"].median() if len(use) else np.nan,
           "median_min_feret_um": use["min_feret_um"].median() if len(use) else np.nan,
           "median_red_frac_band": use["red_frac_band"].median() if len(use) else np.nan,
           "median_red_frac_ring": use["red_frac_ring"].median() if len(use) else np.nan}
    for name, _, _ in BINS + [("no_core", 0, 0)]:
        row[f"n_{name}"] = int((use["bin"] == name).sum()) if len(use) else 0
    big = use["area_um2"] > 2 * use["area_um2"].median() if len(use) else []
    row["n_merge_candidates"] = int(((use["solidity"] < MERGE_SOLIDITY) & big).sum()) if len(use) else 0
    row["n_small_lt50um2"] = int((use["area_um2"] < 50).sum()) if len(use) else 0
    row["collagen_outside_fraction"] = info.get("collagen_outside_fraction", np.nan)
    if len(use):
        px = use[[f"{z}_{c}" for z in ("core", "band") for c in sec.CLASS_NAMES]].sum(axis=1)
        yellow_share = (use["core_yellow"] + use["band_yellow"]) / px
        row["median_red_share_outline"] = ((use["core_red"] + use["band_red"]) / px).median()
        row["median_yellow_share_outline"] = yellow_share.median()
        d_mask = 2 * np.sqrt(use["area_um2"] / np.pi)
        row["median_diameter_bias_um"] = (d_mask - d_mask * np.sqrt(yellow_share)).median()
    row["coverage_away_from_edge"] = coverage_away_from_edge(s)
    row.update({f"sec_{k}": v for k, v in info["seconds"].items()})
    row["peak_memory_gb"] = info.get("peak_memory_gb")
    return row


def agreement(base: Section, other: Section) -> dict:
    """Match other's objects to base's on base's grid (nearest-neighbour resampling)."""
    a = base.image("labels")
    b = other.image("labels")
    if b.shape != a.shape:
        b = resize(b, a.shape, order=0, preserve_range=True, anti_aliasing=False).astype(np.uint32)
    pairs = review.iou_pairs(a, b)
    best = review.best_iou(pairs, "a", range(1, a.max() + 1))
    merges = (pairs["cover_a"] >= 0.5).groupby(pairs["b"]).sum()  # other objects covering half of 2+ base objects
    splits = (pairs["cover_a"] >= 0.25).groupby(pairs["a"]).sum()  # base objects split over 2+ other objects
    return {"sample": base.info["sample"], "variant": other.info["variant"], "n_base": int(a.max()),
            "n_variant": int(b.max()), "matched_iou50": float((best >= 0.5).mean()),
            "mean_best_iou": float(best.mean()), "merges": int((merges >= 2).sum()),
            "splits": int((splits >= 2).sum()),
            "unmatched_variant": int((review.best_iou(pairs, "b", range(1, b.max() + 1)) < 0.5).sum())}


# ---------------------------------------------------------------- figures


def overview(s: Section, out: Path, max_px: int = 2400) -> None:
    rgb = s.image("rgb").copy()
    lab = s.image("labels")
    classes = s.image("classes")
    sect = s.image("section").astype(bool)
    uncovered = sect & (lab == 0) & (classes[..., sec.YELLOW] >= 50)
    rgb[uncovered] = (255, 0, 255)
    colours = {}
    if len(s.objects):
        colours = {int(r.label): BIN_COLOURS[r.bin] for r in s.objects.itertuples()}
    img = review.draw_outlines(rgb, lab, colours, WHITE)
    step = max(1, int(np.ceil(max(img.shape[:2]) / max_px)))
    fig, ax = plt.subplots(figsize=(img.shape[1] / step / 100, img.shape[0] / step / 100 + 0.6), dpi=100)
    ax.imshow(img[::step, ::step])
    ax.set_axis_off()
    counts = s.objects["bin"].value_counts().to_dict() if len(s.objects) else {}
    legend = "   ".join(f"{k}: {counts.get(k, 0)}" for k in BIN_COLOURS)
    ax.set_title(f"{s.info['sample']}: outline colour = core red fraction (clean <5%, low <15%, mid <30%, high <50%, "
                 f"red >=50%); magenta = yellow not covered\n{legend}", fontsize=9)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def random_windows(s: Section, n: int, size_full: int, seed: int) -> list[tuple[int, int]]:
    sect = s.image("section").astype(bool)
    rng = np.random.default_rng(seed)
    ys, xs = np.nonzero(sect)
    picks = rng.choice(len(ys), size=min(n * 20, len(ys)), replace=False)
    out = []
    for i in picks:
        y, x = ys[i] * s.f - size_full // 2, xs[i] * s.f - size_full // 2
        if all(abs(y - a) > size_full or abs(x - b) > size_full for a, b in out):
            out.append((int(y), int(x)))
        if len(out) == n:
            break
    return out


def crops_figure(s: Section, frame: np.ndarray, out: Path, windows: list[tuple[int, int]], size_full: int) -> None:
    colours = {int(r.label): BIN_COLOURS[r.bin] for r in s.objects.itertuples()} if len(s.objects) else {}
    fig, axes = plt.subplots(len(windows), 2, figsize=(10, 5 * len(windows)), dpi=110, squeeze=False)
    for i, (y, x) in enumerate(windows):
        img, cls = window_panels(s, frame, y, x, size_full, colours, step=2)
        axes[i, 0].imshow(img)
        axes[i, 1].imshow(cls)
        axes[i, 0].set_title(f"y={y}, x={x} (full-res px)", fontsize=8)
        for a in axes[i]:
            a.set_axis_off()
    fig.suptitle(f"{s.info['sample']}: 500 um windows. Left: image, outlines coloured by core red bin. Right: pixel "
                 "classes (red, yellow, blue = other, white = background)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def gallery(rows: list[tuple[Section, np.ndarray, pd.DataFrame]], title: str, out: Path, tile_um: float = 100,
            per_row: int = 5) -> None:
    rows = [r for r in rows if len(r[2])]
    if not rows:
        return
    fig, axes = plt.subplots(len(rows), per_row * 2, figsize=(per_row * 4.2, len(rows) * 2.3), dpi=110,
                             squeeze=False)
    for i, (s, frame, objs) in enumerate(rows):
        size = int(tile_um / s.info["pixel_size_um"])
        for j in range(per_row):
            ax_img, ax_cls = axes[i, 2 * j], axes[i, 2 * j + 1]
            ax_img.set_axis_off()
            ax_cls.set_axis_off()
            if j >= len(objs):
                continue
            o = objs.iloc[j]
            y, x = int(o["centroid_y"]) - size // 2, int(o["centroid_x"]) - size // 2
            img, cls = window_panels(s, frame, y, x, size, highlight=int(o["label"]))
            ax_img.imshow(img)
            ax_cls.imshow(cls)
            ax_img.set_title(f"#{int(o['label'])} core {o['red_frac_core']:.0%} band {o['red_frac_band']:.0%}\n"
                             f"{o['area_um2']:.0f} um2, sol {o['solidity']:.2f}", fontsize=6.5)
        axes[i, 0].text(-0.08, 0.5, s.info["sample"].replace("QUAD_", ""), transform=axes[i, 0].transAxes,
                        rotation=90, va="center", ha="right", fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def variants_figure(sections: dict[str, Section], frame: np.ndarray, out: Path, windows: list[tuple[int, int]],
                    size_full: int) -> None:
    names = list(sections)
    fig, axes = plt.subplots(len(windows), len(names), figsize=(3.6 * len(names), 3.6 * len(windows)), dpi=110,
                             squeeze=False)
    for i, (y, x) in enumerate(windows):
        for j, v in enumerate(names):
            s = sections[v]
            rgb = full_res_crop(frame, y, x, size_full, size_full)
            lab = labels_full(s, max(0, y), max(0, x), *rgb.shape[:2])
            glass = np.array(s.info["background_rgb"], np.float32)
            axes[i, j].imshow(review.draw_outlines(glass_normalised(rgb, glass), lab, None, (0, 120, 255), 2)[::3, ::3])
            axes[i, j].set_axis_off()
            if i == 0:
                axes[i, j].set_title(v, fontsize=10)
    fig.suptitle(f"{next(iter(sections.values())).info['sample']}: the same 300 um windows outlined by each variant",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def distributions(objs: pd.DataFrame, out: Path) -> None:
    samples = list(dict.fromkeys(objs["sample"]))
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), dpi=110)
    bins = np.linspace(0, 1, 41)
    for z, ax in zip(["core", "band", "ring"], axes[:3]):
        for smp in samples:
            v = objs.loc[objs["sample"] == smp, f"red_frac_{z}"].dropna()
            ax.hist(v, bins=bins, histtype="step", label=smp.replace("QUAD_", ""), density=True)
        ax.set_yscale("log")
        ax.set_xlabel(f"red fraction, {z}")
    axes[0].legend(fontsize=7)
    for smp in samples:
        o = objs[objs["sample"] == smp]
        axes[3].scatter(o["area_um2"], o["red_frac_core"], s=2, alpha=0.3, label=smp)
    axes[3].set_xscale("log")
    axes[3].set_xlabel("area (um2)")
    axes[3].set_ylabel("red fraction, core")
    fig.suptitle("Objects in the main section (base variant)")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------- main


class Run(NamedTuple):
    paths: pd.Series                         # sample -> scan
    sections: dict[str, dict[str, Section]]  # variant -> sample -> section


def load_run(run: Path, args) -> Run:
    paths = checks.read_samples(run / "samples.csv").set_index("sample")["path"]
    root = run / "sections"
    variants = sorted(d.name for d in root.iterdir() if d.is_dir()) if root.exists() else []
    if "base" not in variants:
        raise SystemExit(f"no base variant in {root}")
    return Run(paths, {v: {d.name: Section(d) for d in sorted((root / v).iterdir()) if (d / "section.json").exists()}
                       for v in variants})


def summary_rows(run: Run) -> list[dict]:
    return [summary_row(s, v) for v, sections in run.sections.items() for s in sections.values()]


def agreement_rows(run: Run) -> list[dict]:
    base = run.sections["base"]
    return [{**agreement(base[smp], s), "variant": v} for v, sections in run.sections.items() if v != "base"
            for smp, s in sections.items() if smp in base]


def arguments(ap) -> None:
    ap.add_argument("--crops", type=int, default=6, help="random 500 um windows per section")
    ap.add_argument("--per-row", type=int, default=5, help="objects per section in each gallery")


def figures(run: Run, args) -> None:
    out, base = args.out, run.sections["base"]
    objs = pd.concat([s.objects.assign(sample=smp) for smp, s in base.items()], ignore_index=True)
    objs.to_csv(out / "objects_all.csv", index=False)
    distributions(objs[objs["in_section"]], out / "distributions.png")
    rng = np.random.default_rng(0)
    with review.open_scans({smp: run.paths[smp] for smp in base}, sec.open_scan) as scans:
        for smp, s in base.items():
            print(smp, flush=True)
            overview(s, out / f"overview_{smp}.png")
            size = int(500 / s.info["pixel_size_um"])
            crops_figure(s, scans[smp][0], out / f"crops_{smp}.png", random_windows(s, args.crops, size, 1), size)
            others = {v: sections[smp] for v, sections in run.sections.items() if smp in sections}
            if len(others) > 1:
                small = int(300 / s.info["pixel_size_um"])
                variants_figure(others, scans[smp][0], out / f"variants_{smp}.png", random_windows(s, 4, small, 2),
                                small)

        for name, lo, hi in BINS:
            rows = []
            for smp, s in base.items():
                o = s.objects[s.objects["in_section"] & (s.objects["bin"] == name)]
                rows.append((s, scans[smp][0], o.sample(min(args.per_row, len(o)), random_state=rng.integers(1e9))))
            gallery(rows, f"Objects with {lo:.0%} to {min(hi, 1):.0%} red in the core (cyan outline; right panel: "
                          "classes)", out / f"gallery_{name}.png", per_row=args.per_row)
        rows = []
        for smp, s in base.items():
            o = s.objects[s.objects["in_section"]]
            o = o[o["area_um2"] > 2 * o["area_um2"].median()].nsmallest(args.per_row, "solidity")
            rows.append((s, scans[smp][0], o))
        gallery(rows, "Least convex large objects (possible merged fibers)", out / "gallery_merge.png",
                tile_um=160, per_row=args.per_row)


REVIEW = review.Review(__doc__, load_run, summary_rows, agreement_rows, figures, arguments)
main = partial(review.main, REVIEW)

if __name__ == "__main__":
    main()
