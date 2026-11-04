"""Full-dataset summary: one positive/negative threshold per muscle from pooled negative controls, the call at 50% of
the ring plus two stricter levels, per-section and per-animal results.

Rule (see GLOSSARY.md): every fiber's ring is 36 pieces (sector values). Per muscle, all negative-control
(untreated mdx) fibers are pooled and the threshold is set at the 98th percentile of their 18th-brightest piece (FPR),
so 2% of control fibers have at least half their ring above it. A fiber is positive when at least 18 of its 36
pieces (50%) reach the threshold; this is the settled call. The stricter levels 60 / 70% (22 / 26 pieces) use the
same threshold, are nested inside the 50% call, and fewer control fibers pass them (the control false-positive rate is
reported per level). Revertant fibers in mdx are real dystrophin and count as control positives.

Leave-one-animal-out: each control animal is scored with a threshold set from the other control animals of that
muscle, which shows what false-positive rate to expect on an unseen control animal. Control animals are reported
by these held-out calls: in calls, sections and animals each control section is called against the threshold
from the other control animals of its muscle, so its % positive is what an unseen control would give rather than
about 2% by construction. Treated sections are called against the threshold from all controls. A muscle with one
control animal has no held-out threshold; that animal is called against its own threshold and marked. The threshold
column says which.

Sections are weighted equally within an animal; animals are the unit of comparison, so results are
descriptive.

Usage:
    python -m msq.dystrophin.full_summary output/full_20x --manifest manifest_20x.csv
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from msq.dystrophin import calibrate as cal  # noqa: E402
from msq.dystrophin import paths  # noqa: E402
from msq.dystrophin import qc  # noqa: E402
from msq.dystrophin import settings  # noqa: E402

COVERAGES = (0.5, 0.6, 0.7)  # stringency levels: share of the 36 ring pieces that must reach the threshold
THRESHOLD_COVERAGE = settings.CALL_LEVEL  # the threshold is calibrated at this level, which is also the call
# share of pooled control fibers at or above the threshold at THRESHOLD_COVERAGE: the threshold is their 98th percentile
FPR = settings.FPR
GROUPS = settings.GROUPS
THRESHOLD_FROM = {"all": "all controls", "held_out": "other control animals", "alone": "this animal alone",
            "borrowed": "borrowed threshold"}  # the threshold_from column: which threshold a section was called against


def pct(c: float) -> str:
    return f"{round(100 * c)}"


# ---------------------------------------------------------------- calibration


def fiber_scores(samples: dict[str, dict], coverages=COVERAGES) -> dict[str, pd.DataFrame]:
    """Per sample: one row per called fiber, one score column per coverage."""
    out = {}
    for s, d in samples.items():
        out[s] = pd.concat({f"score_{pct(c)}": cal.coverage_score(d["sectors"], c) for c in coverages}, axis=1)
    return out


def calibrate_muscle(scores: dict[str, pd.DataFrame], ctrl: list[str], coverages=COVERAGES,
                     fpr: float = FPR) -> dict[float, dict]:
    """One threshold per muscle: the (1 - fpr) percentile of pooled control fibers' scores at THRESHOLD_COVERAGE. Every
    stringency level uses that same threshold; reports the pooled control false-positive rate per level."""
    threshold = cal.threshold_from_controls(
        pd.concat([scores[s][f"score_{pct(THRESHOLD_COVERAGE)}"] for s in ctrl]), fpr)
    out = {}
    for c in coverages:
        pooled = pd.concat([scores[s][f"score_{pct(c)}"] for s in ctrl])
        out[c] = {"sector_threshold": threshold, "ctrl_fpr_pooled": float((pooled >= threshold).mean()),
                  "n_ctrl_fibers": len(pooled)}
    return out


def fixed_threshold(scores: dict[str, pd.DataFrame], ctrl: list[str], threshold: float,
                    coverages=COVERAGES) -> dict[float, dict]:
    """A threshold given from outside (borrowed from an earlier run), in the shape calibrate_muscle returns. The control
    false-positive rate is None when this run has no controls of the muscle."""
    out = {}
    for c in coverages:
        pooled = pd.concat([scores[s][f"score_{pct(c)}"] for s in ctrl]) if ctrl else pd.Series(dtype=float)
        out[c] = {"sector_threshold": float(threshold), "n_ctrl_fibers": len(pooled),
                  "ctrl_fpr_pooled": float((pooled >= threshold).mean()) if len(pooled) else None}
    return out


def held_out_thresholds(scores: dict[str, pd.DataFrame], ctrl: list[str], animal_of: dict[str, str],
                  coverages=COVERAGES, fpr: float = FPR) -> dict[str, dict[float, dict]]:
    """Per control animal, the calibration from the other control animals of the muscle (calibrate_muscle on
    them): the threshold that animal is called against, and that leave_one_animal_out scores it with."""
    return {a: calibrate_muscle(scores, [s for s in ctrl if animal_of[s] != a], coverages, fpr)
            for a in sorted({animal_of[s] for s in ctrl})}


def leave_one_animal_out(scores: dict[str, pd.DataFrame], ctrl: list[str], animal_of: dict[str, str],
                         coverages=COVERAGES, fpr: float = FPR,
                         held: dict[str, dict[float, dict]] | None = None) -> pd.DataFrame:
    held = held if held is not None else held_out_thresholds(scores, ctrl, animal_of, coverages, fpr)
    rows = []
    for a in sorted({animal_of[s] for s in ctrl}):
        test = [s for s in ctrl if animal_of[s] == a]
        th = held[a]
        for c in coverages:
            col = f"score_{pct(c)}"
            v = pd.concat([scores[s][col] for s in test])
            rows.append({"held_out_animal": a, "coverage": c, "sector_threshold": th[c]["sector_threshold"],
                         "n_fibers": len(v), "fpr": float((v >= th[c]["sector_threshold"]).mean())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- calls and tables


def call_fibers(d: dict, score: pd.DataFrame, th: dict[float, dict], with_threshold: bool = False) -> pd.DataFrame:
    """All included fibers; called ones get a score and a call per coverage. with_threshold: add the threshold column
    (the threshold the calls were made against)."""
    f = d["fibers"]
    out = pd.DataFrame({"label": f.index, "dys_membrane_specific": f["dys_membrane_specific"].to_numpy(),
                        "area_um2": f["area_um2"].to_numpy()}).set_index("label")
    out = out.join(score)
    for c, t in th.items():
        col = f"score_{pct(c)}"
        out[f"call_{pct(c)}"] = np.where(out[col].isna(), "not called",
                                         np.where(out[col] >= t["sector_threshold"], "positive", "negative"))
    if with_threshold:
        out["threshold"] = th[THRESHOLD_COVERAGE]["sector_threshold"]
    return out.reset_index()


def section_table(calls: dict[str, pd.DataFrame], meta: pd.DataFrame, coverages=COVERAGES,
                  thresholds: dict[str, tuple[float, str]] | None = None) -> pd.DataFrame:
    """One row per section. thresholds: {sample: (threshold, threshold_from)} the section was called against, as two
    columns."""
    rows = []
    for s, c in calls.items():
        m = meta.loc[s]
        called = c[c[f"call_{pct(THRESHOLD_COVERAGE)}"] != "not called"]
        row = {"sample": s, "group": m["group"], "muscle": m["muscle"], "animal": str(m["animal"]),
               "section": int(m["section"]), "n_fibers": len(c), "n_called": len(called)}
        for cov in coverages:
            pos = int((called[f"call_{pct(cov)}"] == "positive").sum())
            row[f"n_positive_{pct(cov)}"] = pos
            row[f"pct_positive_{pct(cov)}"] = 100 * pos / len(called) if len(called) else np.nan
        row["median_dys_membrane_specific"] = c["dys_membrane_specific"].median()
        if thresholds is not None:
            row["threshold"], row["threshold_from"] = thresholds[s]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["muscle", "group", "animal", "section"]).reset_index(drop=True)


def animal_table(sections: pd.DataFrame, coverages=COVERAGES) -> pd.DataFrame:
    agg = {"n_sections": ("sample", "size"), "n_called": ("n_called", "sum"),
           "median_dys_membrane_specific": ("median_dys_membrane_specific", "mean")}
    for c in coverages:
        k = pct(c)
        agg[f"pct_positive_{k}"] = (f"pct_positive_{k}", "mean")
        agg[f"pct_positive_{k}_min"] = (f"pct_positive_{k}", "min")
        agg[f"pct_positive_{k}_max"] = (f"pct_positive_{k}", "max")
        agg[f"n_positive_{k}"] = (f"n_positive_{k}", "sum")
    out = sections.groupby(["muscle", "group", "animal"]).agg(**agg).reset_index()
    for c in coverages:
        k = pct(c)
        out[f"pct_positive_{k}_pooled"] = 100 * out[f"n_positive_{k}"] / out["n_called"]
    if "threshold_from" in sections:  # one threshold per animal and muscle: all its sections share it
        out["threshold_from"] = sections.groupby(["muscle", "group", "animal"])["threshold_from"].first().to_numpy()
    return out


# ---------------------------------------------------------------- plots


def spread(ys: list[float], gap: float) -> list[float]:
    """Label positions near ys, at least `gap` apart, keeping their order."""
    order = np.argsort(ys)
    out = np.array(ys, dtype=float)
    for i in range(1, len(order)):
        out[order[i]] = max(out[order[i]], out[order[i - 1]] + gap)
    return list(out)


def animal_order(sections: pd.DataFrame, muscle: str) -> list[tuple[str, str]]:
    s = sections[sections["muscle"] == muscle]
    return [(g, a) for g in GROUPS.order for a in sorted(s.loc[s["group"] == g, "animal"].unique())]


def muscles_in(keys) -> list[str]:
    return [m for m in settings.MUSCLES if m in set(keys)]


def ctrl_held_out(th: dict) -> str:
    """'controls held out at 50/60/70%: a / b / c%' (mean over the control animals of the false-positive rate
    against the threshold from the other control animals), or ctrl_passing when there is no held-out check."""
    if any("loao_fpr_mean" not in th[c] for c in COVERAGES):
        return ctrl_passing(th)
    return "controls held out at 50/60/70%: " + " / ".join(f"{100 * th[c]['loao_fpr_mean']:.1f}"
                                                           for c in COVERAGES) + "%"


def ctrl_passing(th: dict) -> str:
    """'controls passing at 50/60/70%: a / b / c%', or a note when the threshold was borrowed and this run has no
    controls."""
    if any(th[c]["ctrl_fpr_pooled"] is None for c in COVERAGES):
        return "threshold borrowed, no controls in this run"
    return "controls passing at 50/60/70%: " + " / ".join(f"{100 * th[c]['ctrl_fpr_pooled']:.1f}" for c in COVERAGES) + "%"


def legend(fig) -> None:
    handles = [plt.Rectangle((0, 0), 1, 1, color=qc.GROUP_COLORS[g], alpha=0.5) for g in GROUPS.order]
    fig.legend(handles, [GROUPS.labels[g] for g in GROUPS.order], loc="upper right", ncol=2, fontsize=8)


def plot_pct_positive(sections: pd.DataFrame, thresholds: dict, out: Path, fpr: float = FPR) -> None:
    """Per animal, one bar per stringency level (lighter = 50%, darker = 70%); dots = sections."""
    muscles = muscles_in(thresholds)
    alphas = dict(zip(COVERAGES, (0.3, 0.6, 0.95)))
    width = 0.26
    fig, axs = plt.subplots(1, len(muscles), figsize=(6.2 * len(muscles), 4.6), sharey=True, squeeze=False)
    for ax, m in zip(axs[0], muscles):
        order = animal_order(sections, m)
        for i, (g, a) in enumerate(order):
            s = sections[(sections["muscle"] == m) & (sections["animal"] == a) & (sections["group"] == g)]
            for j, c in enumerate(COVERAGES):
                x = i + (j - 1) * width
                vals = s[f"pct_positive_{pct(c)}"]
                ax.bar(x, vals.mean(), width=width * 0.92, color=qc.GROUP_COLORS[g], alpha=alphas[c], linewidth=0)
                ax.scatter(np.full(len(vals), x), vals, s=10, color=qc.INK, alpha=0.55, linewidth=0, zorder=3)
                ax.annotate(f"{vals.mean():.1f}", (x, max(vals.mean(), vals.max())), xytext=(0, 3),
                            textcoords="offset points", ha="center", fontsize=6.5, color=qc.INK)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([a for _, a in order], color=qc.INK)
        ax.grid(axis="x", visible=False)
        th = thresholds[m]
        ax.set_title(f"{m}: threshold {th[THRESHOLD_COVERAGE]['sector_threshold']:.0f}; {ctrl_held_out(th)}",
                     loc="left", fontsize=10, color=qc.INK)
        ax.set_xlabel("animal (bars left to right: 50%, 60%, 70% of the ring required)")
    axs[0, 0].set_ylabel("Fibers called dystrophin-positive (%)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=qc.GROUP_COLORS[g], alpha=0.6) for g in GROUPS.order]
    handles += [plt.Rectangle((0, 0), 1, 1, color=qc.INK2, alpha=alphas[c]) for c in COVERAGES]
    labels = ([GROUPS.labels[g] for g in GROUPS.order]
              + [f"≥ {pct(c)}% of ring above the threshold" for c in COVERAGES])
    fig.legend(handles, labels, loc="upper right", ncol=5, fontsize=8)
    fig.text(0.01, 0.01, f"Bars = mean of the animal's sections; dots = sections. One threshold per muscle = "
                         f"{100 * (1 - fpr):.0f}th percentile of pooled mdx control fibers' 18th brightest ring piece; "
                         "60% and 70% need more of the ring above the same threshold. Control animals: each against "
                         "the threshold from the other control animals (held out).",
             fontsize=7, color=qc.MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    qc.save(fig, out, dpi=200)


def plot_call(sections: pd.DataFrame, thresholds: dict, out: Path, fpr: float = FPR) -> None:
    """The call alone (THRESHOLD_COVERAGE): per animal, the mean % positive of its sections (bar) and each section
    (dot)."""
    muscles = muscles_in(thresholds)
    orders = {m: animal_order(sections, m) for m in muscles}
    widths = [max(3.0, 0.75 * len(orders[m]) + 1.2) for m in muscles]
    fig, axs = plt.subplots(1, len(muscles), figsize=(sum(widths), 4.0), sharey=True, squeeze=False,
                            gridspec_kw={"width_ratios": widths})
    col = f"pct_positive_{pct(THRESHOLD_COVERAGE)}"
    for ax, m in zip(axs[0], muscles):
        for i, (g, a) in enumerate(orders[m]):
            vals = sections.loc[(sections["muscle"] == m) & (sections["animal"] == a) & (sections["group"] == g), col]
            ax.bar(i, vals.mean(), width=0.7, color=qc.GROUP_COLORS[g], alpha=0.85, linewidth=0)
            ax.scatter(np.full(len(vals), i), vals, s=12, color=qc.INK, alpha=0.6, linewidth=0, zorder=3)
            ax.annotate(f"{vals.mean():.1f}", (i, max(vals.mean(), vals.max())), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=8, color=qc.INK)
        ax.set_xticks(range(len(orders[m])))
        ax.set_xticklabels([a for _, a in orders[m]], color=qc.INK)
        ax.set_xlim(-0.6, len(orders[m]) - 0.4)
        ax.grid(axis="x", visible=False)
        ax.set_title(f"{m}: threshold {thresholds[m][THRESHOLD_COVERAGE]['sector_threshold']:.0f}", loc="left",
                     fontsize=10, color=qc.INK)
    axs[0, 0].set_ylabel("Fibers positive (%)")
    legend(fig)
    note = [f"Positive: at least {pct(THRESHOLD_COVERAGE)}% of the ring above the threshold "
            f"({100 * (1 - fpr):.0f}th percentile of the pooled control fibers).",
            "Control animals: each against the threshold from the other control animals (held out).",
            "Bars: mean of the animal's sections; dots: sections."]
    lines = [w for line in note for w in textwrap.wrap(line, width=int(17 * sum(widths)))]
    fig.text(0.01, 0.01, "\n".join(lines), fontsize=7, color=qc.MUTED)
    fig.tight_layout(rect=(0, 0.03 + 0.035 * len(lines), 1, 0.92))
    qc.save(fig, out, dpi=200)


def plot_score_distributions(scores: dict, meta: pd.DataFrame, thresholds: dict, out: Path) -> None:
    col = f"score_{pct(THRESHOLD_COVERAGE)}"
    muscles = muscles_in(thresholds)
    fig, axs = plt.subplots(1, len(muscles), figsize=(5.2 * len(muscles), 3.8), squeeze=False)
    for ax, m in zip(axs[0], muscles):
        names = {g: list(meta.index[(meta["muscle"] == m) & (meta["group"] == g)]) for g in GROUPS.order}
        groups = {g: pd.concat([scores[s][col] for s in v]) for g, v in names.items() if v}
        lo, hi = np.percentile(pd.concat(groups.values()), [0.5, 99.5])
        for g, v in groups.items():
            ax.hist(v, bins=np.linspace(lo, hi, 80), density=True, histtype="step", lw=2, color=qc.GROUP_COLORS[g],
                    label=GROUPS.labels[g])
        t = thresholds[m][THRESHOLD_COVERAGE]["sector_threshold"]
        ax.axvline(t, color=qc.INK, lw=1, ls=(0, (4, 3)))
        ax.set_title(f"{m}: fiber scores at {pct(THRESHOLD_COVERAGE)}% coverage (dashed = threshold {t:.0f})",
                     loc="left", fontsize=9, color=qc.INK)
        ax.set_xlabel("fiber score (≈ median ring brightness − own cytoplasm, a.u.)")
        ax.set_yticks([])
        ax.grid(axis="y", visible=False)
    axs[0, 0].legend(fontsize=8)
    fig.tight_layout()
    qc.save(fig, out, dpi=200)


def plot_coverage_comparison(animals: pd.DataFrame, thresholds: dict, out: Path) -> None:
    muscles = muscles_in(thresholds)
    fig, axs = plt.subplots(1, len(muscles), figsize=(5.2 * len(muscles), 4.0), sharey=True, squeeze=False)
    x = [100 * c for c in COVERAGES]
    ymax = max(animals[f"pct_positive_{pct(c)}"].max() for c in COVERAGES)
    for ax, m in zip(axs[0], muscles):
        ends = []
        for _, r in animals[animals["muscle"] == m].iterrows():
            y = [r[f"pct_positive_{pct(c)}"] for c in COVERAGES]
            ax.plot(x, y, color=qc.GROUP_COLORS[r["group"]], lw=2, marker="o", ms=4)
            ends.append((y[-1], r["animal"]))
        for ylab, (y_end, a) in zip(spread([e[0] for e in ends], 0.045 * ymax), ends):
            ax.annotate(a, (x[-1], y_end), xytext=(x[-1] + 2, ylab), textcoords="data", va="center", fontsize=8,
                        color=qc.INK, arrowprops={"arrowstyle": "-", "color": qc.AXIS, "lw": 0.6})
        ax.set_title(f"{m}: threshold {thresholds[m][THRESHOLD_COVERAGE]['sector_threshold']:.0f}; "
                     f"{ctrl_held_out(thresholds[m])}", loc="left", fontsize=9, color=qc.INK)
        ax.set_xticks(x)
        ax.set_xlim(x[0] - 4, x[-1] + 8)
        ax.set_xlabel("coverage required (% of the ring)")
    axs[0, 0].set_ylabel("Fibers called positive (%, mean of sections)")
    legend(fig)
    fig.text(0.01, 0.01, "Same threshold at every level; requiring more of the ring is stricter, so every line falls. "
             "Controls fall fastest.", fontsize=7, color=qc.MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    qc.save(fig, out, dpi=200)


def plot_intensity_by_animal(samples: dict, meta: pd.DataFrame, out: Path) -> None:
    muscles = muscles_in(meta["muscle"])
    fig, axs = plt.subplots(1, len(muscles), figsize=(5.2 * len(muscles), 4.6), squeeze=False)
    for ax, m in zip(axs[0], muscles):
        sec = meta[meta["muscle"] == m]
        order = [(g, a) for g in GROUPS.order for a in sorted(sec.loc[sec["group"] == g, "animal"].astype(str).unique())]
        data = []
        for g, a in order:
            names = [s for s in samples if s in sec.index and sec.loc[s, "group"] == g and str(sec.loc[s, "animal"]) == a]
            data.append(np.concatenate([samples[s]["fibers"]["dys_membrane_specific"].dropna().to_numpy() for s in names]))
        lo, hi = np.percentile(np.concatenate(data), [0.5, 99.5])
        parts = ax.violinplot(data, positions=range(len(data)), widths=0.8, showextrema=False, points=300)
        for body, (g, _) in zip(parts["bodies"], order):
            body.set_facecolor(qc.GROUP_COLORS[g])
            body.set_edgecolor(qc.GROUP_COLORS[g])
            body.set_alpha(0.3)
        for i, (v, (g, _)) in enumerate(zip(data, order)):
            q25, med, q75 = np.percentile(v, [25, 50, 75])
            ax.vlines(i, q25, q75, color=qc.GROUP_COLORS[g], lw=6)
            ax.scatter([i], [med], s=26, facecolor=qc.SURFACE, edgecolor=qc.INK, zorder=3)
        ax.axhline(0, color=qc.AXIS, lw=0.8)
        ax.set_ylim(min(lo, 0) - 50, hi)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([f"{a}\nn={len(v):,}" for (_, a), v in zip(order, data)], fontsize=8, color=qc.INK)
        ax.grid(axis="x", visible=False)
        ax.set_title(m, loc="left", fontsize=10, color=qc.INK)
    axs[0, 0].set_ylabel("Membrane dystrophin, specific (a.u.)")
    legend(fig)
    fig.text(0.01, 0.01, "All included fibers of each animal (sections pooled). Box = IQR, dot = median.",
             fontsize=7, color=qc.MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    qc.save(fig, out, dpi=200)


# ---------------------------------------------------------------- driver


def summarize(results: Path, meta: pd.DataFrame, out: Path, figs: Path, fpr: float = FPR,
              borrowed: dict[str, dict] | None = None) -> dict:
    """Thresholds, calls, tables and figures for the sections in meta (indexed by sample) that have results.

    Each muscle's threshold comes from its control sections. A muscle without controls needs an entry in borrowed,
    {muscle: {"threshold": float, "source": str}}, the threshold of an earlier run. Returns thresholds, the section and
    animal tables and the samples that had no results."""
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    borrowed = borrowed or {}
    samples = {s: d for s, d in cal.load_results(results).items() if s in meta.index}
    missing = sorted(set(meta.index) - set(samples))
    if missing:
        print(f"WARNING: no results yet for {missing}")
    meta = meta.loc[list(samples)]
    animal_of = meta["animal"].astype(str).to_dict()
    scores = fiber_scores(samples)

    thresholds, sources, loao, calls, section_thresholds = {}, {}, [], {}, {}
    for m in sorted(meta["muscle"].unique()):
        ctrl = list(meta.index[(meta["muscle"] == m) & (meta["group"] == GROUPS.control)])
        n_animals = len({animal_of[s] for s in ctrl})
        if ctrl:
            thresholds[m] = calibrate_muscle(scores, ctrl, COVERAGES, fpr)
            sources[m] = (f"controls in this run ({n_animals} animal{'s' * (n_animals != 1)}, "
                          f"{len(ctrl)} section{'s' * (len(ctrl) != 1)})")
        elif m in borrowed:
            thresholds[m] = fixed_threshold(scores, ctrl, borrowed[m]["threshold"])
            sources[m] = f"borrowed from {borrowed[m]['source']}"
        else:
            raise ValueError(f"{m}: no control sections and no threshold to borrow")
        held = held_out_thresholds(scores, ctrl, animal_of, COVERAGES, fpr) if n_animals >= 2 else {}
        if n_animals >= 2:
            lo = leave_one_animal_out(scores, ctrl, animal_of, COVERAGES, fpr, held)
            lo.insert(0, "muscle", m)
            loao.append(lo)
            for c in COVERAGES:
                sub = lo[lo["coverage"] == c]["fpr"]
                thresholds[m][c].update({"loao_fpr_mean": float(sub.mean()), "loao_fpr_max": float(sub.max())})
        for s in meta.index[meta["muscle"] == m]:
            if s in ctrl:
                th_s, origin = (held[animal_of[s]], "held_out") if held else (thresholds[m], "alone")
            else:
                th_s, origin = thresholds[m], "all" if ctrl else "borrowed"
            calls[s] = call_fibers(samples[s], scores[s], th_s, with_threshold=True)
            section_thresholds[s] = (th_s[THRESHOLD_COVERAGE]["sector_threshold"], THRESHOLD_FROM[origin])

    sections = section_table(calls, meta, thresholds=section_thresholds)
    animals = animal_table(sections)
    sections.to_csv(out / "sections.csv", index=False)
    animals.to_csv(out / "animals.csv", index=False)
    loao_cols = ["muscle", "held_out_animal", "coverage", "sector_threshold", "n_fibers", "fpr"]
    (pd.concat(loao) if loao else pd.DataFrame(columns=loao_cols)).to_csv(out / "leave_one_animal_out.csv", index=False)
    pd.concat([c.assign(sample=s) for s, c in calls.items()]).to_csv(out / "calls.csv.gz", index=False)
    info = {"rule": "one threshold per muscle = (1 - fpr) percentile of pooled control fibers' 18th brightest ring "
                    "piece; positive at level c when at least c of the 36 pieces reach the threshold",
            "fpr": fpr, "threshold_coverage": THRESHOLD_COVERAGE, "levels": [pct(c) for c in COVERAGES],
            "muscles": {m: {"threshold": th[THRESHOLD_COVERAGE]["sector_threshold"], "threshold_source": sources[m],
                            **{pct(c): v for c, v in th.items()}}
                        for m, th in thresholds.items()}}
    (out / "thresholds.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    plot_pct_positive(sections, thresholds, figs / "pct_positive_by_animal.png", fpr)
    plot_call(sections, thresholds, figs / "pct_positive_call.png", fpr)
    plot_score_distributions(scores, meta, thresholds, figs / "score_distributions.png")
    plot_coverage_comparison(animals, thresholds, figs / "coverage_comparison.png")
    plot_intensity_by_animal(samples, meta, figs / "intensity_by_animal.png")
    for m, th in thresholds.items():
        print(m, "threshold", round(th[THRESHOLD_COVERAGE]["sector_threshold"]), f"({sources[m]}) |", ctrl_held_out(th))
    cols = ["muscle", "group", "animal", "n_sections", "n_called"] + [f"pct_positive_{pct(c)}" for c in COVERAGES]
    print(animals[cols].round(1).to_string(index=False))
    print(f"wrote {out}")
    return {"thresholds": thresholds, "sources": sources, "sections": sections, "animals": animals,
            "missing": missing}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--tables", type=Path, default=paths.project().tables, help="tables, including calls.csv.gz")
    ap.add_argument("--figures", type=Path, default=paths.project().figures_results)
    ap.add_argument("--fpr", type=float, default=FPR, help="share of pooled control fibers allowed above the threshold")
    args = ap.parse_args()
    meta = pd.read_csv(args.manifest, dtype={"animal": str}).set_index("sample")
    summarize(args.results, meta, args.tables, args.figures, args.fpr)


if __name__ == "__main__":
    main()
