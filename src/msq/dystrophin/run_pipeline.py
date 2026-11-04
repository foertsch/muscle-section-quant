"""Runs the whole pipeline on a folder of ND2 scans, into one run folder: sample table, checks, segmentation and
measurement, calls, figures and reports.

Usage:
    uv run msq-dystrophin SCANS RUN                      # everything (or python -m msq.dystrophin.run_pipeline)
    uv run msq-dystrophin SCANS RUN --check-only         # stop after the sample table and the checks
    uv run msq-dystrophin SCANS RUN --reference OLD_RUN  # threshold for muscles that have no controls in this run

The first time, the sample table (RUN/samples.csv) is built from the folder and file names. After that the
existing table is used as it is, so it can be corrected by hand, and an interrupted run resumes where it
stopped: finished sections are not measured again. Delete samples.csv to rebuild it from the folder.

The threshold of each muscle comes from the untreated controls in the run (98th percentile of their 50% scores, see
settings.py). A muscle without controls can borrow the threshold of an earlier run (--reference: a run folder, a project
folder or a thresholds.json), which is only valid if staining and imaging were identical.

The run folder:
    samples.csv            one row per section: sample, group (ctrl / treated), muscle, animal, section, path
    checks.txt             what the checks found
    run_settings.json      rule, scope, code version, GPU, times; at the end the thresholds, their source, results
    progress.json          live status, rewritten after every step
    run.log                everything printed
    tables/ figures/ reports/   results
    sections/<sample>/     per-section intermediates: fibers.csv, sectors.csv.gz, labels, map

Exit status: 0 done, 1 stopped by errors in the checks, 2 done but some sections, reports or the fiber size figure
failed, or a measured section did not reach the tables, 3 stopped on request (the file RUN/stop_requested, which the
app creates) before all sections were measured, 4 stopped by an unexpected error (the traceback is in run.log), 5 not
started because another runner holds the run folder's lock (RUN/runner.lock).
"""

from __future__ import annotations

import json
import traceback
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from msq.core import runinfo, runner  # noqa: F401  (rp.runinfo, which the tests use)
from msq.core.io import read_json, write_json
from msq.core.problems import ExitStatus
from msq.core.runner import Progress, Summary, Tee, gpu_device, now, weights_file  # noqa: F401  (pinned names)
from msq.core.samples import problems_text
from msq.dystrophin import checks
from msq.dystrophin import compare_report as cr
from msq.dystrophin import decision_report as dr
from msq.dystrophin import measure as dq
from msq.dystrophin import full_summary as fs
from msq.dystrophin import paths
from msq.dystrophin import plot_fiber_size as pfs
from msq.dystrophin import qc
from msq.dystrophin import settings


# ---------------------------------------------------------------- steps


def load_reference(path: Path, weights: dict | None = None) -> dict[str, dict]:
    """Thresholds of an earlier run, {muscle: {"threshold", "source", "checked"}}, from its thresholds.json. path: the
    file itself, the folder holding it, a run folder or a project folder. weights: this run's Cellpose weights
    (msq.core.runinfo.weights), None when not known yet.

    The earlier run must use the same rule and, where its run_settings.json says, the same measurement settings and
    Cellpose weights; "checked" is False when either could not be compared (a project folder, a bare thresholds.json,
    a run from before weights were recorded). A threshold the earlier run borrowed itself keeps its origin in
    "source". A run made before "bar" and "cutoff" became "threshold" is read as well (settings.current_keys)."""
    for f in (path, path / "thresholds.json", path / "tables" / "thresholds.json",
              path / "deliverables" / "tables" / "thresholds.json"):
        if f.is_file():
            break
    else:
        raise FileNotFoundError(f"no thresholds.json in {path}")
    th = settings.current_keys(json.loads(f.read_text(encoding="utf-8")))
    if th["fpr"] != settings.FPR or th["threshold_coverage"] != settings.CALL_LEVEL:
        raise ValueError(f"{f} uses another rule (fpr {th['fpr']}, level {th['threshold_coverage']}); this run uses "
                         f"fpr {settings.FPR}, level {settings.CALL_LEVEL}")
    run_settings = f.parent.parent / "run_settings.json"  # f is <run>/tables/thresholds.json
    recorded = json.loads(run_settings.read_text(encoding="utf-8")) if run_settings.is_file() else {}
    theirs, ours = settings.current_keys(recorded.get("measurement_params")), asdict(dq.Params())
    if theirs is not None and theirs != ours:
        other = sorted(k for k in set(ours) | set(theirs) if theirs.get(k) != ours.get(k))
        raise ValueError(f"{path} was measured with other settings ({', '.join(other)}), so its threshold does not "
                         "apply")
    their_w, our_w = (recorded.get("cellpose_weights") or {}).get("sha256"), (weights or {}).get("sha256")
    if their_w and our_w and their_w != our_w:
        raise ValueError(f"{path} was measured with other Cellpose weights (SHA-256 {their_w[:12]}…, this computer "
                         f"has {our_w[:12]}…), so its threshold does not apply")
    thresholds = {}
    for m, v in th["muscles"].items():
        source = str(path.resolve())
        if str(v.get("threshold_source", "")).startswith("borrowed from "):
            source += f" ({v['threshold_source']})"
        thresholds[m] = {"threshold": float(v["threshold"]), "source": source,
                         "checked": theirs is not None and bool(their_w and our_w)}
    return thresholds


def prepare(scans: Path, rf: paths.RunFolder, suffix: str = ".nd2") -> tuple[pd.DataFrame, list[checks.Problem]]:
    """The sample table: the existing samples.csv if there is one, else built from the folder."""
    if not scans.is_dir() and not rf.samples.exists():
        return pd.DataFrame(columns=checks.COLUMNS), [checks.Problem("error", "folder not found", str(scans))]
    built, problems = (checks.samples_from_folder(scans, suffix=suffix) if scans.is_dir() else (None, []))
    if rf.samples.exists():
        print(f"using the existing sample table {rf.samples} (delete it to rebuild it from the folder)")
        table = checks.read_samples(rf.samples)
        if built is not None:  # scans added to the folder since are not used; say so
            known = {str(Path(p).resolve()) for p in table["path"]}
            new = [Path(p).name for p in built["path"] if p not in known]
            if new:
                problems.append(checks.Problem(
                    "note", f"{len(new)} scan{'s are' if len(new) > 1 else ' is'} in the folder but not in the sample "
                            f"table (left out by hand, or added since): {', '.join(new[:5])}"
                            f"{', …' if len(new) > 5 else ''}. Rebuild the table from the folder to add them."))
        return table, problems
    built.to_csv(rf.samples, index=False)
    print(f"wrote the sample table {rf.samples}")
    return checks.read_samples(rf.samples), problems


def table_overview(samples: pd.DataFrame) -> str:
    rows = []
    for (m, g), d in samples.groupby(["muscle", "group"]):
        animals = ", ".join(f"{a} ({n})" for a, n in d.groupby("animal").size().items())
        rows.append(f"  {m:<5} {settings.GROUP_NAMES.get(g, g):<8} {len(d):>3} sections; animal (sections): {animals}")
    return "\n".join(rows)


def make_reports(samples: pd.DataFrame, rf: paths.RunFolder, reader=None) -> tuple[list[Path], list[checks.Problem]]:
    """The threshold decision report (muscles with enough control animals and treated sections) and one
    controls-vs-treated report per muscle that has both. A failing report is a warning, not a failed run."""
    nd2_paths = dict(zip(samples["sample"], samples["path"].map(Path)))
    by_muscle = {m: d for m, d in samples.groupby("muscle")}
    both = [m for m in fs.muscles_in(by_muscle) if set(by_muscle[m]["group"]) >= {"ctrl", "treated"}]
    decision = [m for m in both if by_muscle[m].loc[by_muscle[m]["group"] == "ctrl", "animal"].nunique()
                >= settings.MIN_CONTROL_ANIMALS]
    extra = {"read": reader.read} if reader is not None else {}  # the reports read the scans too
    jobs = []
    if decision:
        jobs.append((rf.reports / "threshold_decision.html",
                     lambda out: dr.build(rf.sections, rf.tables, nd2_paths, decision, out, **extra)))
    for m in both:
        jobs.append((rf.reports / f"{m.lower()}_ctrl_vs_treated.html",
                     lambda out, m=m: cr.build(rf.sections, rf.tables, nd2_paths, m, out, **extra)))
    written, problems = [], []
    for out, job in jobs:
        try:
            job(out)
            written.append(out)
        except Exception as e:
            traceback.print_exc()
            problems.append(checks.Problem("warning", f"report not written ({type(e).__name__}: {e})", out.name,
                                           code=checks.REPORT_FAILED))
    for m in sorted(set(both) - set(decision)):
        problems.append(checks.Problem("note", "no threshold decision report: it needs at least "
                                               f"{settings.MIN_CONTROL_ANIMALS} control animals", m))
    return written, problems


def results_text(summary: dict) -> str:
    animals, th = summary["animals"], summary["thresholds"]
    lines = [f"% of fibers positive (at least {round(100 * settings.CALL_LEVEL)}% of the ring above the threshold; "
             "mean of each animal's sections; control animals against the threshold from the other control animals)"]
    for m in fs.muscles_in(animals["muscle"]):
        lines.append(f"{m}: threshold {th[m][fs.THRESHOLD_COVERAGE]['sector_threshold']:.0f} ({summary['sources'][m]})")
        for r in animals[animals["muscle"] == m].sort_values(["group", "animal"]).itertuples():
            pct = getattr(r, settings.CALL_COLUMN)
            lines.append(f"  {r.animal:>8}  {settings.GROUP_NAMES[r.group]:<8} {pct:5.1f}%   "
                         f"({r.n_sections} sections, {r.n_called:,} fibers)")
    return "\n".join(lines)


# ---------------------------------------------------------------- the assay


class Dystrophin:
    """dystrophin as core.runner drives it. One instance serves one run: it keeps the thresholds borrowed from the
    reference run from the checks to the calls. Options: reference (a run folder, a project folder or a
    thresholds.json), reader (where the scans are read from, scans.py; default Nikon ND2 files)."""

    name = "dystrophin"

    def __init__(self):
        self.borrowed: dict[str, dict] = {}

    def arguments(self, parser):
        parser.add_argument("--reference", type=Path,
                            help="earlier run whose threshold is used for muscles without controls in this run")

    def device(self):
        return gpu_device()

    def weights_file(self):
        return weights_file()

    def prepare(self, ctx):
        reader = ctx.options.get("reader")
        return prepare(ctx.scans, ctx.run, reader.suffix if reader is not None else ".nd2")

    def check(self, samples, ctx):
        reference, reader, problems = ctx.options.get("reference"), ctx.options.get("reader"), []
        if reference:
            try:
                self.borrowed = load_reference(reference, ctx.weights)
            except (OSError, ValueError, KeyError) as e:
                problems.append(checks.Problem("error", f"cannot use the earlier run: {e}", str(reference)))
        if not samples.empty:
            problems += checks.check_files(samples,
                                           read_header=reader.header if reader is not None else checks.nd2_header)
        problems += checks.check_design(samples, self.borrowed) + checks.check_weights(ctx.weights)
        if ctx.device == "cpu":
            problems.append(checks.Problem("warning", "no GPU found: segmentation runs on the processor and takes "
                                                      "much longer"))
        return problems

    def overview(self, samples):
        return table_overview(samples)

    def write_checks(self, samples, problems, ctx):
        """checks.txt for people, checks.json for the app. Returns the text."""
        text = problems_text(problems, "No problems found.")
        ctx.run.checks.write_text(text + "\n", encoding="utf-8")
        write_json(ctx.run.checks_json, {"problems": [asdict(p) for p in problems]})
        return text

    def settings(self, ctx):
        reference = ctx.options.get("reference")
        return {"rule": {"threshold_percentile_of_controls": round(100 * (1 - settings.FPR), 1),
                         "call_pct_of_ring": round(100 * settings.CALL_LEVEL), "levels_in_tables": [50, 60, 70]},
                "scope": {"muscles": list(settings.MUSCLES), "dystrophin_channel": settings.DYSTROPHIN_CHANNEL,
                          "actin_channel": settings.ACTIN_CHANNEL, "magnification": settings.MAGNIFICATION},
                "reference": str(reference.resolve()) if reference else None,
                "measurement_params": asdict(dq.Params())}

    def is_done(self, row, ctx):
        return dq.is_done(ctx.run.sections / row.sample, Path(row.path))

    def measure(self, row, ctx):
        out, scan, reader = ctx.run.sections / row.sample, Path(row.path), ctx.options.get("reader")
        other_scan = not dq.made_from(out, scan)  # also its cached segmentation belongs to that other scan
        if other_scan:
            print(f"[{row.sample}] the results there come from another scan; measuring {scan.name} again", flush=True)
        extra = {"read": reader.read} if reader is not None else {}
        dq.run(scan, ctx.run.sections, resegment=ctx.redo or other_scan, name=row.sample,
               provenance=ctx.provenance, **extra)

    def summarise(self, done, failed, problems, ctx):
        rf, reader, provenance = ctx.run, ctx.options.get("reader"), ctx.provenance
        if ctx.weights is None:  # the first segmentation downloads the weights if this computer has none yet
            problems += checks.check_weights(provenance.weights)
        problems += checks.check_provenance({s: read_json(rf.sections / s / "run_info.json") for s in done["sample"]},
                                            provenance.code_version, provenance.weights)
        late = checks.errors(checks.check_design(done, self.borrowed)) if not done.empty else [
            checks.Problem("error", "no section could be measured")]
        if late:
            for p in late:
                print(p)
            return Summary(ExitStatus.INCOMPLETE, {}, state="failed",
                           message="Too many sections failed to set the threshold: " + "; ".join(map(str, late)))

        ctx.progress.set("summarizing", "Calling fibers and drawing figures")
        try:
            qc.dystrophin_maps([rf.sections / s for s in done["sample"]])
        except Exception as e:
            traceback.print_exc()
            problems.append(checks.Problem("warning", f"section maps not drawn ({type(e).__name__}: {e})"))
        meta = done.set_index("sample")
        summary = fs.summarize(rf.sections, meta, rf.tables, rf.figures, settings.FPR, self.borrowed)
        problems += [checks.Problem("warning", "measured, but its results were not found, so it is not in the "
                                               "tables", s) for s in summary["missing"]]
        sizes_ok = True
        try:
            pfs.make(rf.sections, meta, rf.figures, rf.tables)
        except Exception as e:  # a figure should not end a run whose sections are all measured
            traceback.print_exc()
            sizes_ok = False
            problems.append(checks.Problem("warning", f"fiber size figure and table not written "
                                                      f"({type(e).__name__}: {e})"))

        ctx.progress.set("reports", "Writing the reports")
        reports, report_problems = make_reports(done, rf, reader)
        problems += report_problems
        problems += [checks.Problem("warning", "failed and left out: " + ctx.progress.data["sections"][s]["error"], s)
                     for s in failed]

        text = "\n" + results_text(summary)
        if failed:
            text += f"\n\n{len(failed)} section(s) failed and are left out: {', '.join(failed)}"
        bad = bool(failed or summary["missing"]) or not sizes_ok or any(p.code == checks.REPORT_FAILED
                                                                         for p in report_problems)
        return Summary(
            ExitStatus.INCOMPLETE if bad else ExitStatus.DONE,
            {"thresholds": {m: {"threshold": th[fs.THRESHOLD_COVERAGE]["sector_threshold"],
                                "source": summary["sources"][m]} for m, th in summary["thresholds"].items()},
             "results": summary["animals"][["muscle", "group", "animal", "n_sections", "n_called",
                                            settings.CALL_COLUMN]].round(2).to_dict(orient="records"),
             "reports": [str(p.relative_to(rf.root)) for p in reports],
             "problems": [str(p) for p in problems]},
            problems, text + f"\n\nResults in {rf.root}: tables/, figures/, reports/")


def run(scans: Path, root: Path, reference: Path | None = None, check_only: bool = False, redo: bool = False,
        reader=None) -> int:
    """reader: where the scans are read from (scans.py); default Nikon ND2 files."""
    return runner.run(Dystrophin(), scans, root, check_only=check_only, redo=redo, reference=reference, reader=reader)


def main() -> None:
    runner.main(Dystrophin())


if __name__ == "__main__":
    main()
