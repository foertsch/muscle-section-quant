"""One runner for every assay: a run folder with its sample table and checks, the sections measured one after the
other (a stopped run resumes where it stopped), the summary, and a record of what made the run. Each assay plugs in
as an Assay (the protocol below) and keeps a three-line main; dystrophin, psr and necrosis are in
<assay>/run_pipeline.py.

The run folder is a core.io.RunFolder: samples.csv, checks.txt (and checks.json if the assay writes one),
run_settings.json, progress.json, run.log, runner.lock, stop_requested (written by the app to ask for a stop), and
sections/, tables/, figures/, reports/ as the assay uses them.

run() goes through these stages, in this order:
 1. lock: when another runner holds runner.lock, print that to stderr and return LOCKED without writing anything;
    open run.log in append mode and tee stdout and stderr into it;
 2. samples, problems = assay.prepare(ctx);
 3. Progress(rf.progress, samples), state "checking"; ctx.weights = runinfo.weights(assay.weights_file());
    ctx.device = assay.device();
 4. problems += assay.check(samples, ctx) (the assay decides what it checks and in which order, the device warning
    included, so checks.txt and checks.json keep their order);
 5. print assay.overview(samples) and the device; print assay.write_checks(samples, problems, ctx);
 6. errors: progress "stopped", return CHECKS_FAILED; check_only: progress "checked", return DONE;
 7. ctx.provenance = Provenance(assay.weights_file, ctx.weights); run_settings.json = started, scans, run_folder,
    code_version, software, device, then assay.settings(ctx);
 8. remove the stop file; for each row in table order: assay.is_done(row, ctx) and not redo: section "done", skipped;
    stop file present: return to stage 9 as stopped; else section "running", progress "measuring",
    assay.measure(row, ctx): section "done" with its seconds, or on an exception the traceback is printed, the
    section is "failed" with "<Type>: <message>" and the run goes on; remove the stop file again;
 9. run_settings.json gets cellpose_weights = ctx.provenance.weights; stopped: progress "paused", run_settings.json
    gets paused and failed_sections, return STOPPED;
10. summary = assay.summarise(done, failed, problems, ctx), where done is the rows assay.is_done accepts;
11. run_settings.json gets finished, failed_sections, summary.settings; checks written again with summary.problems
    (unless it is None); summary.text printed; progress summary.state with summary.message; return summary.status.
An exception after the lock is taken: traceback into run.log, progress "failed" ("The run stopped: <Type>: <msg>"),
CRASHED. main() maps an exception before run.log is open to CRASHED, and exits with the status.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Protocol

import pandas as pd

from msq.core import runinfo, segment
from msq.core.io import Heartbeat, RunFolder, hold_lock, write_json
from msq.core.problems import ExitStatus, Problem, errors


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class Tee:
    """Writes to several streams: the console and run.log."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s: str) -> int:
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self) -> None:
        for st in self.streams:
            st.flush()


def weights_file() -> Path:
    """This computer's Cellpose weights file, where the first segmentation downloads it if it is not there yet."""
    return segment.weights_file(segment.DEFAULT_MODEL)


def gpu_device() -> str:
    import torch

    if torch.cuda.is_available():
        return f"cuda: {torch.cuda.get_device_name(0)}"
    if torch.backends.mps.is_available():
        return "mps: Apple GPU"
    return "cpu"


class Progress:
    """progress.json: overall state, a message, and the status of every section, with an estimate of the time left
    from the sections measured in this session. A background thread refreshes "heartbeat" every HEARTBEAT_S
    seconds while the run is alive, so the app can tell a long section from a run whose process died."""

    HEARTBEAT_S = 60.0

    def __init__(self, path: Path, samples: list[str]):
        self.path = path
        self.durations: list[float] = []
        self.data = {"state": "starting", "message": "", "started": now(), "updated": now(), "heartbeat": now(),
                     "eta_s": None, "n_sections": len(samples), "n_done": 0, "n_failed": 0,
                     "sections": {s: {"status": "waiting", "seconds": None, "error": ""} for s in samples}}
        self._lock = threading.Lock()
        self._heartbeat = Heartbeat(self._beat, self.HEARTBEAT_S)

    def set(self, state: str, message: str = "") -> None:
        with self._lock:
            self.data.update(state=state, message=message)
            self._write()

    def section(self, sample: str, status: str, seconds: float | None = None, error: str = "") -> None:
        with self._lock:
            self.data["sections"][sample] = {"status": status, "seconds": seconds, "error": error}
            if status == "done" and seconds is not None:
                self.durations.append(seconds)
            secs = self.data["sections"].values()
            self.data["n_done"] = sum(v["status"] == "done" for v in secs)
            self.data["n_failed"] = sum(v["status"] == "failed" for v in secs)
            left = sum(v["status"] in ("waiting", "running") for v in secs)
            self.data["eta_s"] = round(left * sum(self.durations) / len(self.durations)) if self.durations else None
            self._write()

    def close(self) -> None:
        self._heartbeat.close()

    def _beat(self) -> None:
        """One beat; a failed write is reported once by the Heartbeat, and the next beat tries again."""
        with self._lock:
            self.data["heartbeat"] = now()
            write_json(self.path, self.data)

    def _write(self) -> None:
        self.data["updated"] = self.data["heartbeat"] = now()
        write_json(self.path, self.data)


class Provenance:
    """The code and the Cellpose weights a section is made with, as its run_info.json records them, so a run that
    resumes can tell sections measured by other code or segmented with other weights. The code version
    is read once; the weights' size and SHA-256 once the file exists (the first segmentation downloads it).
    weights_file: where the weights are (looked up at call time); weights: their record if already known."""

    def __init__(self, weights_file: Callable[[], Path], weights: dict | None = None):
        self.code_version = runinfo.code_version()
        self._weights_file = weights_file
        self._weights = weights

    @property
    def weights(self) -> dict | None:
        if self._weights is None:
            self._weights = runinfo.weights(self._weights_file())
        return self._weights


@dataclass
class RunContext:
    """What every stage of one run can see. options holds the assay's own command-line options (dystrophin:
    reference, reader; psr: cohort, variants; necrosis: variants), by the names the assay gives them."""

    run: RunFolder
    scans: Path
    check_only: bool = False
    redo: bool = False
    options: dict = field(default_factory=dict)
    device: str = ""
    weights: dict | None = None  # runinfo.weights() before the run; None until the weights are downloaded
    progress: Progress | None = None
    provenance: Provenance | None = None
    settings: dict = field(default_factory=dict)  # run_settings.json as written so far


@dataclass
class Summary:
    """What assay.summarise returns: the run's end status, what it adds to run_settings.json (thresholds, results,
    reports), the problems of the whole run (checks.txt is written again with them; None: it is left as it is), the
    text printed at the end, and the last progress state and message ("failed" with a reason when too few sections
    were measured). An empty message becomes "Finished", with the number of failed sections when there are some."""

    status: ExitStatus
    settings: dict
    problems: list[Problem] | None = None
    text: str = ""
    state: str = "done"
    message: str = ""


class Assay(Protocol):
    """One assay as the runner drives it. A row is one line of samples.csv (pandas itertuples). One instance serves
    one run: dystrophin keeps the thresholds it borrows between check and summarise."""

    name: str  # "dystrophin", "psr", "necrosis"

    def arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add the assay's own options to the runner's (SCANS RUN --check-only --redo)."""

    def device(self) -> str:
        """"cuda", "mps" or "cpu"."""

    def weights_file(self) -> Path:
        """The Cellpose weights file the sections are segmented with."""

    def prepare(self, ctx: RunContext) -> tuple[pd.DataFrame, list[Problem]]:
        """The sample table and its problems: dystrophin keeps a samples.csv that exists (it can be corrected by
        hand) and else builds it from ctx.scans; psr and necrosis build it from ctx.scans every time."""

    def check(self, samples: pd.DataFrame, ctx: RunContext) -> list[Problem]:
        """Every check before the run, in the order the assay reports them."""

    def overview(self, samples: pd.DataFrame) -> str:
        """The sample table summed up for the console."""

    def write_checks(self, samples: pd.DataFrame, problems: list[Problem], ctx: RunContext) -> str:
        """checks.txt (and checks.json, samples.csv, tasks.tsv where the assay writes them); returns the text."""

    def settings(self, ctx: RunContext) -> dict:
        """The assay's part of run_settings.json (rule, scope, measurement parameters, reference ...)."""

    def is_done(self, row, ctx: RunContext) -> bool:
        """Whether the section's results are complete and made from this row's scan."""

    def measure(self, row, ctx: RunContext) -> None:
        """Segment and measure one section into ctx.run.sections; raise on failure."""

    def summarise(self, done: pd.DataFrame, failed: list[str], problems: list[Problem], ctx: RunContext) -> Summary:
        """Calls, tables, figures and reports from the finished sections."""


def run(assay: Assay, scans: Path, root: Path, *, check_only: bool = False, redo: bool = False,
        **options) -> ExitStatus:
    """One run of the assay on the scans into the run folder root, through the stages in the module docstring."""
    ctx = RunContext(RunFolder(root), scans, check_only, redo, options)
    rf = ctx.run
    rf.root.mkdir(parents=True, exist_ok=True)
    with hold_lock(rf.lock) as free:
        if not free:  # leave progress.json and run.log to the runner that is working
            print(f"Another runner is working on {rf.root}; this one does not start.", file=sys.stderr, flush=True)
            return ExitStatus.LOCKED
        with rf.log.open("a", encoding="utf-8") as log, \
                contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            print(f"\n===== {now()}: {scans} -> {root}", flush=True)
            try:
                return _run(assay, ctx)
            except Exception as e:
                traceback.print_exc()
                if ctx.progress:
                    ctx.progress.set("failed", f"The run stopped: {type(e).__name__}: {e}")
                return ExitStatus.CRASHED
            finally:
                if ctx.progress:
                    ctx.progress.close()


def _run(assay: Assay, ctx: RunContext) -> ExitStatus:
    rf = ctx.run
    samples, problems = assay.prepare(ctx)
    progress = ctx.progress = Progress(rf.progress, list(samples["sample"]))
    progress.set("checking", "Checking the scans")
    ctx.weights = runinfo.weights(assay.weights_file())  # None until the first segmentation downloads them
    ctx.device = assay.device()
    problems += assay.check(samples, ctx)

    print(f"Sample table ({len(samples)} sections):\n{assay.overview(samples)}")
    print(f"GPU: {ctx.device}")
    print(assay.write_checks(samples, problems, ctx), flush=True)
    if errors(problems):
        progress.set("stopped", f"{len(errors(problems))} problem(s) must be fixed first, see checks.txt")
        return ExitStatus.CHECKS_FAILED
    if ctx.check_only:
        progress.set("checked", "Checks passed; ready to run")
        return ExitStatus.DONE

    provenance = ctx.provenance = Provenance(assay.weights_file, ctx.weights)  # what each section records
    info = ctx.settings = {"started": now(), "scans": str(ctx.scans.resolve()), "run_folder": str(rf.root.resolve()),
                           "code_version": provenance.code_version, "software": runinfo.packages(),
                           "device": ctx.device, **assay.settings(ctx)}
    write_json(rf.settings, info)

    rf.stop.unlink(missing_ok=True)
    failed, stopped = _measure(assay, samples, ctx)
    rf.stop.unlink(missing_ok=True)  # also when it was asked for during the last section: nothing is left to stop
    info["cellpose_weights"] = provenance.weights  # the first segmentation downloads the weights if there are none
    if stopped:
        progress.set("paused", f"Stopped on request after {progress.data['n_done']} of {len(samples)} sections. "
                               "Start again to continue where it stopped.")
        info.update(paused=now(), failed_sections=failed)
        write_json(rf.settings, info)
        return ExitStatus.STOPPED

    done = samples[[assay.is_done(r, ctx) for r in samples.itertuples()]]
    summary = assay.summarise(done, failed, problems, ctx)
    info.update(finished=now(), failed_sections=failed, **summary.settings)
    write_json(rf.settings, info)
    if summary.problems is not None:
        assay.write_checks(samples, summary.problems, ctx)
    if summary.text:
        print(summary.text, flush=True)
    progress.set(summary.state, summary.message or (f"Finished; {len(failed)} section(s) failed" if failed
                                                      else "Finished"))
    return summary.status


def _measure(assay: Assay, samples: pd.DataFrame, ctx: RunContext) -> tuple[list[str], bool]:
    """Measure every section that is not finished yet. Returns the samples that failed, and whether the run
    stopped early because the app asked it to (rf.stop exists)."""
    failed = []
    for i, r in enumerate(samples.itertuples(), start=1):
        if not ctx.redo and assay.is_done(r, ctx):
            print(f"=== {i}/{len(samples)} {r.sample}: finished earlier, skipped", flush=True)
            ctx.progress.section(r.sample, "done")
            continue
        if ctx.run.stop.exists():
            print(f"=== stop requested; {i - 1} of {len(samples)} sections handled", flush=True)
            return failed, True
        print(f"=== {i}/{len(samples)} {r.sample}", flush=True)
        ctx.progress.section(r.sample, "running")
        ctx.progress.set("measuring", f"Measuring {r.sample} ({i} of {len(samples)})")
        t0 = time.time()
        try:
            assay.measure(r, ctx)
        except Exception as e:  # keep going; one bad section should not stop a long run
            traceback.print_exc()
            ctx.progress.section(r.sample, "failed", error=f"{type(e).__name__}: {e}")
            failed.append(r.sample)
        else:
            ctx.progress.section(r.sample, "done", seconds=round(time.time() - t0, 1))
    return failed, False


def main(assay: Assay, argv: list[str] | None = None) -> None:
    """The command line: SCANS RUN [--check-only] [--redo] plus assay.arguments; exits with run()'s status. The
    help text is the docstring of the module the assay is defined in."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=sys.modules[type(assay).__module__].__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scans", type=Path, help="folder with the scans (subfolders are searched too)")
    ap.add_argument("run", type=Path, help="run folder for the results (created if needed)")
    ap.add_argument("--check-only", action="store_true", help="stop after the sample table and the checks")
    ap.add_argument("--redo", action="store_true", help="segment and measure finished sections again")
    assay.arguments(ap)
    args = vars(ap.parse_args(argv))
    scans, root, check_only, redo = (args.pop(k) for k in ("scans", "run", "check_only", "redo"))
    try:
        code = run(assay, scans, root, check_only=check_only, redo=redo, **args)
    except Exception:  # before run.log is open, e.g. the run folder cannot be created
        traceback.print_exc()
        code = ExitStatus.CRASHED
    sys.exit(code)
