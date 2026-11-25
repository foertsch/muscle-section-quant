"""Runs the PSR (fibrosis) assay on a folder of scans into one run folder: the sample table and the checks, every
section segmented and measured, and the summary tables.

Usage:
    python -m msq.psr.run_pipeline SCANS RUN --cohort NAME [--variants base ds4]
    python -m msq.psr.run_pipeline SCANS RUN --cohort NAME --check-only     # only samples.csv and checks.txt

--cohort names the dataset's cohort in cohorts.local.toml (see cohorts.example.toml). The sample table (one row per
section with its imaging settings) is built from the scans every time. A section is written to
RUN/sections/<variant>/<sample>/ and is finished when its section.json is there, so a run that was interrupted is
started again with the same command and goes on where it stopped; --redo measures all again. The tables in RUN/tables/
are made from the first variant.

Exit status (msq.core.problems.ExitStatus): 0 done, 1 stopped by errors in the checks, 2 done but some sections
failed, 3 stopped on request (the file RUN/stop_requested), 4 stopped by an unexpected error (the traceback is in
RUN/run.log), 5 not started because another runner holds the run folder (RUN/runner.lock).
"""

from __future__ import annotations

from pathlib import Path

from msq.core import runner
from msq.core import samples as sample_tables
from msq.core.problems import ExitStatus
from msq.core.runner import Summary
from msq.psr import checks, section, settings, summary


class Psr:
    """psr as core.runner drives it. Options: cohort (its name), variants (default base), segment (a function from a
    sample name to what section.run takes as segment, e.g. cached labels; default Cellpose; the same for every
    variant)."""

    name = "psr"

    def arguments(self, parser):
        parser.add_argument("--cohort", required=True, help="the dataset's cohort in cohorts.local.toml")
        parser.add_argument("--variants", nargs="+", default=["base"], choices=list(section.VARIANTS),
                            help="segmentation variants to run; the tables use the first")

    def cohort(self, ctx):
        return settings.load_cohort(ctx.options["cohort"])

    def variants(self, ctx):
        return ctx.options.get("variants") or ["base"]

    def device(self):
        return runner.gpu_device()

    def weights_file(self):
        return runner.weights_file()

    def prepare(self, ctx):
        return checks.check_scans(ctx.scans, self.cohort(ctx))

    def check(self, samples, ctx):
        return []  # check_scans has made them all

    def overview(self, samples):
        return samples.drop(columns=["path", "muscle"]).to_string(index=False)

    def write_checks(self, samples, problems, ctx):
        return sample_tables.write_checks(ctx.run, problems, samples)

    def settings(self, ctx):
        return {"cohort": ctx.options["cohort"], "variants": self.variants(ctx)}

    def is_done(self, row, ctx):
        return all((ctx.run.section(row.sample, v) / "section.json").exists() for v in self.variants(ctx))

    def measure(self, row, ctx):
        segment = ctx.options.get("segment")
        for v in self.variants(ctx):
            section.run(Path(row.path), ctx.run.section(row.sample, v), row.sample, v,
                        segment=segment and segment(row.sample))

    def summarise(self, done, failed, problems, ctx):
        if done.empty:
            return Summary(ExitStatus.INCOMPLETE, {}, state="failed", message="No section could be measured")
        animals, genotypes = summary.write_tables(ctx.run.root, ctx.run.tables, self.cohort(ctx),
                                                  self.variants(ctx)[0])
        text = f"{animals.round(3).to_string(index=False)}\n{genotypes.round(3).to_string(index=False)}"
        return Summary(ExitStatus.INCOMPLETE if failed else ExitStatus.DONE, {}, text=text)


def main() -> None:
    runner.main(Psr())


if __name__ == "__main__":
    main()
