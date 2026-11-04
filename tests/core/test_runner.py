"""The runner's contract, on a fake assay that records what it is asked to do. Written before the runner (wave 3,
decision D4), implemented in core.runner by task T3b."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from msq.core import runner
from msq.core.io import RunFolder, hold_lock, lock_is_held
from msq.core.problems import ExitStatus, Problem
from msq.core.runner import RunContext, Summary

SAMPLES = ["s1", "s2", "s3"]


class FakeAssay:
    name = "fake"

    def __init__(self, problems=(), fail=(), done=(), stop_after=None, summary_error=False):
        self.problems, self.fail, self.done, self.stop_after = list(problems), set(fail), set(done), stop_after
        self.summary_error = summary_error
        self.calls: list[tuple] = []

    def arguments(self, parser):
        parser.add_argument("--colour", default="grey")

    def device(self):
        return "cpu"

    def weights_file(self):
        return Path("no-weights.pt")

    def prepare(self, ctx):
        self.calls.append(("prepare",))
        t = pd.DataFrame({"sample": SAMPLES, "path": [str(ctx.scans / f"{s}.nd2") for s in SAMPLES]})
        t.to_csv(ctx.run.samples, index=False)
        return t, []

    def check(self, samples, ctx):
        self.calls.append(("check", ctx.device))
        return list(self.problems)

    def overview(self, samples):
        return f"{len(samples)} sections"

    def write_checks(self, samples, problems, ctx):
        text = "\n".join(map(str, problems)) or "no problems found"
        ctx.run.checks.write_text(text + "\n", encoding="utf-8")
        return text

    def settings(self, ctx):
        return {"rule": {"fake": True}, "colour": ctx.options.get("colour")}

    def is_done(self, row, ctx):
        return row.sample in self.done

    def measure(self, row, ctx):
        self.calls.append(("measure", row.sample))
        if row.sample in self.fail:
            raise RuntimeError(f"cannot read {row.sample}")
        self.done.add(row.sample)
        if row.sample == self.stop_after:
            ctx.run.stop.write_text("")

    def summarise(self, done, failed, problems, ctx):
        self.calls.append(("summarise", list(done["sample"]), list(failed)))
        if self.summary_error:
            raise ValueError("no tables")
        status = ExitStatus.INCOMPLETE if failed else ExitStatus.DONE
        return Summary(status, {"results": len(done)}, problems, text="results")


def start(tmp_path, assay, **kw):
    return runner.run(assay, tmp_path / "scans", tmp_path / "run", **kw)


def progress(tmp_path) -> dict:
    return json.loads((tmp_path / "run" / "progress.json").read_text())


def settings(tmp_path) -> dict:
    return json.loads((tmp_path / "run" / "run_settings.json").read_text())


def test_check_only_stops_after_the_checks(tmp_path):
    a = FakeAssay()
    assert start(tmp_path, a, check_only=True) == ExitStatus.DONE
    assert [c[0] for c in a.calls] == ["prepare", "check"] and a.calls[1] == ("check", "cpu")
    assert progress(tmp_path)["state"] == "checked"
    assert (tmp_path / "run" / "checks.txt").read_text() == "no problems found\n"
    assert not (tmp_path / "run" / "run_settings.json").exists()


def test_an_error_in_the_checks_stops_the_run_before_any_section(tmp_path):
    a = FakeAssay(problems=[Problem("error", "no controls")])
    assert start(tmp_path, a) == ExitStatus.CHECKS_FAILED
    assert "measure" not in [c[0] for c in a.calls] and progress(tmp_path)["state"] == "stopped"


def test_a_full_run_measures_what_is_not_done_and_records_it(tmp_path):
    a = FakeAssay(done={"s2"})
    assert start(tmp_path, a, colour="red") == ExitStatus.DONE
    assert [c for c in a.calls if c[0] == "measure"] == [("measure", "s1"), ("measure", "s3")]
    assert a.calls[-1] == ("summarise", SAMPLES, [])
    s = settings(tmp_path)
    assert {"started", "finished", "scans", "run_folder", "code_version", "software", "device", "cellpose_weights",
            "failed_sections", "rule", "colour", "results"} <= set(s)
    assert s["colour"] == "red" and s["results"] == 3 and s["failed_sections"] == []
    p = progress(tmp_path)
    assert p["state"] == "done" and p["n_done"] == 3 and p["n_failed"] == 0 and p["message"] == "Finished"
    assert (tmp_path / "run" / "run.log").read_text().count("s1") >= 1


def test_redo_measures_finished_sections_again(tmp_path):
    a = FakeAssay(done={"s1", "s2", "s3"})
    start(tmp_path, a, redo=True)
    assert [c[1] for c in a.calls if c[0] == "measure"] == SAMPLES


def test_a_failed_section_is_left_out_and_the_run_goes_on(tmp_path):
    a = FakeAssay(fail={"s2"})
    assert start(tmp_path, a) == ExitStatus.INCOMPLETE
    assert a.calls[-1] == ("summarise", ["s1", "s3"], ["s2"])
    sec = progress(tmp_path)["sections"]["s2"]
    assert sec["status"] == "failed" and sec["error"] == "RuntimeError: cannot read s2"
    assert settings(tmp_path)["failed_sections"] == ["s2"]
    assert progress(tmp_path)["message"] == "Finished; 1 section(s) failed"


def test_a_stop_request_pauses_after_the_section_in_work(tmp_path):
    a = FakeAssay(stop_after="s1")
    assert start(tmp_path, a) == ExitStatus.STOPPED
    assert [c[1] for c in a.calls if c[0] == "measure"] == ["s1"] and "summarise" not in [c[0] for c in a.calls]
    assert progress(tmp_path)["state"] == "paused" and "paused" in settings(tmp_path)
    assert not RunFolder(tmp_path / "run").stop.exists()
    b = FakeAssay(done=a.done)  # started again: goes on where it stopped
    assert start(tmp_path, b) == ExitStatus.DONE
    assert [c[1] for c in b.calls if c[0] == "measure"] == ["s2", "s3"]


def test_a_second_runner_does_not_start_and_writes_nothing(tmp_path):
    rf = RunFolder(tmp_path / "run")
    rf.root.mkdir(parents=True)
    with hold_lock(rf.lock):
        assert start(tmp_path, FakeAssay()) == ExitStatus.LOCKED
    assert not rf.log.exists() and not rf.progress.exists()


def test_an_unexpected_error_ends_the_run_as_crashed(tmp_path):
    assert start(tmp_path, FakeAssay(summary_error=True)) == ExitStatus.CRASHED
    p = progress(tmp_path)
    assert p["state"] == "failed" and p["message"] == "The run stopped: ValueError: no tables"
    assert "Traceback" in (tmp_path / "run" / "run.log").read_text()
    assert lock_is_held(RunFolder(tmp_path / "run").lock) is False  # released after the crash


def test_main_parses_the_runner_and_the_assay_options(tmp_path, monkeypatch):
    a = FakeAssay()
    monkeypatch.setattr(sys, "argv", ["fake", str(tmp_path / "scans"), str(tmp_path / "run"), "--check-only",
                                      "--colour", "blue"])
    with pytest.raises(SystemExit) as e:
        runner.main(a)
    assert e.value.code == ExitStatus.DONE and progress(tmp_path)["state"] == "checked"


def test_a_summary_without_problems_leaves_the_checks_as_they_were_and_can_end_the_run_as_failed(tmp_path):
    class Quiet(FakeAssay):
        writes = 0

        def write_checks(self, samples, problems, ctx):
            self.writes += 1
            return super().write_checks(samples, problems, ctx)

        def summarise(self, done, failed, problems, ctx):
            return Summary(ExitStatus.INCOMPLETE, {"results": 0}, state="failed", message="No section could be measured")

    a = Quiet()
    assert start(tmp_path, a) == ExitStatus.INCOMPLETE and a.writes == 1  # after the checks only
    p = progress(tmp_path)
    assert p["state"] == "failed" and p["message"] == "No section could be measured"
    assert settings(tmp_path)["results"] == 0 and "finished" in settings(tmp_path)


def test_the_context_carries_the_assay_options():
    ctx = RunContext(run=RunFolder(Path("r")), scans=Path("s"), options={"cohort": "x"})
    assert ctx.options["cohort"] == "x" and ctx.check_only is False and ctx.weights is None
