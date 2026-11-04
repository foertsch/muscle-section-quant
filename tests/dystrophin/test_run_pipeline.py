import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from msq.dystrophin import checks  # noqa: E402
from msq.dystrophin import run_pipeline as rp  # noqa: E402
from msq.dystrophin import settings  # noqa: E402
from msq.core import io as core_io  # noqa: E402
from msq.core import runner as core_runner  # noqa: E402
from msq.core.synthetic.dystrophin import HEADER, write_section  # noqa: E402

CONTROLS = ["Neg-ctrl-TA/1106/TA sec1 1106 20x.nd2", "Neg-ctrl-TA/1107/TA sec1 1107 20x.nd2",
            "Neg-ctrl-TA/1108/TA sec1 1108 20x.nd2"]
TREATED = ["1104 TA/TA sec2 1104 20x.nd2", "1104 TA/TA sec3 1104 20x.nd2", "1105 TA/TA sec1 1105 20x.nd2"]


def make_scans(root: Path, files: list[str]) -> Path:
    for rel in files:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")
    return root


@pytest.fixture
def fake_pipeline(monkeypatch):
    """Header reading, segmentation, maps and the image reports replaced: the rest runs for real."""
    calls = {"measured": [], "resegment": [], "decision": [], "compare": []}

    def measure(path, out_root, p=None, resegment=False, name=None, provenance=None):
        if "fail" in calls and name in calls["fail"]:
            raise RuntimeError("out of GPU memory")
        calls["measured"].append(name)
        calls["resegment"].append(resegment)
        write_section(out_root, name, source=path)
        if provenance is not None:  # recorded as measure.run records a section it segments
            info_path = out_root / name / "run_info.json"
            info = json.loads(info_path.read_text())
            info.update(code_version=provenance.code_version, cellpose_weights=provenance.weights)
            info_path.write_text(json.dumps(info))

    monkeypatch.setattr(checks, "nd2_header", lambda p: HEADER)
    monkeypatch.setattr(rp.dq, "run", measure)
    monkeypatch.setattr(rp, "gpu_device", lambda: "test device")
    monkeypatch.setattr(rp, "weights_file", lambda: Path("no weights here"))  # never hash the real 1.2 GB
    monkeypatch.setattr(rp.qc, "dystrophin_maps", lambda dirs: None)
    monkeypatch.setattr(rp.dr, "build", lambda results, tables, nd2, muscles, out: calls["decision"].append(muscles))
    monkeypatch.setattr(rp.cr, "build", lambda results, tables, nd2, muscle, out: calls["compare"].append(muscle))
    return calls


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_full_run_then_resume(tmp_path, fake_pipeline):
    scans = make_scans(tmp_path / "scans", CONTROLS + TREATED)
    run = tmp_path / "run"
    assert rp.run(scans, run) == 0
    assert len(fake_pipeline["measured"]) == 6
    assert fake_pipeline["decision"] == [["TA"]] and fake_pipeline["compare"] == ["TA"]

    th = read(run / "tables" / "thresholds.json")
    ctrl = pd.read_csv(run / "tables" / "calls.csv.gz")
    ctrl = ctrl[ctrl["sample"].str.endswith("_ctrl")]["score_50"].dropna()
    assert th["fpr"] == 0.02
    assert th["muscles"]["TA"]["threshold"] == pytest.approx(np.percentile(ctrl, 98))
    assert th["muscles"]["TA"]["threshold_source"] == "controls in this run (3 animals, 3 sections)"

    progress = read(run / "progress.json")
    assert progress["state"] == "done" and progress["n_done"] == 6 and progress["n_failed"] == 0
    info = read(run / "run_settings.json")
    assert info["device"] == "test device" and info["rule"]["threshold_percentile_of_controls"] == 98.0
    assert {r["animal"] for r in info["results"]} == {"1104", "1105", "1106", "1107", "1108"}
    treated = [r["pct_positive_50"] for r in info["results"] if r["group"] == "treated"]
    assert min(treated) > 30  # fake treated sections have 40% positives
    for f in ("samples.csv", "checks.txt", "run.log", "tables/sections.csv", "tables/animals.csv",
              "tables/leave_one_animal_out.csv", "tables/fiber_size_by_animal.csv",
              "figures/pct_positive_by_animal.png", "figures/pct_positive_call.png",
              "figures/fiber_size_distribution.png"):
        assert (run / f).exists(), f

    # resume: finished sections are not measured again; a hand-edited table is kept
    fake_pipeline["measured"].clear()
    assert rp.run(scans, run) == 0
    assert fake_pipeline["measured"] == []
    assert "using the existing sample table" in (run / "run.log").read_text(encoding="utf-8")


def test_errors_stop_the_run_before_measuring(tmp_path, fake_pipeline):
    scans = make_scans(tmp_path / "scans", TREATED)
    run = tmp_path / "run"
    assert rp.run(scans, run) == 1
    assert fake_pipeline["measured"] == []
    assert read(run / "progress.json")["state"] == "stopped"
    assert "no untreated control sections" in (run / "checks.txt").read_text(encoding="utf-8")


def test_check_only(tmp_path, fake_pipeline):
    scans = make_scans(tmp_path / "scans", CONTROLS + TREATED)
    run = tmp_path / "run"
    assert rp.run(scans, run, check_only=True) == 0
    assert fake_pipeline["measured"] == []
    assert read(run / "progress.json")["state"] == "checked"
    assert len(pd.read_csv(run / "samples.csv")) == 6


def test_borrowed_threshold(tmp_path, fake_pipeline):
    first = tmp_path / "first"
    assert rp.run(make_scans(tmp_path / "scans1", CONTROLS + TREATED), first) == 0
    threshold = read(first / "tables" / "thresholds.json")["muscles"]["TA"]["threshold"]

    second = tmp_path / "second"
    fake_pipeline["decision"].clear()
    assert rp.run(make_scans(tmp_path / "scans2", TREATED[:1]), second, reference=first) == 0
    th = read(second / "tables" / "thresholds.json")["muscles"]["TA"]
    assert th["threshold"] == pytest.approx(threshold)
    assert th["threshold_source"] == f"borrowed from {first.resolve()}"
    assert th["50"]["ctrl_fpr_pooled"] is None
    assert fake_pipeline["decision"] == []  # no controls, no decision report
    assert "borrowed" in (second / "checks.txt").read_text(encoding="utf-8")


def test_a_run_made_before_the_rename_can_still_lend_its_threshold(tmp_path, fake_pipeline):
    """thresholds.json and run_settings.json of a run from when the threshold was still called "bar" (and the
    measurement parameter "inverted_cutoff"), written here key by key as that run wrote them."""
    first = tmp_path / "first"
    assert rp.run(make_scans(tmp_path / "scans1", CONTROLS + TREATED), first) == 0
    th = read(first / "tables" / "thresholds.json")
    old = {"rule": "one bar per muscle", "fpr": th["fpr"], "bar_coverage": th["threshold_coverage"],
           "muscles": {m: {"bar": v["threshold"], "bar_source": "borrowed from elsewhere",
                           "50": {"sector_cutoff": v["threshold"]}} for m, v in th["muscles"].items()}}
    (first / "tables" / "thresholds.json").write_text(json.dumps(old))
    info = read(first / "run_settings.json")
    params = info["measurement_params"]
    params["inverted_cutoff"] = params.pop("inverted_threshold")
    (first / "run_settings.json").write_text(json.dumps(info))

    got = rp.load_reference(first)
    assert got["TA"]["threshold"] == pytest.approx(th["muscles"]["TA"]["threshold"])
    assert got["TA"]["source"].endswith("(borrowed from elsewhere)")
    assert got["TA"]["checked"] is False  # no weights here
    second = tmp_path / "second"
    assert rp.run(make_scans(tmp_path / "scans2", TREATED[:1]), second, reference=first) == 0
    assert read(second / "tables" / "thresholds.json")["muscles"]["TA"]["threshold"] == pytest.approx(
        th["muscles"]["TA"]["threshold"])


def test_current_keys_reads_every_old_key_at_any_depth():
    old = {"bar_coverage": 0.5, "muscles": [{"bar": 1.0, "bar_source": "x", "50": {"sector_cutoff": 2.0}}],
           "measurement_params": {"inverted_cutoff": 150.0, "band_um": 1.5}}
    assert settings.current_keys(old) == {
        "threshold_coverage": 0.5, "muscles": [{"threshold": 1.0, "threshold_source": "x",
                                                "50": {"sector_threshold": 2.0}}],
        "measurement_params": {"inverted_threshold": 150.0, "band_um": 1.5}}
    assert settings.current_keys(None) is None and settings.current_keys(3.0) == 3.0


def test_reference_with_another_rule_is_refused(tmp_path):
    (tmp_path / "thresholds.json").write_text(json.dumps({"fpr": 0.05, "threshold_coverage": 0.5, "muscles": {}}))
    with pytest.raises(ValueError, match="another rule"):
        rp.load_reference(tmp_path)
    with pytest.raises(FileNotFoundError):
        rp.load_reference(tmp_path / "missing")


def test_a_failed_section_is_left_out(tmp_path, fake_pipeline):
    fake_pipeline["fail"] = {"TA_sec3_1104_20x_treated"}
    run = tmp_path / "run"
    assert rp.run(make_scans(tmp_path / "scans", CONTROLS + TREATED), run) == 2
    progress = read(run / "progress.json")
    assert progress["n_failed"] == 1 and progress["state"] == "done"
    assert "out of GPU memory" in progress["sections"]["TA_sec3_1104_20x_treated"]["error"]
    sections = pd.read_csv(run / "tables" / "sections.csv")
    assert "TA_sec3_1104_20x_treated" not in set(sections["sample"]) and len(sections) == 5
    assert read(run / "run_settings.json")["failed_sections"] == ["TA_sec3_1104_20x_treated"]


def test_progress_estimates_time_left(tmp_path):
    p = rp.Progress(tmp_path / "progress.json", ["a", "b", "c"])
    p.section("a", "done", seconds=100)
    d = read(tmp_path / "progress.json")
    assert d["n_done"] == 1 and d["eta_s"] == 200
    p.section("b", "failed", error="x")
    d = read(tmp_path / "progress.json")
    assert d["n_failed"] == 1 and d["eta_s"] == 100


def test_muscles_with_different_animals_finish(tmp_path, fake_pipeline):
    files = ["Neg-ctrl/1106/QUAD sec1 1106 20x.nd2", "Neg-ctrl/1107/QUAD sec1 1107 20x.nd2", "1104/QUAD sec1 1104 20x.nd2",
             "Neg-ctrl/1106/TA sec1 1106 20x.nd2", "Neg-ctrl/1107/TA sec1 1107 20x.nd2", "1105/TA sec1 1105 20x.nd2"]
    run = tmp_path / "run"
    assert rp.run(make_scans(tmp_path / "scans", files), run) == 0
    assert read(run / "progress.json")["state"] == "done"
    assert (run / "figures" / "fiber_size_distribution.png").exists()


def test_a_failed_fiber_size_figure_is_a_warning(tmp_path, fake_pipeline, monkeypatch):
    def broken(*args):
        raise RuntimeError("cannot draw")

    monkeypatch.setattr(rp.pfs, "make", broken)
    run = tmp_path / "run"
    assert rp.run(make_scans(tmp_path / "scans", CONTROLS + TREATED), run) == 2
    assert read(run / "progress.json")["state"] == "done"
    assert (run / "tables" / "animals.csv").exists() and fake_pipeline["compare"] == ["TA"]
    assert "fiber size figure and table not written (RuntimeError: cannot draw)" in (run / "checks.txt").read_text(
        encoding="utf-8")


def test_write_json_retries_a_refused_replace(tmp_path, monkeypatch):
    """On Windows the replace fails while the app has progress.json open."""
    real, refused = core_io.os.replace, []

    def replace(src, dst):
        if len(refused) < 2:
            refused.append(dst)
            raise PermissionError(13, "the file is open in another process")
        real(src, dst)

    monkeypatch.setattr(core_io.os, "replace", replace)
    monkeypatch.setattr(core_io.time, "sleep", lambda s: None)
    rp.write_json(tmp_path / "x.json", {"a": 1})
    assert len(refused) == 2 and read(tmp_path / "x.json") == {"a": 1}


def test_the_heartbeat_survives_a_failed_write(tmp_path, monkeypatch):
    real, calls = rp.write_json, []

    def flaky(path, data):
        calls.append(path)
        if len(calls) == 1:
            raise PermissionError(13, "the file is open in another process")
        real(path, data)

    monkeypatch.setattr(core_runner, "write_json", flaky)  # Progress writes through the runner's name
    monkeypatch.setattr(rp.Progress, "HEARTBEAT_S", 0.02)
    p = rp.Progress(tmp_path / "progress.json", ["a"])
    deadline = time.time() + 10
    while len(calls) < 4 and time.time() < deadline:
        time.sleep(0.01)
    p.close()
    assert len(calls) >= 4 and (tmp_path / "progress.json").exists()  # beats went on after the failed one


def test_main_exits_4_when_the_run_folder_cannot_be_made(tmp_path, monkeypatch):
    blocker = tmp_path / "a file"
    blocker.write_text("")
    err = io.StringIO()
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.setattr(sys, "stderr", err)
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", str(tmp_path / "scans"), str(blocker / "run")])
    with pytest.raises(SystemExit) as e:
        rp.main()
    assert e.value.code == 4 and "Traceback" in err.getvalue()


def test_a_stop_asked_for_during_the_last_section_lets_the_run_finish(tmp_path, fake_pipeline, monkeypatch):
    run = tmp_path / "run"
    real_run = rp.dq.run

    def measure_then_ask_to_stop(path, out_root, p=None, resegment=False, name=None, provenance=None):
        real_run(path, out_root, p, resegment, name, provenance=provenance)
        if len(fake_pipeline["measured"]) == 6:
            (run / "stop_requested").write_text("")

    monkeypatch.setattr(rp.dq, "run", measure_then_ask_to_stop)
    assert rp.run(make_scans(tmp_path / "scans", CONTROLS + TREATED), run) == 0
    assert read(run / "progress.json")["state"] == "done" and not (run / "stop_requested").exists()


def test_an_animal_number_with_a_slash_stops_at_the_checks(tmp_path, fake_pipeline):
    scans, run = make_scans(tmp_path / "scans", CONTROLS + TREATED), tmp_path / "run"
    assert rp.run(scans, run, check_only=True) == 0
    samples = checks.read_samples(run / "samples.csv")
    samples.loc[samples["sample"] == "TA_sec3_1104_20x_treated", "animal"] = "1104/2"  # typed in the app
    samples["sample"] = [rp.checks.manifest.sample_id(r) for r in samples.to_dict(orient="records")]
    samples.to_csv(run / "samples.csv", index=False)
    assert rp.run(scans, run) == 1
    assert fake_pipeline["measured"] == []
    assert "must be letters, digits or '-'" in (run / "checks.txt").read_text(encoding="utf-8")


def test_a_measured_section_missing_from_the_results_is_not_silent(tmp_path, fake_pipeline, monkeypatch):
    real = rp.fs.cal.load_results
    monkeypatch.setattr(rp.fs.cal, "load_results",
                        lambda results: {s: d for s, d in real(results).items() if s != "TA_sec3_1104_20x_treated"})
    run = tmp_path / "run"
    assert rp.run(make_scans(tmp_path / "scans", CONTROLS + TREATED), run) == 2
    assert "TA_sec3_1104_20x_treated: measured, but its results were not found" in (run / "checks.txt").read_text(
        encoding="utf-8")


def test_scans_added_after_the_table_was_built_are_noted(tmp_path, fake_pipeline):
    scans, run = make_scans(tmp_path / "scans", CONTROLS + TREATED), tmp_path / "run"
    assert rp.run(scans, run, check_only=True) == 0
    make_scans(scans, ["Neg-ctrl-TA/1108/TA sec2 1108 20x.nd2"])
    assert rp.run(scans, run, check_only=True) == 0
    text = (run / "checks.txt").read_text(encoding="utf-8")
    assert "NOTE: 1 scan is in the folder but not in the sample table" in text and "TA sec2 1108 20x.nd2" in text
    assert len(checks.read_samples(run / "samples.csv")) == 6  # the table itself is not changed


def test_a_section_whose_scan_changed_is_measured_again(tmp_path, fake_pipeline):
    scans, run = make_scans(tmp_path / "scans", CONTROLS + TREATED), tmp_path / "run"
    assert rp.run(scans, run) == 0
    rescan = make_scans(scans, ["1104 TA/TA sec2 1104 20x rescan.nd2"]) / "1104 TA/TA sec2 1104 20x rescan.nd2"
    samples = checks.read_samples(run / "samples.csv")
    samples.loc[samples["sample"] == "TA_sec2_1104_20x_treated", "path"] = str(rescan.resolve())  # corrected by hand
    samples.to_csv(run / "samples.csv", index=False)
    fake_pipeline["measured"].clear()
    fake_pipeline["resegment"].clear()
    assert rp.run(scans, run) == 0
    assert fake_pipeline["measured"] == ["TA_sec2_1104_20x_treated"] and fake_pipeline["resegment"] == [True]
    assert "come from another scan" in (run / "run.log").read_text(encoding="utf-8")


def test_a_borrowed_threshold_keeps_where_it_came_from(tmp_path, fake_pipeline):
    first, second, third = tmp_path / "first", tmp_path / "second", tmp_path / "third"
    assert rp.run(make_scans(tmp_path / "s1", CONTROLS + TREATED), first) == 0
    assert rp.run(make_scans(tmp_path / "s2", TREATED[:1]), second, reference=first) == 0
    assert rp.run(make_scans(tmp_path / "s3", TREATED[1:2]), third, reference=second) == 0
    th = read(third / "tables" / "thresholds.json")["muscles"]["TA"]
    assert th["threshold_source"] == f"borrowed from {second.resolve()} (borrowed from {first.resolve()})"
    assert th["threshold"] == read(first / "tables" / "thresholds.json")["muscles"]["TA"]["threshold"]


def test_a_reference_measured_with_other_settings_is_refused(tmp_path, fake_pipeline):
    first = tmp_path / "first"
    assert rp.run(make_scans(tmp_path / "s1", CONTROLS + TREATED), first) == 0
    info = read(first / "run_settings.json")
    info["measurement_params"]["band_um"] = 2.0
    (first / "run_settings.json").write_text(json.dumps(info))
    with pytest.raises(ValueError, match=r"other settings \(band_um\)"):
        rp.load_reference(first)
    second = tmp_path / "second"
    assert rp.run(make_scans(tmp_path / "s2", TREATED[:1]), second, reference=first) == 1
    assert "measured with other settings" in (second / "checks.txt").read_text(encoding="utf-8")


def test_a_reference_without_settings_is_used_with_a_warning(tmp_path, fake_pipeline):
    first = tmp_path / "first"
    assert rp.run(make_scans(tmp_path / "s1", CONTROLS + TREATED), first) == 0
    bare = tmp_path / "bare" / "thresholds.json"  # e.g. sent by e-mail, without its run folder
    bare.parent.mkdir()
    bare.write_text((first / "tables" / "thresholds.json").read_text(encoding="utf-8"), encoding="utf-8")
    second = tmp_path / "second"
    assert rp.run(make_scans(tmp_path / "s2", TREATED[:1]), second, reference=bare) == 0
    assert "could not be compared" in (second / "checks.txt").read_text(encoding="utf-8")


def test_a_failed_report_is_a_warning_with_exit_2(tmp_path, fake_pipeline, monkeypatch):
    def broken(*args):
        raise RuntimeError("cannot read the scan")

    monkeypatch.setattr(rp.cr, "build", broken)
    run = tmp_path / "run"
    assert rp.run(make_scans(tmp_path / "scans", CONTROLS + TREATED), run) == 2
    (p,) = [p for p in read(run / "checks.json")["problems"] if p["code"] == checks.REPORT_FAILED]
    assert p["where"] == "ta_ctrl_vs_treated.html" and "cannot read the scan" in p["message"]


def test_a_run_records_its_versions_and_weights(tmp_path, fake_pipeline, monkeypatch):
    weights = tmp_path / "cpsam_v2"
    weights.write_bytes(b"weights")
    monkeypatch.setattr(rp, "weights_file", lambda: weights)
    run = tmp_path / "run"
    assert rp.run(make_scans(tmp_path / "scans", CONTROLS + TREATED), run) == 0
    info = read(run / "run_settings.json")
    assert info["code_version"] and {"cellpose", "torch"} <= set(info["software"])
    assert info["cellpose_weights"] == {"file": "cpsam_v2", "bytes": 7, "sha256": rp.runinfo.weights(weights)["sha256"]}


def weights_at(monkeypatch, path: Path, content: bytes | None) -> Path:
    """This computer's Cellpose weights for the runner: a small file with content, or none yet."""
    if content is not None:
        path.write_bytes(content)
    monkeypatch.setattr(rp, "weights_file", lambda: path)
    return path


def test_other_weights_give_a_warning_not_an_error(tmp_path, fake_pipeline, monkeypatch):
    weights_at(monkeypatch, tmp_path / "cpsam_v2", b"other weights")
    run = tmp_path / "run"
    assert rp.run(make_scans(tmp_path / "scans", CONTROLS + TREATED), run) == 0
    assert [p["code"] for p in read(run / "checks.json")["problems"]].count(checks.OTHER_WEIGHTS) == 1


def test_weights_downloaded_by_the_first_segmentation_are_checked_too(tmp_path, fake_pipeline, monkeypatch):
    w = weights_at(monkeypatch, tmp_path / "cpsam_v2", None)
    real_run = rp.dq.run

    def measure_and_download(*args, **kwargs):
        if not w.exists():
            w.write_bytes(b"downloaded weights")
        return real_run(*args, **kwargs)

    monkeypatch.setattr(rp.dq, "run", measure_and_download)
    run = tmp_path / "run"
    assert rp.run(make_scans(tmp_path / "scans", CONTROLS + TREATED), run) == 0
    assert read(run / "run_settings.json")["cellpose_weights"]["bytes"] == len(b"downloaded weights")
    assert "not the reference ones" in (run / "checks.txt").read_text(encoding="utf-8")


def test_a_threshold_is_not_borrowed_across_different_weights(tmp_path, fake_pipeline, monkeypatch):
    first, second = tmp_path / "first", tmp_path / "second"
    weights_at(monkeypatch, tmp_path / "a", b"weights A")
    assert rp.run(make_scans(tmp_path / "s1", CONTROLS + TREATED), first) == 0
    assert rp.run(make_scans(tmp_path / "s2", TREATED[:1]), second, reference=first) == 0  # same weights: fine
    assert "could not be compared" not in (second / "checks.txt").read_text(encoding="utf-8")

    weights_at(monkeypatch, tmp_path / "b", b"weights B")
    third = tmp_path / "third"
    assert rp.run(make_scans(tmp_path / "s3", TREATED[:1]), third, reference=first) == 1
    assert "measured with other Cellpose weights" in (third / "checks.txt").read_text(encoding="utf-8")


def test_a_reference_without_recorded_weights_is_used_with_a_warning(tmp_path, fake_pipeline, monkeypatch):
    first, second = tmp_path / "first", tmp_path / "second"
    weights_at(monkeypatch, tmp_path / "a", b"weights A")
    assert rp.run(make_scans(tmp_path / "s1", CONTROLS + TREATED), first) == 0
    info = read(first / "run_settings.json")
    del info["cellpose_weights"]  # a run from before weights were recorded
    (first / "run_settings.json").write_text(json.dumps(info))
    assert rp.run(make_scans(tmp_path / "s2", TREATED[:1]), second, reference=first) == 0
    assert "Cellpose weights could not be compared" in (second / "checks.txt").read_text(encoding="utf-8")


def stop_after(n: int, run: Path, fake_pipeline, monkeypatch) -> None:
    """Ask the runner to stop once n sections are measured."""
    real_run = rp.dq.run

    def measure_then_stop(path, out_root, p=None, resegment=False, name=None, provenance=None):
        real_run(path, out_root, p, resegment, name, provenance=provenance)
        if len(fake_pipeline["measured"]) == n:
            (run / "stop_requested").write_text("")

    monkeypatch.setattr(rp.dq, "run", measure_then_stop)


def test_every_section_records_the_code_and_weights_that_made_it(tmp_path, fake_pipeline, monkeypatch):
    weights_at(monkeypatch, tmp_path / "cpsam_v2", b"weights A")
    monkeypatch.setattr(rp.dq.runinfo, "code_version", lambda: "v9.9.9")
    run = tmp_path / "run"
    assert rp.run(make_scans(tmp_path / "scans", CONTROLS + TREATED), run) == 0
    for s in checks.read_samples(run / "samples.csv")["sample"]:
        info = read(run / "sections" / s / "run_info.json")
        assert info["code_version"] == "v9.9.9" and info["cellpose_weights"]["bytes"] == len(b"weights A")
    assert read(run / "run_settings.json")["code_version"] == "v9.9.9"
    assert not [p for p in read(run / "checks.json")["problems"]
                if p["code"] in (checks.MIXED_CODE, checks.MIXED_WEIGHTS)]


def test_a_resume_with_other_code_or_weights_warns(tmp_path, fake_pipeline, monkeypatch):
    scans, run = make_scans(tmp_path / "scans", CONTROLS + TREATED), tmp_path / "run"
    weights_at(monkeypatch, tmp_path / "a", b"weights A")
    monkeypatch.setattr(rp.dq.runinfo, "code_version", lambda: "v1.0.0")
    stop_after(2, run, fake_pipeline, monkeypatch)
    assert rp.run(scans, run) == 3  # two sections measured, then stopped

    weights_at(monkeypatch, tmp_path / "b", b"weights B")
    monkeypatch.setattr(rp.dq.runinfo, "code_version", lambda: "v1.1.0")
    assert rp.run(scans, run) == 0  # the other four, with other code and other weights
    problems = {p["code"]: p for p in read(run / "checks.json")["problems"]}
    assert "2 section(s) were measured by other code (v1.0.0) than this run (v1.1.0)" in problems[checks.MIXED_CODE]["message"]
    assert "2 section(s) were segmented with other Cellpose weights" in problems[checks.MIXED_WEIGHTS]["message"]
    assert problems[checks.MIXED_CODE]["level"] == "warning"  # a warning: the run finishes


def test_a_resume_over_sections_without_a_record_warns(tmp_path, fake_pipeline, monkeypatch):
    scans, run = make_scans(tmp_path / "scans", CONTROLS + TREATED), tmp_path / "run"
    assert rp.run(scans, run) == 0
    for s in checks.read_samples(run / "samples.csv")["sample"][:3]:  # as dystrophin-quant v1.0.0 wrote them
        info = read(run / "sections" / s / "run_info.json")
        del info["code_version"], info["cellpose_weights"]
        (run / "sections" / s / "run_info.json").write_text(json.dumps(info))
    assert rp.run(scans, run) == 0
    text = (run / "checks.txt").read_text(encoding="utf-8")
    assert "3 section(s) carry no record of the code that measured them" in text


def test_a_reader_replaces_the_nd2_files(tmp_path, monkeypatch):
    """The runner reads scans through a reader (scans.py): synthetic .npz sections here, ND2 files by default."""
    import tifffile

    from msq.core.synthetic.dystrophin import synthetic_section
    from msq.dystrophin.scans import NpzScans, save_npz

    monkeypatch.setattr(rp, "gpu_device", lambda: "test device")  # else a machine without a GPU adds a warning
    scans, run = tmp_path / "scans", tmp_path / "run"
    built = {}
    for rel, group in (("Neg-ctrl-TA/1001/TA sec1 1001 20x.npz", "ctrl"), ("Neg-ctrl-TA/1002/TA sec1 1002 20x.npz", "ctrl"),
                       ("1101 TA/TA sec1 1101 20x.npz", "treated")):
        built[(scans / rel).resolve()] = s = synthetic_section(len(built) + 1, group, size=512)
        save_npz(scans / rel, s)
    reader = NpzScans()
    assert rp.run(scans, run, check_only=True, reader=reader) == 0
    for sample, path in zip(checks.read_samples(run / "samples.csv")["sample"], checks.read_samples(run / "samples.csv")["path"]):
        (run / "sections" / sample).mkdir(parents=True)
        tifffile.imwrite(run / "sections" / sample / "labels_ds4.tif", built[Path(path)]["labels_ds"])
    seen = []  # the reports need sections larger than these (the demo test builds them); here: do they get the reader
    monkeypatch.setattr(rp.dr, "build", lambda *args, **kwargs: seen.append(kwargs.get("read")))
    monkeypatch.setattr(rp.cr, "build", lambda *args, **kwargs: seen.append(kwargs.get("read")))
    assert rp.run(scans, run, reader=reader) == 0
    assert len(seen) == 2 and all(r == reader.read for r in seen)
    assert len(pd.read_csv(run / "tables" / "sections.csv")) == 3
    assert read(run / "checks.json")["problems"] == []


def test_the_sample_table_looks_for_the_readers_files(tmp_path):
    (tmp_path / "Neg-ctrl-TA" / "1001").mkdir(parents=True)
    (tmp_path / "Neg-ctrl-TA" / "1001" / "TA sec1 1001 20x.npz").write_bytes(b"")
    table, problems = checks.samples_from_folder(tmp_path)
    assert table.empty and "no .nd2 scans found" in problems[0].message
    table, problems = checks.samples_from_folder(tmp_path, suffix=".npz")
    assert list(table["sample"]) == ["TA_sec1_1001_20x_ctrl"] and problems == []
