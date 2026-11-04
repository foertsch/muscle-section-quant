import json
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from msq.dystrophin import checks  # noqa: E402
from msq.core import io as runlock  # noqa: E402  (the lock moved to msq.core.io)
from msq.dystrophin import runs  # noqa: E402


def make_scans(root: Path, files: list[str]) -> Path:
    for rel in files:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")
    return root


def write_progress(folder: Path, **data) -> None:
    (folder / "progress.json").write_text(json.dumps(data))


def test_config_round_trip(tmp_path):
    cfg_path = tmp_path / "cfg.json"
    assert runs.load_config(cfg_path)["runs_folder"] == str(runs.DEFAULT_RUNS)
    runs.save_config({"runs_folder": "X:\\runs", "last_scans": ""}, cfg_path)
    assert runs.load_config(cfg_path)["runs_folder"] == "X:\\runs"
    cfg_path.write_text("not json")
    assert runs.load_config(cfg_path)["runs_folder"] == str(runs.DEFAULT_RUNS)


def test_create_run(tmp_path):
    scans = make_scans(tmp_path / "scans", ["a.nd2"])
    run = runs.create_run(tmp_path / "runs", "2026-10-20 TA batch", scans)
    assert run.state == "new" and run.scans == scans.resolve() and run.reference == ""
    with pytest.raises(ValueError, match="already exists"):
        runs.create_run(tmp_path / "runs", "2026-10-20 TA batch", scans)
    with pytest.raises(ValueError, match="letters"):
        runs.create_run(tmp_path / "runs", "../escape", scans)
    with pytest.raises(ValueError, match="does not exist"):
        runs.create_run(tmp_path / "runs", "other", tmp_path / "missing")
    assert runs.default_name(Path("/x/1104 TA (new)"), date(2026, 10, 20)) == "2026-10-20 1104 TA new"


def test_states(tmp_path):
    now = datetime(2026, 10, 20, 12, 0, 0)
    folder = tmp_path / "r"
    folder.mkdir()
    assert runs.read_run(folder).state == "new"
    write_progress(folder, state="measuring", heartbeat=(now - timedelta(seconds=90)).isoformat())
    assert runs.read_run(folder, now=now).state == "measuring"
    assert runs.read_run(folder, exit_code=1, now=now).state == "interrupted"  # its process ended
    write_progress(folder, state="measuring", heartbeat=(now - timedelta(minutes=30)).isoformat())
    assert runs.read_run(folder, now=now).state == "interrupted"  # no heartbeat for 30 min
    write_progress(folder, state="done", heartbeat=(now - timedelta(days=3)).isoformat())
    assert runs.read_run(folder, now=now).state == "done"


def test_a_locked_run_stays_active_however_old_its_heartbeat(tmp_path):
    """After the laptop slept, or when heartbeats could not be written, the runner can be alive with an old heartbeat;
    showing it as interrupted would offer to start a second runner on the same folder."""
    now = datetime(2026, 10, 20, 12, 0, 0)
    folder = tmp_path / "r"
    folder.mkdir()
    write_progress(folder, state="measuring", heartbeat=(now - timedelta(minutes=30)).isoformat())
    with runlock.hold_lock(folder / "runner.lock"):
        run = runs.read_run(folder, now=now)
        assert run.state == "measuring"
        assert runs.busy_run([run]) is run  # still holds the GPU
    assert runs.read_run(folder, now=now).state == "interrupted"  # lock released: the runner has ended

def test_list_runs_and_busy(tmp_path):
    for name, state, updated in (("a", "done", "2026-10-19T10:00:00"), ("b", "checking", "2026-10-20T10:00:00"),
                                 ("c", "measuring", "2026-10-20T09:00:00")):
        (tmp_path / name).mkdir()
        write_progress(tmp_path / name, state=state, updated=updated, heartbeat=runs.now_iso())
    (tmp_path / "not a run").mkdir()
    found = runs.list_runs(tmp_path)
    assert [r.name for r in found] == ["b", "c", "a"]
    assert runs.busy_run(found).name == "c"  # checking does not hold the GPU
    assert runs.busy_run(found, other_than=tmp_path / "c") is None
    assert runs.list_runs(tmp_path / "missing") == []


def test_apply_edits():
    samples = pd.DataFrame([
        {"sample": "TA_sec1_1106_20x_ctrl", "group": "ctrl", "muscle": "TA", "animal": "1106", "section": 1,
         "mag": "20x", "edited": False, "note": "", "path": "/s/a.nd2"},
        {"sample": "TA_sec2_1104_20x_treated", "group": "treated", "muscle": "TA", "animal": "1104", "section": 2,
         "mag": "20x", "edited": False, "note": "", "path": "/s/b.nd2"},
        {"sample": "TA_sec3_1104_20x_treated", "group": "treated", "muscle": "TA", "animal": "1104", "section": 3,
         "mag": "20x", "edited": False, "note": "", "path": "/s/c.nd2"}])
    out = runs.apply_edits(samples, {"TA_sec1_1106_20x_ctrl": {"animal": " 1003 "},
                                     "TA_sec2_1104_20x_treated": {"group": "ctrl"},
                                     "TA_sec3_1104_20x_treated": {"include": False}})
    assert list(out["sample"]) == ["TA_sec1_1003_20x_ctrl", "TA_sec2_1104_20x_ctrl"]
    assert list(out["path"]) == ["/s/a.nd2", "/s/b.nd2"]


def test_runner_command_and_stop(tmp_path):
    scans = make_scans(tmp_path / "scans", ["a.nd2"])
    run = runs.create_run(tmp_path / "runs", "r1", scans)
    cmd = runs.runner_command(run, check_only=True)
    assert cmd[1:3] == ["-m", "msq.dystrophin.run_pipeline"] and cmd[3:5] == [str(scans.resolve()), str(run.folder)]
    assert cmd[-1] == "--check-only" and "--reference" not in cmd
    runs.set_reference(run, "/old/run")
    run = runs.read_run(run.folder)
    assert runs.runner_command(run, check_only=False)[-2:] == ["--reference", "/old/run"]
    runs.request_stop(run)
    assert runs.stop_requested(run)


def test_start_runs_the_checks(tmp_path):
    """The real runner in its own process: empty files cannot be opened, so the checks stop the run."""
    scans = make_scans(tmp_path / "scans", ["Neg-ctrl-TA/1106/TA sec1 1106 20x.nd2", "1104 TA/TA sec2 1104 20x.nd2"])
    run = runs.create_run(tmp_path / "runs", "r1", scans)
    proc = runs.start(run, check_only=True)
    assert proc.wait(timeout=300) == 1
    run = runs.read_run(run.folder, exit_code=proc.returncode)
    assert run.state == "stopped"
    assert len(run.samples()) == 2
    errors = checks.errors(run.problems())
    assert len(errors) == 2 and all("cannot be opened" in p.message for p in errors)


def test_helpers(tmp_path):
    assert runs.fmt_duration(None) == "estimating"
    assert runs.fmt_duration(20) == "under a minute"
    assert runs.fmt_duration(14 * 60) == "14 min"
    assert runs.fmt_duration(125 * 60) == "2 h 05 min"
    make_scans(tmp_path, ["b/x.nd2", "a/y.nd2", "a/z.txt", ".hidden/w.nd2"])
    assert [p.name for p in runs.subfolders(tmp_path)] == ["a", "b"]
    assert runs.count_scans(tmp_path) == 3
    (tmp_path / "log.txt").write_text("\n".join(str(i) for i in range(30)))
    assert runs.tail(tmp_path / "log.txt", 3) == "27\n28\n29"
    assert runs.folder_roots()


def test_windows_drives_are_listed_without_touching_them(monkeypatch):
    probed = []
    with monkeypatch.context() as m:  # only around the call, so pytest's own reporting is not affected
        m.setattr(runs.os, "listdrives", lambda: ["C:\\", "Z:\\"], raising=False)
        m.setattr(runs.Path, "exists", lambda self: probed.append(self) or True)
        drives = [str(d) for d in runs.windows_drives()]
    assert drives == ["C:\\", "Z:\\"] and probed == []
