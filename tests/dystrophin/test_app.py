import json
import os
from pathlib import Path

import pytest
from fastapi import HTTPException

from msq.dystrophin import app  # noqa: E402
from msq.dystrophin import checks  # noqa: E402
from msq.core import io as runlock  # noqa: E402  (the lock moved to msq.core.io)
from msq.dystrophin import runs  # noqa: E402


class Alive:
    """A runner process that has not ended."""

    def poll(self):
        return None


def test_start_runner_never_starts_a_second_runner_on_a_folder(tmp_path, monkeypatch):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"runs_folder": str(tmp_path / "runs")}))
    monkeypatch.setattr(runs, "CONFIG", cfg)
    monkeypatch.setattr(app, "PROCS", {})
    notes, started = [], []
    monkeypatch.setattr(app.ui, "notify", lambda message, **kwargs: notes.append(message))
    monkeypatch.setattr(runs, "start", lambda run, check_only: started.append(check_only) or Alive())
    (tmp_path / "scans").mkdir()
    r = runs.create_run(tmp_path / "runs", "r1", tmp_path / "scans")

    with runlock.hold_lock(r.rf.lock):  # a runner from earlier, e.g. one started before the app was restarted
        assert not app.start_runner(r, check_only=True) and not app.start_runner(r, check_only=False)
    assert started == [] and len(notes) == 2 and "still working" in notes[0]

    assert app.start_runner(r, check_only=True) and started == [True]
    assert not app.start_runner(r, check_only=False)  # this app's own runner on the folder has not ended
    assert started == [True]


@pytest.fixture
def runs_folder(tmp_path, monkeypatch):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"runs_folder": str(tmp_path / "runs")}))
    monkeypatch.setattr(runs, "CONFIG", cfg)
    (tmp_path / "scans").mkdir()
    return tmp_path / "runs"


def test_the_borrow_box_follows_the_problem_code_not_its_wording(runs_folder):
    r = runs.create_run(runs_folder, "r1", runs_folder.parent / "scans")
    reworded = checks.Problem("error", "this muscle has no controls", "TA", code=checks.NO_CONTROLS)
    assert app.wants_borrow(r, [reworded])
    assert not app.wants_borrow(r, [checks.Problem("error", "no untreated control sections", "TA")])  # no code
    runs.set_reference(r, "/earlier/run")
    assert app.wants_borrow(runs.read_run(r.folder), [])  # a run that borrows keeps the box


def test_the_file_route_serves_run_files_only(runs_folder):
    r = runs.create_run(runs_folder, "r1", runs_folder.parent / "scans")
    (r.folder / "reports").mkdir()
    (r.folder / "reports" / "a.html").write_text("ok")
    (runs_folder.parent / "secret.txt").write_text("no")
    outside = [("..", "secret.txt"), ("r1", "../../secret.txt"), ("r1", str(runs_folder.parent / "secret.txt")),
               ("r1", "reports/missing.html")]
    try:
        os.symlink(runs_folder.parent / "secret.txt", r.folder / "link.txt")
        outside.append(("r1", "link.txt"))
    except OSError:  # Windows allows symlinks only with extra rights
        pass
    assert Path(app.run_file("r1", "reports/a.html").path).name == "a.html"
    for name, rel in outside:
        with pytest.raises(HTTPException) as e:
            app.run_file(name, rel)
        assert e.value.status_code == 404


def test_the_run_list_is_rebuilt_only_when_a_run_changed(runs_folder):
    r = runs.create_run(runs_folder, "r1", runs_folder.parent / "scans")
    before = app.list_key(runs_folder, runs.list_runs(runs_folder))
    assert app.list_key(runs_folder, runs.list_runs(runs_folder)) == before
    (r.folder / "progress.json").write_text(json.dumps({"state": "checked", "updated": "2026-10-21T10:00:00"}))
    assert app.list_key(runs_folder, runs.list_runs(runs_folder)) != before
