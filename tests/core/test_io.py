import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from msq.core.io import Heartbeat, RunFolder, hold_lock, lock_is_held, partial_folder, read_json, write_json


def test_json_round_trip_and_missing_file(tmp_path):
    p = tmp_path / "x.json"
    write_json(p, {"a": 1, "b": [1.5, None]})
    assert json.loads(p.read_text()) == {"a": 1, "b": [1.5, None]} and not p.with_name("x.json.tmp").exists()
    assert read_json(p) == {"a": 1, "b": [1.5, None]}
    assert read_json(tmp_path / "missing.json") == {} and read_json(tmp_path / "missing.json", default=[]) == []
    p.write_text("{not json")
    assert read_json(p) == {}


def test_lock_is_exclusive_while_held(tmp_path):
    lock = tmp_path / "runner.lock"
    assert lock_is_held(lock) is False
    with hold_lock(lock) as free:
        assert free is True
        assert lock_is_held(lock) is True
        with hold_lock(lock, tries=2, wait_s=0.01) as second:
            assert second is False
    assert lock_is_held(lock) is False


HOLD = """
import sys
from pathlib import Path
from msq.core.io import hold_lock
with hold_lock(Path(sys.argv[1])) as free:
    print(free, flush=True)
    sys.stdin.read()
"""


def holder(lock: Path) -> subprocess.Popen:
    """Another process that holds the lock until it is ended."""
    proc = subprocess.Popen([sys.executable, "-c", HOLD, str(lock)], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, text=True)
    assert proc.stdout.readline().strip() == "True"
    return proc


def test_a_lock_held_by_another_process_is_seen_until_that_process_is_killed(tmp_path):
    lock = tmp_path / "runner.lock"
    proc = holder(lock)
    try:
        assert lock_is_held(lock) is True
        with hold_lock(lock, tries=2, wait_s=0.01) as free:
            assert free is False
    finally:
        proc.kill()
        proc.wait(timeout=30)
    assert lock_is_held(lock) is False  # the OS released it with the killed process
    with hold_lock(lock) as free:
        assert free is True


def test_heartbeat_beats_until_closed():
    beats = []
    hb = Heartbeat(lambda: beats.append(time.time()), interval_s=0.02)
    time.sleep(0.15)
    hb.close()
    n = len(beats)
    assert n >= 3
    time.sleep(0.1)
    assert len(beats) <= n + 1


def test_heartbeat_reports_a_failed_beat_once(capsys):
    def fail():
        raise OSError("disk gone")

    hb = Heartbeat(fail, interval_s=0.02)
    time.sleep(0.12)
    hb.close()
    assert capsys.readouterr().out.count("heartbeat not written") == 1


def test_partial_folder_replaces_the_target_only_when_complete(tmp_path):
    out = tmp_path / "section"
    out.mkdir()
    (out / "stale.txt").write_text("old")
    with partial_folder(out) as tmp:
        assert tmp.name == "section.partial" and tmp.is_dir()
        (tmp / "fibers.csv").write_text("new")
    assert (out / "fibers.csv").read_text() == "new" and not (out / "stale.txt").exists()
    assert not out.with_name("section.partial").exists()
    with pytest.raises(RuntimeError):
        with partial_folder(out) as tmp:
            (tmp / "half.csv").write_text("x")
            raise RuntimeError("crash")
    assert (out / "fibers.csv").exists() and out.with_name("section.partial").exists()  # the old result survives


def test_run_folder_layout():
    rf = RunFolder(Path("/run"))
    assert rf.samples == Path("/run/samples.csv") and rf.lock == Path("/run/runner.lock")
    assert rf.sections == Path("/run/sections") and rf.tables == Path("/run/tables")
    assert rf.settings.name == "run_settings.json" and rf.checks_json.name == "checks.json"
    assert rf.tasks == Path("/run/tasks.tsv")
    assert rf.section("QUAD_sec1") == Path("/run/sections/QUAD_sec1")
    assert rf.section("QUAD_sec1", "base") == Path("/run/sections/base/QUAD_sec1")
