"""Files a run writes while it works: JSON written atomically, the lock that says a runner owns a run folder, a
heartbeat that shows the process is alive, folders written whole or not at all, and the run folder layout.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator


# ---------------------------------------------------------------- JSON


def write_json(path: Path, data: dict, tries: int = 10) -> None:
    """Write via a temporary file, so a reader never sees half a file. On Windows the replace fails while another
    process has the file open (the app reading progress.json), so it is retried for up to a second."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    for attempt in range(tries):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == tries - 1:
                raise
            time.sleep(0.1)


def read_json(path: Path, default=None):
    """The parsed file, or default when it is missing or not valid JSON (a reader racing a writer)."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {} if default is None else default


# ---------------------------------------------------------------- the runner's lock


def _lock(f) -> bool | None:
    """Lock f without waiting: True if locked, False if another process holds it, None if this file system has
    no locks."""
    try:
        if os.name == "nt":
            import msvcrt

            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, PermissionError):  # Windows reports a held lock as EACCES
        return False
    except OSError:
        return None
    return True


def _unlock(f) -> None:
    if os.name == "nt":
        import msvcrt

        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


@contextmanager
def hold_lock(path: Path, tries: int = 5, wait_s: float = 0.2) -> Iterator[bool]:
    """Hold the lock for the with-block; yields False if another process holds it. It tries a few times, because
    lock_is_held() takes the lock for a moment when it looks. The operating system releases the lock when the
    process ends, also when it is killed; where the file system has no locks the block runs without one."""
    with open(path, "a+") as f:
        for _ in range(tries):
            got = _lock(f)
            if got is not False:
                break
            time.sleep(wait_s)
        try:
            yield got is not False
        finally:
            if got:
                _unlock(f)


def lock_is_held(path: Path) -> bool | None:
    """True if a process holds the lock, False if not or if there is no lock file, None if locks are not
    supported here."""
    if not path.exists():
        return False
    try:
        f = open(path, "a+")
    except OSError:
        return None
    with f:
        got = _lock(f)
        if got:
            _unlock(f)
    return None if got is None else not got


# ---------------------------------------------------------------- heartbeat


class Heartbeat:
    """A daemon thread that calls beat() every interval while the run is alive, so a reader can tell a long step
    from a process that died. One failed call is reported once; the next beat tries again."""

    def __init__(self, beat: Callable[[], None], interval_s: float = 60.0):
        self.beat, self.interval_s = beat, interval_s
        self._closed = threading.Event()
        self._reported = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._closed.set()

    def _run(self) -> None:
        while not self._closed.wait(self.interval_s):
            try:
                self.beat()
            except OSError as e:
                if not self._reported:
                    print(f"heartbeat not written ({e}); trying again every {self.interval_s:.0f} s", flush=True)
                    self._reported = True


# ---------------------------------------------------------------- folders written whole


@contextmanager
def partial_folder(out_dir: Path) -> Iterator[Path]:
    """Yields <out_dir>.partial, empty, to write into; when the block ends without an error the folder replaces
    out_dir. A .partial left behind means an interrupted run."""
    tmp = out_dir.with_name(out_dir.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    yield tmp
    shutil.rmtree(out_dir, ignore_errors=True)
    tmp.rename(out_dir)


# ---------------------------------------------------------------- the run folder


@dataclass(frozen=True)
class RunFolder:
    """One run: results at the top, per-section intermediates in sections/. Assays use the parts they need."""

    root: Path

    @property
    def samples(self) -> Path:
        return self.root / "samples.csv"

    @property
    def checks(self) -> Path:
        return self.root / "checks.txt"

    @property
    def checks_json(self) -> Path:
        return self.root / "checks.json"

    @property
    def settings(self) -> Path:
        return self.root / "run_settings.json"

    @property
    def progress(self) -> Path:
        return self.root / "progress.json"

    @property
    def log(self) -> Path:
        return self.root / "run.log"

    @property
    def stop(self) -> Path:
        """Created by an app to ask the runner to stop after the current section."""
        return self.root / "stop_requested"

    @property
    def lock(self) -> Path:
        """Held by the runner while it works on this folder (hold_lock), so a second runner cannot start."""
        return self.root / "runner.lock"

    @property
    def app(self) -> Path:
        """What an app needs to restart the run."""
        return self.root / "app.json"

    @property
    def tasks(self) -> Path:
        """The task list: one line per section (necrosis: sample and scan, no header)."""
        return self.root / "tasks.tsv"

    @property
    def sections(self) -> Path:
        return self.root / "sections"

    def section(self, sample: str, variant: str | None = None) -> Path:
        """The folder of one section: sections/<sample>, or sections/<variant>/<sample> for an assay that keeps a
        folder per variant (psr)."""
        return self.sections / variant / sample if variant else self.sections / sample

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def figures(self) -> Path:
        return self.root / "figures"

    @property
    def tables(self) -> Path:
        return self.root / "tables"
