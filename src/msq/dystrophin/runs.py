"""Run folders as the app sees them: create and list them, read their state, start the runner on them, ask it to
stop, edit the sample table, and browse folders. No user interface here, so it can be tested; app.py draws the
pages.

A run's state comes from its progress.json (written by run_pipeline.py):
    new          created, nothing started yet
    starting / checking / measuring / summarizing / reports   the runner is working
    checked      the checks passed; the sample table is ready for review
    stopped      the checks found errors
    paused       stopped on request between sections
    done         finished
    failed       the runner stopped with an error
    interrupted  the runner was working but its process is gone: nobody holds its lock (runner.lock), and it has
                 exited or has not refreshed its heartbeat for STALE_S
"""

from __future__ import annotations

import json
import os
import re
import string
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from msq.core.io import lock_is_held, read_json
from msq.dystrophin import checks
from msq.dystrophin import manifest
from msq.dystrophin import paths

CONFIG = Path.home() / ".dystrophin_quant.json"
DEFAULT_RUNS = Path.home() / "Dystrophin runs"
ACTIVE = ("starting", "checking", "measuring", "summarizing", "reports")
STALE_S = 5 * 60  # the runner refreshes its heartbeat every minute
NAME_OK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._-]{0,79}$")


# ---------------------------------------------------------------- settings of the app


def load_config(path: Path | None = None) -> dict:
    path = path or CONFIG
    cfg = {"runs_folder": str(DEFAULT_RUNS), "last_scans": ""}
    try:
        cfg.update(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        pass
    return cfg


def save_config(cfg: dict, path: Path | None = None) -> None:
    path = path or CONFIG
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


# ---------------------------------------------------------------- runs


@dataclass
class Run:
    folder: Path
    progress: dict = field(default_factory=dict)
    settings: dict = field(default_factory=dict)
    app: dict = field(default_factory=dict)
    state: str = "new"

    @property
    def name(self) -> str:
        return self.folder.name

    @property
    def rf(self) -> paths.RunFolder:
        return paths.RunFolder(self.folder)

    @property
    def scans(self) -> Path:
        return Path(self.app.get("scans", self.settings.get("scans", "")))

    @property
    def reference(self) -> str:
        return self.app.get("reference") or ""

    @property
    def active(self) -> bool:
        return self.state in ACTIVE

    @property
    def updated(self) -> str:
        return self.progress.get("updated") or self.app.get("created", "")

    def problems(self) -> list[checks.Problem]:
        return [checks.Problem(**p) for p in read_json(self.rf.checks_json).get("problems", [])]

    def samples(self) -> pd.DataFrame:
        return checks.read_samples(self.rf.samples) if self.rf.samples.exists() else pd.DataFrame(
            columns=checks.COLUMNS)

    def reports(self) -> list[Path]:
        return sorted(self.rf.reports.glob("*.html"), key=lambda p: (p.name != "threshold_decision.html", p.name))


def seconds_since(stamp: str, now: datetime | None = None) -> float:
    try:
        return ((now or datetime.now()) - datetime.fromisoformat(stamp)).total_seconds()
    except (TypeError, ValueError):
        return float("inf")


def read_progress(path: Path) -> dict:
    """progress.json, read a second time when it exists but cannot be read (on Windows, while the runner
    replaces it), so the page does not flash "Not started"."""
    data = read_json(path)
    if not data and path.exists():
        time.sleep(0.05)
        data = read_json(path)
    return data


def read_run(folder: Path, exit_code: int | None = None, now: datetime | None = None) -> Run:
    """exit_code: the runner's exit status when the app started it and it has ended, else None.

    A held runner lock means the runner is alive, however old its heartbeat (after the laptop slept, or when a write
    failed). Without it, the run is interrupted when the runner's process has ended or its heartbeat is stale."""
    rf = paths.RunFolder(folder)
    run = Run(folder, read_progress(rf.progress), read_json(rf.settings), read_json(rf.app))
    state = run.progress.get("state", "new")
    if state in ACTIVE and not lock_is_held(rf.lock):
        beat = run.progress.get("heartbeat") or run.progress.get("updated")
        if exit_code is not None or seconds_since(beat, now) > STALE_S:
            state = "interrupted"
    run.state = state
    return run


def is_run(folder: Path) -> bool:
    rf = paths.RunFolder(folder)
    return folder.is_dir() and any(f.exists() for f in (rf.app, rf.progress, rf.samples))


def list_runs(root: Path, exit_codes: dict[Path, int | None] | None = None) -> list[Run]:
    """Run folders directly under root, most recently changed first."""
    exit_codes = exit_codes or {}
    if not root.is_dir():
        return []
    out = [read_run(d, exit_codes.get(d)) for d in root.iterdir() if is_run(d)]
    return sorted(out, key=lambda r: r.updated, reverse=True)


def busy_run(runs: list[Run], other_than: Path | None = None) -> Run | None:
    """A run that is measuring or summarizing (one at a time on the GPU); checks do not count."""
    for r in runs:
        if r.active and r.state != "checking" and r.folder != other_than:
            return r
    return None


def default_name(scans: Path, today: date | None = None) -> str:
    base = re.sub(r"[^A-Za-z0-9 ._-]", "", scans.name).strip() or "run"
    return f"{(today or date.today()).isoformat()} {base}"[:80]


def create_run(root: Path, name: str, scans: Path) -> Run:
    name = name.strip()
    if not NAME_OK.match(name):
        raise ValueError("Use letters, numbers, spaces, dots, dashes or underscores (at most 80), starting with a "
                         "letter or number.")
    folder = root / name
    if folder.exists():
        raise ValueError(f"A run called '{name}' already exists.")
    if not scans.is_dir():
        raise ValueError(f"The scan folder does not exist: {scans}")
    folder.mkdir(parents=True)
    write_app(folder, {"scans": str(scans.resolve()), "reference": "", "created": now_iso()})
    return read_run(folder)


def write_app(folder: Path, data: dict) -> None:
    paths.RunFolder(folder).app.write_text(json.dumps(data, indent=2), encoding="utf-8")


def set_reference(run: Run, reference: str) -> None:
    write_app(run.folder, {**run.app, "reference": reference})


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------- the runner


def runner_command(run: Run, check_only: bool) -> list[str]:
    cmd = [sys.executable, "-m", "msq.dystrophin.run_pipeline", str(run.scans), str(run.folder)]
    if run.reference:
        cmd += ["--reference", run.reference]
    if check_only:
        cmd.append("--check-only")
    return cmd


def start(run: Run, check_only: bool) -> subprocess.Popen:
    """Start the runner in its own process, so it keeps going when the app window closes. Its console output goes
    to console.log (run.log has the same, once the runner is up)."""
    rf = run.rf
    rf.stop.unlink(missing_ok=True)
    rf.progress.write_text(json.dumps({**run.progress, "state": "starting", "message": "Starting",
                                       "updated": now_iso(), "heartbeat": now_iso()}), encoding="utf-8")
    kwargs: dict = {}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    env = {**os.environ, "PYTHONUTF8": "1"}
    with (run.folder / "console.log").open("ab") as log:
        return subprocess.Popen(runner_command(run, check_only), cwd=run.folder, stdout=log,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env, **kwargs)


def request_stop(run: Run) -> None:
    run.rf.stop.write_text(now_iso(), encoding="utf-8")


def stop_requested(run: Run) -> bool:
    return run.rf.stop.exists()


# ---------------------------------------------------------------- the sample table


def apply_edits(samples: pd.DataFrame, edits: dict[str, dict]) -> pd.DataFrame:
    """edits: {sample: {"group"?, "animal"?, "include"?}}. Rows with include False are dropped; sample names are
    rebuilt from the edited values (manifest.sample_id)."""
    df = samples.copy()
    keep = []
    for i, r in df.iterrows():
        e = edits.get(r["sample"], {})
        for col in ("group", "animal"):
            if col in e:
                df.at[i, col] = str(e[col]).strip()
        keep.append(e.get("include", True))
    df = df[keep].copy()
    df["sample"] = [manifest.sample_id(r) for r in df.to_dict(orient="records")]
    return df.reset_index(drop=True)


def save_samples(run: Run, samples: pd.DataFrame) -> None:
    samples[checks.COLUMNS].to_csv(run.rf.samples, index=False)


def rebuild_samples(run: Run) -> None:
    """Forget the sample table, so the next check builds it again from the folder."""
    run.rf.samples.unlink(missing_ok=True)


def reference_candidates(runs: list[Run], other_than: Path) -> list[Run]:
    """Finished runs with a threshold to borrow."""
    return [r for r in runs if r.folder != other_than and (r.rf.tables / "thresholds.json").exists()]


# ---------------------------------------------------------------- small helpers for the pages


def fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "estimating"
    m = round(seconds / 60)
    if m < 1:
        return "under a minute"
    return f"{m} min" if m < 60 else f"{m // 60} h {m % 60:02d} min"


def tail(path: Path, n: int = 15) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-n:])


def windows_drives() -> list[Path]:
    """Drive roots, listed without touching them: probing a disconnected network drive can block for many seconds."""
    if hasattr(os, "listdrives"):  # Windows, Python 3.12 and later
        return [Path(d) for d in os.listdrives()]
    return [Path(f"{d}:\\") for d in string.ascii_uppercase if Path(f"{d}:\\").exists()]


def folder_roots() -> list[Path]:
    """Starting points for the folder browser: drives on Windows, home and mounted volumes elsewhere."""
    if os.name == "nt":
        return windows_drives()
    roots = [Path.home()]
    volumes = Path("/Volumes")
    if volumes.is_dir():
        roots += sorted(p for p in volumes.iterdir() if p.is_dir())
    return roots + [Path("/")]


def subfolders(path: Path) -> list[Path]:
    """Visible subfolders, sorted by name; empty if the folder cannot be read."""
    try:
        return sorted((p for p in path.iterdir() if p.is_dir() and not p.name.startswith(".")),
                      key=lambda p: p.name.lower())
    except OSError:
        return []


def count_scans(path: Path, limit: int = 2000) -> int:
    n = 0
    try:
        for _ in path.rglob("*.nd2"):
            n += 1
            if n >= limit:
                break
    except OSError:
        pass
    return n


def open_in_file_manager(path: Path) -> None:
    if os.name == "nt":
        os.startfile(path)  # noqa: S606 (local folder chosen in the app)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])
