"""The app: run the pipeline without a terminal. Opens in the browser on this computer only (127.0.0.1).

Pages:
    /             runs folder and the list of runs
    /new          pick a scan folder and name the run; the scans are checked
    /run/<name>   one run: review the sample table and the checks, then progress, then results

Every run is a folder under the runs folder, written by run_pipeline.py, which the app starts in its own process:
closing the window does not stop a run, and reopening the app shows where it is.

Usage:
    uv run msq-dystrophin-app                 # opens a browser tab
    uv run msq-dystrophin-app --no-browser    # serve only, at http://127.0.0.1:8080
"""

from __future__ import annotations

import argparse
import subprocess
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import quote

from fastapi import HTTPException
from fastapi.responses import FileResponse
from nicegui import app, background_tasks, run, ui

from msq.core import runinfo
from msq.core.io import lock_is_held, read_json
from msq.dystrophin import checks, runs, settings

TITLE = "Dystrophin quantification"
BLUE = "#2a78d6"
STATES = {  # state: (label, chip colour, icon)
    "new": ("Not started", "grey-6", "radio_button_unchecked"),
    "starting": ("Starting", "primary", "hourglass_top"),
    "checking": ("Checking the scans", "primary", "hourglass_top"),
    "checked": ("Ready to run", "teal-7", "task_alt"),
    "stopped": ("Needs fixing", "red-7", "error"),
    "measuring": ("Measuring", "primary", "hourglass_top"),
    "summarizing": ("Calling fibers", "primary", "hourglass_top"),
    "reports": ("Writing reports", "primary", "hourglass_top"),
    "paused": ("Paused", "amber-9", "pause_circle"),
    "done": ("Done", "green-8", "check_circle"),
    "failed": ("Failed", "red-7", "error"),
    "interrupted": ("Interrupted", "red-7", "error"),
}
LEVELS = {"error": ("Error", "error", "text-red-8"), "warning": ("Warning", "warning", "text-amber-10"),
          "note": ("Note", "info", "text-grey-7")}
SECTION_ICONS = {"waiting": ("schedule", "text-grey-5"), "running": ("autorenew", "text-primary"),
                 "done": ("check_circle", "text-green-8"), "failed": ("error", "text-red-8")}
REPORT_NAMES = {"threshold_decision.html": "Threshold decision report"}

PROCS: dict[Path, subprocess.Popen] = {}  # runner processes started by this app
VERSION = runinfo.code_version()


# ---------------------------------------------------------------- shared


def runs_root() -> Path:
    return Path(runs.load_config()["runs_folder"])


def exit_codes() -> dict[Path, int | None]:
    """Exit status of the runners this app started that have ended; None for the ones still going."""
    return {f: p.poll() for f, p in PROCS.items()}


def current(folder: Path) -> runs.Run:
    return runs.read_run(folder, exit_codes().get(folder))


def all_runs() -> list[runs.Run]:
    return runs.list_runs(runs_root(), exit_codes())


def start_runner(run_: runs.Run, check_only: bool) -> bool:
    """Start the runner on the run as it is on disk now (the page may hold an older copy). Never a second runner on
    the same folder: a page can show "Interrupted" for a runner that is alive (its heartbeat is stale after sleep)."""
    run_ = current(run_.folder)
    proc = PROCS.get(run_.folder)
    if lock_is_held(run_.rf.lock) or (proc is not None and proc.poll() is None):
        ui.notify(f"The runner of '{run_.name}' is still working. Wait until it has finished.", type="warning",
                  multi_line=True)
        return False
    if not check_only:
        busy = runs.busy_run(all_runs(), other_than=run_.folder)
        if busy:
            ui.notify(f"'{busy.name}' is running. One run at a time uses the GPU; start this one when it is done.",
                      type="warning", multi_line=True)
            return False
    PROCS[run_.folder] = runs.start(run_, check_only)
    return True


def file_url(run_: runs.Run, rel: str) -> str:
    return f"/files/{quote(run_.name)}/{quote(rel)}"


@app.get("/files/{run_name}/{rel:path}")
def run_file(run_name: str, rel: str):
    root = runs_root().resolve()
    f = (root / run_name / rel).resolve()
    if not f.is_relative_to(root) or not f.is_file():
        raise HTTPException(status_code=404)
    return FileResponse(f)


@contextmanager
def frame(subtitle: str = ""):
    ui.colors(primary=BLUE)
    ui.query("body").style("background-color: #f6f6f3")
    with ui.header().classes("bg-white text-grey-10 shadow-1"):
        with ui.row().classes("w-full max-w-5xl mx-auto items-center no-wrap"):
            ui.icon("biotech", size="sm").classes("text-primary")
            ui.link(TITLE, "/").classes("text-lg font-medium text-grey-10 no-underline")
            ui.label(VERSION).classes("text-xs text-grey-6")
            ui.space()
            ui.label(subtitle).classes("text-grey-7")
    with ui.column().classes("w-full max-w-5xl mx-auto px-4 py-6 gap-5"):
        yield


def state_chip(state: str) -> None:
    label, color, icon = STATES.get(state, (state, "grey-6", "help"))
    ui.chip(label, icon=icon, color=color, text_color="white").props("dense square")


def problem_list(problems: list[checks.Problem]) -> None:
    for p in problems:
        word, icon, cls = LEVELS.get(p.level, LEVELS["note"])
        with ui.row().classes("items-start no-wrap gap-2"):
            ui.icon(icon).classes(f"{cls} mt-0.5")
            with ui.column().classes("gap-0"):
                ui.label(f"{word}{': ' + p.where if p.where else ''}").classes(f"text-sm font-medium {cls}")
                ui.label(p.message).classes("text-sm text-grey-9")


def log_box(run_: runs.Run, lines: int = 14) -> None:
    text = runs.tail(run_.rf.log, lines) or runs.tail(run_.folder / "console.log", lines) or "(nothing yet)"
    with ui.expansion("Details (log)", icon="notes").classes("w-full bg-white"):
        ui.label(text).classes("w-full font-mono text-xs whitespace-pre-wrap break-all bg-grey-2 p-2 rounded")


def folder_picker(start: str, on_pick, title: str = "Choose a folder") -> ui.dialog:
    """A folder browser that runs on this computer, so it shows the same drives and network shares the runner
    sees. Paths can also be typed or pasted."""
    start_path = Path(start) if start and Path(start).is_dir() else Path.home()
    state = {"path": start_path}
    with ui.dialog() as dialog, ui.card().classes("w-[44rem] max-w-full"):
        ui.label(title).classes("text-lg font-medium")
        path_in = ui.input("Folder", value=str(start_path)).classes("w-full").props("outlined dense")
        with ui.row().classes("gap-1"):
            for root in runs.folder_roots():
                ui.button(str(root) if root != Path.home() else "Home", icon="home" if root == Path.home() else
                          "storage", on_click=lambda r=root: show(r)).props("flat dense no-caps")
        listing = ui.list().props("dense bordered separator").classes("w-full max-h-80 overflow-auto bg-white")

        def show(p: Path) -> None:
            if not p.is_dir():
                ui.notify(f"Not a folder: {p}", type="warning")
                return
            state["path"] = p
            path_in.value = str(p)
            listing.clear()
            with listing:
                if p.parent != p:
                    with ui.item(on_click=lambda: show(p.parent)):
                        with ui.item_section().props("avatar"):
                            ui.icon("arrow_upward")
                        ui.item_section("Up one level")
                subs = runs.subfolders(p)
                for sub in subs:
                    with ui.item(on_click=lambda s=sub: show(s)):
                        with ui.item_section().props("avatar"):
                            ui.icon("folder").classes("text-amber-8")
                        ui.item_section(sub.name)
                if not subs:
                    ui.item("No subfolders").classes("text-grey-6")

        path_in.on("keydown.enter", lambda: show(Path(path_in.value.strip().strip('"'))))
        with ui.row().classes("w-full justify-end"):
            ui.button("Cancel", on_click=dialog.close).props("flat no-caps")

            def pick() -> None:
                p = Path(path_in.value.strip().strip('"'))
                if not p.is_dir():
                    ui.notify(f"Not a folder: {p}", type="warning")
                    return
                dialog.close()
                on_pick(p)

            ui.button("Use this folder", icon="check", on_click=pick).props("no-caps")
    show(start_path)
    return dialog


# ---------------------------------------------------------------- home


def list_key(root: Path, found: list[runs.Run]) -> tuple:
    """What the run list shows that can change: the runs folder, and each run's name, state and last change."""
    return str(root), tuple((r.name, r.state, r.updated) for r in found)


@ui.page("/")
def home() -> None:
    with frame():
        with ui.row().classes("w-full items-center"):
            with ui.column().classes("gap-0"):
                ui.label("Runs").classes("text-2xl font-medium")
                root_label = ui.label().classes("text-sm text-grey-7")
            ui.space()
            ui.button("Change runs folder", icon="folder_open", on_click=lambda: change_root()).props("flat no-caps")
            ui.button("New run", icon="add", on_click=lambda: ui.navigate.to("/new")).props("no-caps")

        def change_root() -> None:
            def picked(p: Path) -> None:
                cfg = runs.load_config()
                cfg["runs_folder"] = str(p)
                runs.save_config(cfg)
                run_list.refresh()

            folder_picker(str(runs_root()), picked, "Folder that holds the runs").open()

        seen = {"key": None}

        @ui.refreshable
        def run_list() -> None:
            root = runs_root()
            root_label.text = f"Runs folder: {root}"
            found = all_runs()
            seen["key"] = list_key(root, found)
            busy = runs.busy_run(found)
            if busy:
                with ui.card().classes("w-full bg-blue-1"):
                    ui.label(f"'{busy.name}' is running: {busy.progress.get('message', '')}").classes("text-sm")
            if not found:
                with ui.card().classes("w-full items-center py-10"):
                    ui.icon("inbox", size="lg").classes("text-grey-5")
                    ui.label("No runs yet. Start with New run.").classes("text-grey-7")
                return
            with ui.list().props("bordered separator").classes("w-full bg-white rounded"):
                for r in found:
                    with ui.item(on_click=lambda r=r: ui.navigate.to(f"/run/{quote(r.name)}")):
                        with ui.item_section():
                            ui.item_label(r.name).classes("font-medium")
                            n = r.progress.get("n_sections")
                            detail = f"{n} section{'s' * (n != 1)} · " if n else ""
                            ui.item_label(f"{detail}last change {r.updated.replace('T', ' ')}").props("caption")
                        with ui.item_section().props("side"):
                            state_chip(r.state)

        def poll() -> None:  # rebuild the list only when a run changed, as the run page does
            if list_key(runs_root(), all_runs()) != seen["key"]:
                run_list.refresh()

        run_list()
        ui.timer(5.0, poll)


# ---------------------------------------------------------------- new run


@ui.page("/new")
async def new_run() -> None:
    cfg = runs.load_config()
    with frame("New run"):
        ui.label("New run").classes("text-2xl font-medium")
        with ui.card().classes("w-full"):
            ui.label("1. The folder with the scans").classes("text-base font-medium")
            ui.markdown(
                "Every `.nd2` file in the folder and its subfolders is used. **Controls** go in a folder whose name "
                "starts with `Neg-ctrl`. File names need the muscle (`QUAD` or `TA`), the section (`sec 2`), the "
                f"4-digit animal number and `{settings.MAGNIFICATION}`, for example `0001 1001 L TA sec 2 20x.nd2`. "
                "You can correct groups and animals in the next step.").classes("text-sm text-grey-8")
            with ui.row().classes("w-full items-center no-wrap"):
                scans_in = ui.input("Scan folder", value=cfg.get("last_scans", "")).classes("grow").props(
                    "outlined dense")
                ui.button("Browse", icon="folder_open",
                          on_click=lambda: folder_picker(scans_in.value, set_scans, "Folder with the scans").open()
                          ).props("outline no-caps")
            found = ui.label().classes("text-sm text-grey-7")

            ui.label("2. A name for the run").classes("text-base font-medium mt-4")
            name_in = ui.input("Run name").classes("w-full").props("outlined dense")
            ui.label(f"The run is saved in {runs_root()}").classes("text-sm text-grey-7")

            check_btn = ui.button("Check the scans", icon="fact_check").props("no-caps").classes("mt-2")
        auto = {"name": ""}

        async def update_count() -> None:
            p = Path(scans_in.value.strip().strip('"'))
            if not scans_in.value.strip() or not p.is_dir():
                found.text = "Choose a folder." if not scans_in.value.strip() else "This folder does not exist."
                return
            default = runs.default_name(p)
            if not name_in.value or name_in.value == auto["name"]:  # follow the folder unless the user typed a name
                name_in.value = default
            auto["name"] = default
            found.text = "Looking for scans…"
            n = await run.io_bound(runs.count_scans, p)
            found.text = f"{n} .nd2 file{'s' * (n != 1)} found" + ("" if n else "; check the folder")

        def set_scans(p: Path) -> None:
            scans_in.value = str(p)
            background_tasks.create(update_count())

        scans_in.on("blur", update_count)
        scans_in.on("keydown.enter", update_count)

        def check() -> None:
            p = Path(scans_in.value.strip().strip('"'))
            try:
                r = runs.create_run(runs_root(), name_in.value, p)
            except ValueError as e:
                ui.notify(str(e), type="negative", multi_line=True)
                return
            cfg = runs.load_config()
            cfg["last_scans"] = str(p)
            runs.save_config(cfg)
            start_runner(r, check_only=True)
            ui.navigate.to(f"/run/{quote(r.name)}")

        check_btn.on_click(check)
        if scans_in.value:
            await update_count()


# ---------------------------------------------------------------- one run


@ui.page("/run/{name}")
def run_page(name: str) -> None:
    folder = runs_root() / name
    with frame(name):
        if not runs.is_run(folder):
            ui.label(f"There is no run called '{name}' in {runs_root()}.")
            ui.link("Back to the runs", "/")
            return
        seen = {"key": None}

        @ui.refreshable
        def body() -> None:
            r = current(folder)
            seen["key"] = (r.state, r.progress.get("updated"))
            with ui.row().classes("w-full items-center"):
                with ui.column().classes("gap-0"):
                    ui.label(r.name).classes("text-2xl font-medium")
                    ui.label(f"Scans: {r.scans}").classes("text-sm text-grey-7")
                ui.space()
                state_chip(r.state)
            if r.state in ("new", "starting", "checking"):
                waiting(r)
            elif r.state in ("checked", "stopped"):
                review(r, body)
            elif r.active:
                progress(r, body)
            elif r.state == "done":
                results(r)
            else:
                stopped(r, body)

        def poll() -> None:
            r = current(folder)
            if (r.state, r.progress.get("updated")) != seen["key"]:
                body.refresh()

        body()
        ui.timer(2.0, poll)


def waiting(r: runs.Run) -> None:
    with ui.card().classes("w-full items-center py-8"):
        ui.spinner(size="lg")
        ui.label(r.progress.get("message") or "Starting").classes("text-grey-8")
        ui.label("Reading the folder and opening every scan; this takes a few seconds per scan.").classes(
            "text-sm text-grey-6")


def wants_borrow(r: runs.Run, errors: list[checks.Problem]) -> bool:
    """Offer to borrow a threshold when a muscle has no controls, or when the run already borrows one."""
    return any(p.code == checks.NO_CONTROLS for p in errors) or bool(r.reference)


def review(r: runs.Run, body) -> None:
    problems = r.problems()
    errors = checks.errors(problems)
    samples = r.samples()
    edits: dict[str, dict] = {}

    with ui.card().classes("w-full"):
        if errors:
            ui.label(f"{len(errors)} problem{'s' * (len(errors) != 1)} must be fixed before the run can start."
                     ).classes("text-base font-medium text-red-8")
        else:
            ui.label(f"{len(samples)} section{'s' * (len(samples) != 1)} found. The checks passed; look over the "
                     "table, then start the run."
                     ).classes("text-base font-medium")
        problem_list(errors + [p for p in problems if p.level != "error"])
        if wants_borrow(r, errors):
            borrow(r)

    with ui.card().classes("w-full"):
        ui.label("Sections").classes("text-base font-medium")
        ui.label("Change a group or an animal number if the folder or file name got it wrong, or untick a section to "
                 "leave it out. Then save and check again.").classes("text-sm text-grey-7")
        dirty = ui.label("").classes("text-sm text-amber-10")

        def edit(sample: str, col: str, value) -> None:
            edits.setdefault(sample, {})[col] = value
            dirty.text = "Unsaved changes"
            start_btn.disable()

        for muscle in [m for m in settings.MUSCLES if m in set(samples["muscle"])] + sorted(
                set(samples["muscle"]) - set(settings.MUSCLES)):
            ui.label(muscle).classes("text-sm font-medium text-grey-8 mt-2")
            with ui.grid(columns="auto minmax(10rem,1fr) 7rem 9rem minmax(8rem,2fr)").classes(
                    "w-full items-center gap-x-3 gap-y-1"):
                for h in ("Use", "Section", "Animal", "Group", "File"):
                    ui.label(h).classes("text-xs text-grey-6")
                for row in samples[samples["muscle"] == muscle].sort_values(["group", "animal", "section"]).itertuples():
                    s = row.sample
                    ui.checkbox(value=True, on_change=lambda e, s=s: edit(s, "include", e.value)).props("dense")
                    ui.label(f"sec {row.section}").classes("text-sm")
                    ui.input(value=str(row.animal), on_change=lambda e, s=s: edit(s, "animal", e.value)).props(
                        "dense outlined").classes("text-sm")
                    ui.select(settings.GROUP_NAMES, value=row.group if row.group in settings.GROUP_NAMES else None,
                              on_change=lambda e, s=s: edit(s, "group", e.value)).props("dense outlined")
                    with ui.label(Path(row.path).name).classes("text-xs text-grey-7 truncate"):
                        ui.tooltip(row.path)

    def save_and_check() -> None:
        if edits:
            runs.save_samples(r, runs.apply_edits(samples, edits))
        if start_runner(r, check_only=True):
            body.refresh()

    def rebuild() -> None:
        runs.rebuild_samples(r)
        if start_runner(r, check_only=True):
            body.refresh()

    def start_run() -> None:
        if start_runner(r, check_only=False):
            body.refresh()

    with ui.row().classes("w-full items-center"):
        ui.button("Save and check again", icon="fact_check", on_click=save_and_check).props("outline no-caps")
        ui.button("Rebuild the table from the folder", icon="restart_alt", on_click=rebuild).props("flat no-caps")
        ui.space()
        start_btn = ui.button("Start the run", icon="play_arrow", on_click=start_run).props("no-caps size=lg")
        if errors:
            start_btn.disable()
    ui.label("The run measures every section (about 1 to 2.5 min each), then calls the fibers and writes the "
             "reports. You can close this window while it runs.").classes("text-sm text-grey-7")
    log_box(r)


def borrow(r: runs.Run) -> None:
    cands = runs.reference_candidates(all_runs(), r.folder)
    ui.separator()
    ui.label("No controls for a muscle? You can borrow the threshold from an earlier run instead.").classes(
        "text-sm font-medium")
    ui.label("Only do this if staining and imaging were identical: same antibody lot, exposure, laser power and "
             "gain. The results will say where the threshold came from.").classes("text-sm text-grey-7")
    options = {"": "Don't borrow"}
    for c in cands:
        th = settings.current_keys(read_json(c.rf.tables / "thresholds.json")).get("muscles", {})
        options[str(c.folder)] = f"{c.name} ({', '.join(f'{m} {v['threshold']:.0f}' for m, v in th.items())})"
    if r.reference and r.reference not in options:
        options[r.reference] = r.reference
    if len(options) == 1:
        ui.label("There is no finished run to borrow from yet.").classes("text-sm text-grey-6")
        return

    def chose(e) -> None:
        runs.set_reference(r, e.value or "")
        ui.notify("Saved. Check again to use it.", type="info")

    ui.select(options, value=r.reference or "", label="Borrow the threshold from", on_change=chose).classes(
        "w-full").props("outlined dense")


def progress(r: runs.Run, body) -> None:
    p = r.progress
    n, done, failed = p.get("n_sections", 0), p.get("n_done", 0), p.get("n_failed", 0)
    with ui.card().classes("w-full"):
        ui.label(p.get("message", "")).classes("text-base font-medium")
        ui.linear_progress(value=(done + failed) / n if n else 0, show_value=False).props("rounded size=12px")
        parts = [f"{done} of {n} sections measured"]
        if failed:
            parts.append(f"{failed} failed")
        if r.state == "measuring":
            eta = p.get("eta_s")
            parts.append(f"about {runs.fmt_duration(eta)} left" if eta is not None else "estimating the time left")
        ui.label(" · ".join(parts)).classes("text-sm text-grey-8")
        with ui.grid(columns="repeat(auto-fill, minmax(15rem, 1fr))").classes("w-full gap-1 mt-2"):
            for s, v in p.get("sections", {}).items():
                icon, cls = SECTION_ICONS.get(v["status"], SECTION_ICONS["waiting"])
                with ui.row().classes("items-center no-wrap gap-1"):
                    ui.icon(icon, size="xs").classes(cls)
                    lab = ui.label(s).classes("text-xs truncate")
                    if v.get("error"):
                        with lab:
                            ui.tooltip(v["error"])
        with ui.row().classes("w-full items-center mt-2"):
            ui.label("You can close this window; the run keeps going.").classes("text-sm text-grey-7")
            ui.space()
            if runs.stop_requested(r):
                ui.label("Stopping after the current section…").classes("text-sm text-amber-10")
            elif r.state == "measuring":
                def stop() -> None:
                    runs.request_stop(r)
                    ui.notify("The run stops after the current section. Finished sections are kept.")
                    body.refresh()

                ui.button("Stop after this section", icon="pause", on_click=stop).props("outline no-caps")
    log_box(r)


def stopped(r: runs.Run, body) -> None:
    texts = {
        "paused": "The run was stopped on request. Finished sections are kept; continue to measure the rest.",
        "interrupted": "The run stopped unexpectedly (the program was closed or the computer restarted). "
                       "Finished sections are kept; continue to measure the rest.",
        "failed": "The run stopped with an error. The details below say why; finished sections are kept.",
    }
    with ui.card().classes("w-full"):
        ui.label(texts.get(r.state, "")).classes("text-base")
        if r.progress.get("message"):
            ui.label(r.progress["message"]).classes("text-sm text-grey-8")

        def again() -> None:
            if start_runner(r, check_only=False):
                body.refresh()

        ui.button("Continue the run", icon="play_arrow", on_click=again).props("no-caps")
    log_box(r, 25)


def results(r: runs.Run) -> None:
    s = r.settings
    thresholds, rows = s.get("thresholds", {}), s.get("results", [])
    level = round(100 * settings.CALL_LEVEL)
    with ui.card().classes("w-full"):
        ui.label("% of fibers positive").classes("text-base font-medium")
        ui.label(f"A fiber is positive when at least {level}% of its ring is above the threshold. The threshold is the "
                 f"{round(100 * (1 - settings.FPR))}th percentile of the control fibers of that muscle (in this run, "
                 "or in the run it was borrowed from). Treated animals are called against it; each control animal "
                 "against the threshold from the other control animals, so its value is what an unseen control animal "
                 "would give. Values are the mean of each animal's sections.").classes("text-sm text-grey-7")
        for m in [m for m in settings.MUSCLES if m in thresholds] + sorted(set(thresholds) - set(settings.MUSCLES)):
            ui.label(f"{m}: threshold {thresholds[m]['threshold']:.0f} ({thresholds[m]['source']})").classes(
                "text-sm font-medium mt-3")
            table_rows = [{"animal": x["animal"], "group": settings.GROUP_NAMES.get(x["group"], x["group"]),
                           "pct": f"{x[settings.CALL_COLUMN]:.1f}%", "sections": x["n_sections"],
                           "fibers": f"{x['n_called']:,}"}
                          for x in sorted((x for x in rows if x["muscle"] == m), key=lambda x: (x["group"], x["animal"]))]
            ui.table(columns=[
                {"name": "animal", "label": "Animal", "field": "animal", "align": "left"},
                {"name": "group", "label": "Group", "field": "group", "align": "left"},
                {"name": "pct", "label": "Positive", "field": "pct", "align": "right"},
                {"name": "sections", "label": "Sections", "field": "sections", "align": "right"},
                {"name": "fibers", "label": "Fibers called", "field": "fibers", "align": "right"},
            ], rows=table_rows, row_key="animal").props("flat dense bordered").classes("w-full")

    problems = [p for p in r.problems() if p.level != "note"]
    if problems:
        with ui.card().classes("w-full"):
            ui.label("Worth knowing").classes("text-base font-medium")
            problem_list(problems)

    with ui.card().classes("w-full"):
        ui.label("Reports and files").classes("text-base font-medium")
        with ui.row().classes("gap-2"):
            for rep in r.reports():
                label = REPORT_NAMES.get(rep.name) or f"{rep.stem.split('_')[0].upper()}: controls vs treated"
                ui.button(label, icon="open_in_new",
                          on_click=lambda rep=rep: ui.navigate.to(file_url(r, f"reports/{rep.name}"), new_tab=True)
                          ).props("outline no-caps")
            ui.button("Open the run folder", icon="folder_open",
                      on_click=lambda: runs.open_in_file_manager(r.folder)).props("flat no-caps")
        fig = r.rf.figures / "pct_positive_call.png"
        if fig.exists():
            ui.image(file_url(r, "figures/pct_positive_call.png") + f"?v={int(fig.stat().st_mtime)}").classes(
                "w-full max-w-3xl mt-2")
        ui.label(f"Finished {s.get('finished', '').replace('T', ' ')} · GPU: {s.get('device', '?')} · code "
                 f"{s.get('code_version', '?')}").classes("text-xs text-grey-6")


# ---------------------------------------------------------------- start


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    ap.add_argument("--native", action="store_true", help="own window instead of a browser tab (needs pywebview)")
    ap.add_argument("--config", type=Path, help=f"settings file of the app (default {runs.CONFIG})")
    args = ap.parse_args()
    if args.config:
        runs.CONFIG = args.config
    runs_root().mkdir(parents=True, exist_ok=True)
    ui.run(host="127.0.0.1", port=args.port, title=TITLE, reload=False, show=not args.no_browser,
           native=args.native, favicon="🔬")


if __name__ in {"__main__", "__mp_main__"}:
    main()
