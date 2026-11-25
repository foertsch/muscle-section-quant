"""The data the psr overlay report and the click-review page embed, on the synthetic section and on a few written
windows: pinned, so the page code and the report base can change but not a pixel of the layers or a number."""

import json
from functools import partial
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from msq.core.segment import FixedLabels
from msq.core.synthetic.psr import synthetic_psr_section
from msq.psr import build_click_report as bcr
from msq.psr import click_windows
from msq.psr import overlay_report as orp
from msq.psr import section as sec
from tests.pins import assert_pinned, embedded

SEED = 13


class Reader:
    def close(self):
        pass


def overlay_page(root: Path) -> dict:
    """views and layers of the overlay report: the whole section and one window (the synthetic section is small,
    so a 40 um window with a 15 um margin), written to a page and read back from it."""
    s = synthetic_psr_section(SEED, "high")
    folder = sec.run_frame(s["frame"], s["px_um"], root / "sections" / "base" / "S1", sec.params_for("base"), "S1",
                           source="synthetic", segment=FixedLabels(s["labels_ds"]), variant="base")
    narrow = partial(click_windows.pick_window, edge_um=15.0)
    with (mock.patch.object(sec, "open_scan", lambda path: (Reader(), s["frame"], s["px_um"])),
          mock.patch.object(orp, "WINDOW_UM", 40.0), mock.patch.object(click_windows, "pick_window", narrow)):
        views = orp.section_views(folder, Path("scan.nd2"), "Control 1 sec 1", 1, np.random.default_rng(3))
    html = orp.render(views, root / "overlay.html", "Two <examples> & more", "Overlay check").read_text(encoding="utf-8")
    return html


def click_page(root: Path) -> str:
    wins = [{"id": "w01", "size_px": 8, "size_um": 1.7, "px_um": 0.2167, "n_outlines": 3},
            {"id": "w02", "size_px": 8, "size_um": 1.7, "px_um": 0.2167, "n_outlines": 5}]
    (root / "windows.json").write_text(json.dumps(wins))
    rng = np.random.default_rng(5)
    for w in wins:
        Image.fromarray(rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)).save(root / f"{w['id']}.jpg", quality=90)
        Image.fromarray(rng.integers(0, 255, (8, 8, 4), dtype=np.uint8)).save(root / f"{w['id']}_outlines.png")
    return bcr.build(root, root / "page.html", "test_run").read_text(encoding="utf-8")


OVERLAY = {
    "views": "23b741fbf56f",
    "layers": "1e92b976d44e"
}
CLICK = {
    "windows": "20e8074e900c",
    "images": "a4e0d59a9e85"
}


def test_overlay_report_data_is_pinned(tmp_path):
    data = embedded(overlay_page(tmp_path))
    assert [v["kind"] for v in data["views"]] == ["overview", "window"]
    assert_pinned(data, OVERLAY)


def test_click_review_data_is_pinned(tmp_path):
    data = embedded(click_page(tmp_path))
    assert data["run"] == "test_run" and [w["id"] for w in data["windows"]] == ["w01", "w02"]
    assert_pinned({k: v for k, v in data.items() if k != "run"}, CLICK)
