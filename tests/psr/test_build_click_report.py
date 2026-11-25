import json

import numpy as np
from PIL import Image

from msq.psr import build_click_report as bcr
from tests.pins import embedded


def test_page_embeds_every_window_and_leaves_no_placeholder(tmp_path):
    wins = [{"id": "w01", "size_px": 8, "size_um": 1.7, "px_um": 0.2167, "n_outlines": 3},
            {"id": "w02", "size_px": 8, "size_um": 1.7, "px_um": 0.2167, "n_outlines": 5}]
    (tmp_path / "windows.json").write_text(json.dumps(wins))
    for w in wins:
        Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(tmp_path / f"{w['id']}.jpg")
        Image.fromarray(np.zeros((8, 8, 4), np.uint8)).save(tmp_path / f"{w['id']}_outlines.png")
    out = bcr.build(tmp_path, tmp_path / "page.html", "test_run")
    html = out.read_text()
    assert html.count("data:image/jpeg;base64,") == 2 and html.count("data:image/png;base64,") == 2
    assert "/*DATA*/null" not in html and "<!--RUN-->" not in html and "<!--ASSET" not in html
    data = embedded(html)
    assert data["run"] == "test_run" and [w["n_outlines"] for w in data["windows"]] == [3, 5]
    assert set(data["images"]) == {"w01", "w02"} and "Run: test_run" in html
    assert html.startswith("<!doctype html>") and "claude.use" not in html
