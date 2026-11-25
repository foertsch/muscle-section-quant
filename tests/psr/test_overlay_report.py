
import numpy as np
import pytest

from msq.core.report import images
from msq.psr import overlay_report as orp
from msq.psr import section as sec
from tests.pins import embedded


def test_layers_split_red_by_outline_and_skip_outside_the_section():
    classes = np.array([[sec.RED, sec.RED, sec.YELLOW, sec.BG],
                        [sec.RED, sec.RED, sec.YELLOW, sec.RED]], np.uint8)
    labels = np.array([[0, 1, 1, 0],
                       [0, 1, 1, 0]], np.uint32)
    inside = np.array([[True, True, True, True],
                       [True, True, True, False]])        # bottom-right pixel is outside the main section
    m = orp.layer_masks(classes, labels, inside)
    assert m["counted"].sum() == 2 and m["inside"].sum() == 2 and m["unstained"].sum() == 1
    st = orp.shares(classes, labels, inside)
    assert st["collagen_outside"] == pytest.approx(2 / 6) and st["red_inside"] == pytest.approx(2 / 6)
    assert st["red_total"] == pytest.approx(4 / 6)


def test_render_fills_the_template(tmp_path):
    view = {"title": "Control 1 sec 1: whole section", "kind": "overview", "size_um": None, "base": "data:x",
            "layers": {n: "data:y" for n in orp.LAYERS}, "boxes": [],
            "stats": {"collagen_outside": 0.07, "red_inside": 0.03, "red_total": 0.1, "fibers": 10}}
    html = orp.render([view], tmp_path / "r.html").read_text()
    assert "/*DATA*/null" not in html and "<!--ASSET" not in html and "whole section" in html
    data = embedded(html)
    assert data["views"] == [view] and [l["id"] for l in data["layers"]] == list(orp.LAYERS)


def test_render_fills_the_subtitle(tmp_path):
    out = orp.render([], tmp_path / "r.html", "Control & Group A <examples>")
    html = out.read_text()
    assert "Control &amp; Group A &lt;examples&gt;" in html and "<!--SUBTITLE-->" not in html
    assert orp.SUBTITLE in orp.render([], tmp_path / "d.html").read_text().replace("&quot;", '"').replace("&#x27;", "'")


def test_the_overlay_pages_encode_jpeg_at_quality_86_without_optimising():
    rgb = np.random.default_rng(1).integers(0, 255, (24, 30, 3), dtype=np.uint8)
    assert orp.jpeg_uri(rgb) == images.jpeg_uri(rgb, 86, False) != images.jpeg_uri(rgb, 86, True)
    assert orp.jpeg_uri(rgb, 70) == images.jpeg_uri(rgb, 70, False)
