"""msq.core.report: filling a template, images as data URIs, outlines, histogram bins, the representative section."""

import base64
import io

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from msq.core.report import hist, images, outlines, pick, render
from msq.core.report.render import ASSETS

# ---------------------------------------------------------------- render


def template(tmp_path, body: str):
    p = tmp_path / "t.html"
    p.write_text(body, encoding="utf-8")
    return p


def test_render_puts_the_data_on_one_compact_line(tmp_path):
    t = template(tmp_path, "<script>\nconst DATA = /*DATA*/null;\nrun(DATA);\n</script>")
    out = render(t, tmp_path / "sub" / "o.html", data={"a": [1, 2.5], "µ": "x y"})
    assert 'const DATA = {"a":[1,2.5],"\\u00b5":"x y"};\nrun(DATA);' in out.read_text(encoding="utf-8")


def test_render_without_data_leaves_the_marker_and_needs_exactly_one_marker(tmp_path):
    t = template(tmp_path, "const DATA = /*DATA*/null;\n")
    assert "/*DATA*/null" in render(t, tmp_path / "o.html").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one"):
        render(template(tmp_path, "no marker"), tmp_path / "o.html", data={})
    with pytest.raises(ValueError, match="exactly one"):
        render(template(tmp_path, "/*DATA*/null /*DATA*/null"), tmp_path / "o.html", data={})


def test_render_slots_are_escaped_everywhere_and_must_exist(tmp_path):
    t = template(tmp_path, "<title><!--TITLE--></title><h1><!--TITLE--></h1>")
    html = render(t, tmp_path / "o.html", slots={"TITLE": "A & B <c>"}).read_text(encoding="utf-8")
    assert html == "<title>A &amp; B &lt;c&gt;</title><h1>A &amp; B &lt;c&gt;</h1>"
    with pytest.raises(ValueError, match="no <!--OTHER-->"):
        render(t, tmp_path / "o.html", slots={"OTHER": "x"})


def test_render_inlines_the_shared_assets_and_extra_ones(tmp_path):
    t = template(tmp_path, "<!--ASSET base.css--><!--ASSET lightbox.css--><!--ASSET base.js--><!--ASSET mine.js-->")
    html = render(t, tmp_path / "o.html", assets={"mine.js": "let own = {};"}).read_text(encoding="utf-8")
    assert html.count("<style>") == 2 and html.count("<script>") == 2 and "<!--ASSET" not in html
    assert (ASSETS / "base.css").read_text(encoding="utf-8").strip() in html and "let own = {};" in html
    own = tmp_path / "own.css"
    own.write_text("p { color: red; }", encoding="utf-8")
    t = template(tmp_path, "<!--ASSET base.css-->")
    assert render(t, tmp_path / "o2.html", assets={"base.css": own}).read_text(encoding="utf-8") == \
        "<style>\np { color: red; }\n</style>"
    with pytest.raises(FileNotFoundError):
        render(template(tmp_path, "<!--ASSET nope.css-->"), tmp_path / "o3.html")


def test_render_does_not_interpret_data_or_asset_text(tmp_path):
    t = template(tmp_path, "<!--ASSET x.js-->\nconst DATA = /*DATA*/null;\n")
    html = render(t, tmp_path / "o.html", data={"s": "<!--ASSET base.js--> {0} \\1"}, assets={"x.js": "f({0}, '\\1');"})
    text = html.read_text(encoding="utf-8")
    assert "f({0}, '\\1');" in text and text.count("<style>") == 0 and "<script>\nf(" in text
    assert text.count("<!--ASSET base.js-->") == 1  # data stays data


# ---------------------------------------------------------------- images


def decode(uri: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1]))))


def test_png_is_lossless_and_jpeg_keeps_size():
    rgba = np.random.default_rng(0).integers(0, 255, (9, 7, 4), dtype=np.uint8)
    uri = images.png_uri(rgba)
    assert uri.startswith("data:image/png;base64,") and np.array_equal(decode(uri), rgba)
    rgb = np.random.default_rng(1).integers(0, 255, (16, 12, 3), dtype=np.uint8)
    for optimize in (True, False):
        uri = images.jpeg_uri(rgb, 90, optimize)
        assert uri.startswith("data:image/jpeg;base64,") and decode(uri).shape == rgb.shape


def test_data_uri_of_a_file(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"\x00\x01abc")
    assert images.data_uri(p, "application/x") == "data:application/x;base64," + base64.b64encode(b"\x00\x01abc").decode()


def test_rgba_paints_only_the_mask():
    out = images.rgba(np.array([[True, False]]), (1, 2, 3, 4))
    assert out[0, 0].tolist() == [1, 2, 3, 4] and out[0, 1].tolist() == [0, 0, 0, 0]


def test_to_uint8_scales_and_clips():
    assert images.to_uint8(np.array([-5.0, 0, 50, 100, 500]), 0, 100).tolist() == [0, 0, 127, 255, 255]


def test_glass_normalised_maps_glass_to_white_and_clips():
    glass = np.array([200.0, 100.0, 50.0], np.float32)
    rgb = np.array([[[200, 100, 50], [100, 50, 25], [255, 255, 255]]], np.uint8)
    assert images.glass_normalised(rgb, glass).tolist() == [[[255, 255, 255], [128, 128, 128], [255, 255, 255]]]


# ---------------------------------------------------------------- outlines


def square(n=20, lo=5, hi=15):
    m = np.zeros((n, n), bool)
    m[lo:hi, lo:hi] = True
    return m


def points(path: str) -> np.ndarray:
    assert path.startswith("M") and path.endswith("Z")
    return np.array([[float(v) for v in p.split(",")] for p in path[1:-1].split("L")])


def test_outline_path_traces_the_mask_and_empty_mask_has_none():
    xy = points(outlines.outline_path(square(), oy=100, ox=200))
    assert xy[:, 0].min() == pytest.approx(204.5, abs=0.6) and xy[:, 1].max() == pytest.approx(114.5, abs=0.6)
    assert outlines.outline_path(np.zeros((4, 4), bool), 0, 0) == ""


def test_centre_moves_half_a_pixel_and_scale_stretches_the_mask_grid():
    a = points(outlines.outline_path(square(), 0, 0, tol=0.7))
    b = points(outlines.outline_path(square(), 0, 0, tol=0.7, centre=True))
    assert np.allclose(b, a + 0.5, atol=0.051)
    c = points(outlines.outline_path(square(), 10, 20, scale=4, tol=0.7, centre=True))
    assert np.allclose(c[:, 0], 20 + (a[:, 0] + 0.5) * 4, atol=0.4)   # x gets the column offset, y the row offset
    assert np.allclose(c[:, 1], 10 + (a[:, 1] + 0.5) * 4, atol=0.4)


def test_label_paths_only_for_wanted_labels_and_at_the_label_offset():
    labels = np.zeros((10, 10), np.uint32)
    labels[1:4, 1:4], labels[5:9, 5:9] = 1, 2
    paths = outlines.label_paths(labels, np.array([2]), oy=100, ox=0, tol=0.7)
    assert set(paths) == {2} and points(paths[2])[:, 1].min() >= 100 + 5 - 1


def test_outline_paths_are_pinned_to_the_decimal():
    """The text the report pages carry: corners or pixel centres, the simplification, one decimal."""
    m = np.zeros((6, 6), bool)
    m[1:4, 2:5] = True
    assert outlines.outline_path(m, 10, 20) == "M24.0,13.5L21.5,13.0L22.0,10.5L24.5,11.0L24.0,13.5Z"
    assert outlines.outline_path(m, 10, 20, scale=4, tol=0.7, centre=True) == \
        "M38.0,26.0L28.0,24.0L30.0,14.0L40.0,16.0L38.0,26.0Z"
    holed = np.zeros((9, 10), bool)
    holed[1:8, 1:9] = True
    holed[3:6, 3:7] = False  # the longest contour is the outer one
    assert outlines.outline_path(holed, 0, 0) == "M8.0,7.5L1.0,7.5L0.5,1.0L8.0,0.5L8.0,7.5Z"
    labels = np.zeros((10, 10), np.uint32)
    labels[1:4, 1:4], labels[5:9, 5:9] = 1, 2
    paths = outlines.label_paths(labels, np.array([2, 1]), oy=3.0, ox=-5.0, scale=0.5, tol=0.7, centre=True)
    assert list(paths) == [1, 2]  # label order, not the order asked for
    assert paths == {1: "M-3.2,5.0L-4.5,4.8L-4.2,3.5L-3.0,3.8L-3.2,5.0Z",
                     2: "M-0.8,7.5L-2.5,7.2L-2.2,5.5L-0.5,5.8L-0.8,7.5Z"}


def test_jpeg_defaults_are_quality_88_optimised():
    rgb = np.random.default_rng(3).integers(0, 255, (20, 24, 3), dtype=np.uint8)
    assert images.jpeg_uri(rgb) == images.jpeg_uri(rgb, 88, True) != images.jpeg_uri(rgb, 88, False)


# ---------------------------------------------------------------- hist, pick


def test_shared_edges_and_hist_pct_on_the_core_names():
    e = hist.shared_edges([np.array([-30.0, 10, 60]), np.array([120.0, np.nan])], width=25, lo_pct=0, hi_pct=100)
    assert e[0] == -50 and e[-1] == 125 and np.allclose(np.diff(e), 25)
    h = hist.hist_pct([-1000, 0, 30, 5000, np.nan], np.arange(-50, 126, 25.0))
    assert h.sum() == pytest.approx(100) and h[0] == pytest.approx(25) and h[-1] == pytest.approx(25)
    assert hist.hist_pct([np.nan], e).sum() == 0


def test_representative_section_is_closest_to_the_mean_and_the_first_on_a_tie():
    assert pick.representative_section(pd.Series({"a_0": 0.1, "a_1": 0.5, "a_2": 0.35})) == "a_2"  # mean 0.3167
    assert pick.representative_section(pd.Series({"s1": 0.25, "s2": 0.75})) == "s1"                # both 0.25 from 0.5
    assert pick.representative_section(pd.Series({"only": 0.9})) == "only"
