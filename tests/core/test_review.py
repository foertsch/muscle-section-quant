import numpy as np
import pandas as pd
import pytest

from msq.core import review


def squares(a_cols, b_cols):
    """Two 10 x 10 label images, each with one 4 x 4 object at the columns given and a second one identical in both."""
    a, b = np.zeros((10, 10), np.uint32), np.zeros((10, 10), np.uint32)
    a[0:4, a_cols], b[0:4, b_cols] = 1, 1
    a[6:10, 6:10] = b[6:10, 6:10] = 2
    return a, b


def test_iou_pairs_lists_the_overlaps_with_iou_and_cover():
    a, b = squares(slice(0, 4), slice(2, 6))  # half of each object shared
    pairs = review.iou_pairs(a, b)
    assert pairs.to_dict("list") == {"a": [1, 2], "b": [1, 2], "inter": [8, 16], "iou": [8 / 24, 1.0],
                                     "cover_a": [0.5, 1.0]}


def test_iou_pairs_are_sorted_and_count_an_object_met_twice():
    a = np.zeros((4, 8), np.uint32)
    a[:, :4], a[:, 4:] = 2, 1
    b = np.zeros_like(a)
    b[:, :6], b[:, 6:] = 7, 3  # b's 7 covers all of a's 2 and a third of a's 1
    pairs = review.iou_pairs(a, b)
    assert pairs[["a", "b"]].values.tolist() == [[1, 3], [1, 7], [2, 7]]
    assert pairs["inter"].tolist() == [8, 8, 16] and pairs["cover_a"].tolist() == [0.5, 0.5, 1.0]


def test_iou_pairs_of_images_with_nothing_in_common_is_empty():
    a, b = np.zeros((5, 5), np.uint32), np.zeros((5, 5), np.uint32)
    a[:2], b[3:] = 1, 4
    pairs = review.iou_pairs(a, b)
    assert pairs.empty and list(pairs.columns) == ["a", "b", "inter", "iou", "cover_a"]
    assert review.best_iou(pairs, "a", [1, 2]).tolist() == [0.0, 0.0]


def test_best_iou_takes_the_best_partner_and_zero_for_the_unmatched():
    a, b = squares(slice(0, 4), slice(1, 4))  # a's 1 shares 12 of 16 px, b's 1 has 12
    pairs = review.iou_pairs(a, b)
    assert review.best_iou(pairs, "a", [1, 2, 3]).tolist() == [12 / 16, 1.0, 0.0]
    assert review.best_iou(pairs, "b", range(1, 3)).tolist() == [12 / 16, 1.0]


def test_draw_outlines_marks_only_the_edges_in_their_colours():
    lab = np.zeros((12, 12), np.uint32)
    lab[2:8, 2:8] = 1
    lab[8:12, 6:12] = 2
    out = review.draw_outlines(np.zeros((12, 12, 3), np.uint8), lab, {1: (9, 8, 7)}, default=(1, 2, 3))
    assert out[2, 2].tolist() == [9, 8, 7] and out[10, 8].tolist() == [0, 0, 0]
    assert out[11, 8].tolist() == [0, 0, 0]  # the image border is no edge
    assert out[8, 8].tolist() == [1, 2, 3]  # another object, no colour given: the default
    assert out[4, 4].tolist() == [0, 0, 0] and out[0, 0].tolist() == [0, 0, 0]  # inside, and outside the objects


def test_draw_outlines_has_a_grey_default_and_leaves_the_input_alone():
    lab = np.zeros((6, 6), np.uint32)
    lab[1:5, 1:5] = 1
    rgb = np.zeros((6, 6, 3), np.uint8)
    out = review.draw_outlines(rgb, lab)
    assert out[1, 1].tolist() == list(review.GREY) and rgb.sum() == 0


def test_draw_outlines_width_and_highlight():
    lab = np.zeros((14, 14), np.uint32)
    lab[3:11, 3:11] = 1
    rgb = np.zeros((14, 14, 3), np.uint8)
    thin, thick = (review.draw_outlines(rgb, lab, None, (255, 255, 255), w) for w in (1, 2))
    assert thin[4, 5].tolist() == [0, 0, 0] and thick[4, 5].tolist() == [255, 255, 255]
    marked = review.draw_outlines(rgb, lab, None, (255, 255, 255), 1, highlight=1)
    assert marked[3, 5].tolist() == list(review.HIGHLIGHT)


class Reader:
    def __init__(self, log, name):
        self.log, self.name = log, name

    def close(self):
        self.log.append(self.name)


def test_open_scans_gives_the_frames_and_closes_every_reader():
    log = []
    opened = lambda path: (Reader(log, path), f"frame {path}", "names")  # noqa: E731
    with review.open_scans({"a": "x", "b": "y"}, opened) as scans:
        assert scans == {"a": ("frame x", "names"), "b": ("frame y", "names")} and log == []
    assert log == ["x", "y"] and scans == {}  # closed, and the frames let go first


def test_open_scans_closes_when_the_pictures_fail():
    log = []
    with pytest.raises(RuntimeError):
        with review.open_scans({"a": "x"}, lambda path: (Reader(log, path), None)):
            raise RuntimeError("boom")
    assert log == ["x"]


def make_review(calls, summary=None, pairs=None, **kw):
    return review.Review(
        "The doc.", lambda run, args: calls.append(("load", run, args.seed)) or "loaded",
        lambda loaded: [{"sample": "S1", "n": 3, "x": 1.5}] if summary is None else summary,
        lambda loaded: [{"sample": "S1", "of": "a", "by": "b"}] if pairs is None else pairs,
        lambda loaded, args: calls.append(("render", loaded, args.out)),
        lambda ap: ap.add_argument("--seed", type=int, default=1), **kw)


def test_main_writes_the_tables_then_the_pictures_and_prints_them(tmp_path, capsys):
    calls = []
    review.main(make_review(calls), [str(tmp_path / "run"), "--out", str(tmp_path / "out" / "r"), "--seed", "4"])
    out = tmp_path / "out" / "r"
    assert calls == [("load", tmp_path / "run", 4), ("render", "loaded", out)]
    assert pd.read_csv(out / "summary.csv").to_dict("list") == {"sample": ["S1"], "n": [3], "x": [1.5]}
    assert pd.read_csv(out / "agreement.csv").to_dict("list") == {"sample": ["S1"], "of": ["a"], "by": ["b"]}
    printed = capsys.readouterr().out
    assert printed.split() == ["sample", "n", "x", "S1", "3", "1.5", "sample", "of", "by", "S1", "a", "b"]


def test_main_prints_a_wide_summary_one_column_per_row_and_no_empty_agreement(tmp_path, capsys):
    review.main(make_review([], wide=True, pairs=[]), [str(tmp_path), "--out", str(tmp_path / "o")])
    printed = capsys.readouterr().out
    assert printed.splitlines()[0].split() == ["0"] and any(line.split() == ["n", "3"] for line in printed.splitlines())
    assert "by" not in printed and (tmp_path / "o" / "agreement.csv").exists()


def test_main_has_the_assays_help(capsys):
    with pytest.raises(SystemExit):
        review.main(make_review([]), ["--help"])
    printed = capsys.readouterr().out
    assert "The doc." in printed and "--out" in printed and "--seed" in printed
