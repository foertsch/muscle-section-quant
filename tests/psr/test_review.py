
import numpy as np
import pandas as pd
import pytest

from msq.core.segment import FixedLabels
from msq.core.synthetic.psr import synthetic_psr_section
from msq.psr import review_segmentation as rv
from msq.psr import section as sec


def test_core_bins():
    got = rv.core_bin(pd.Series([0.0, 0.049, 0.05, 0.2, 0.3, 0.7, 1.0, np.nan])).tolist()
    assert got == ["clean", "clean", "low", "mid", "high", "red", "red", "no_core"]


class FakeSection:
    def __init__(self, labels, sample="S", variant="base"):
        self.labels = labels
        self.info = {"sample": sample, "variant": variant}

    def image(self, name):
        return self.labels


def test_agreement_identical():
    a = np.zeros((20, 20), np.uint32)
    a[2:8, 2:8], a[10:18, 10:18] = 1, 2
    got = rv.agreement(FakeSection(a), FakeSection(a, variant="v"))
    assert got["matched_iou50"] == 1.0 and got["merges"] == 0 and got["splits"] == 0


def test_agreement_counts_a_merge_and_a_split():
    a = np.zeros((20, 30), np.uint32)
    a[2:8, 2:8], a[2:8, 8:14] = 1, 2      # two fibers side by side
    a[10:18, 10:26] = 3                   # one long fiber
    b = np.zeros_like(a)
    b[2:8, 1:15] = 1                      # the two outlined as one (IoU with each 36/84)
    b[10:18, 10:15], b[10:18, 15:20], b[10:18, 20:26] = 2, 3, 4  # the long one cut in three
    got = rv.agreement(FakeSection(a), FakeSection(b, variant="v"))
    assert got["merges"] == 1 and got["splits"] == 1 and got["matched_iou50"] == 0.0


def test_agreement_resamples_a_finer_grid():
    a = np.zeros((10, 10), np.uint32)
    a[2:6, 2:6] = 1
    b = np.repeat(np.repeat(a, 2, 0), 2, 1)  # the same object on a 2x finer grid
    assert rv.agreement(FakeSection(a), FakeSection(b, variant="v"))["mean_best_iou"] == 1.0


class FakeImages:
    def __init__(self, images, grid_um=1.0):
        self.images = images
        self.info = {"grid_um": grid_um}

    def image(self, name):
        return self.images[name]


def test_coverage_away_from_edge_ignores_the_rim():
    section = np.zeros((40, 40), np.uint8)
    section[5:35, 5:35] = 1
    classes = np.zeros((40, 40, 4), np.uint8)
    classes[5:35, 5:35, 2] = 100            # all yellow
    labels = np.zeros((40, 40), np.uint32)
    labels[10:30, 10:30] = 1                # everything outlined except a 5 px rim
    s = FakeImages({"section": section, "classes": classes, "labels": labels})
    assert rv.coverage_away_from_edge(s, edge_um=5) == 1.0
    assert rv.coverage_away_from_edge(s, edge_um=1) < 1.0


class Reader:
    def close(self):
        pass


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """One synthetic scan, segmented as base and as a second variant that finds the same fibers."""
    root = tmp_path_factory.mktemp("run")
    s = synthetic_psr_section(13, "high")
    for variant in ("base", "alt"):
        sec.run_frame(s["frame"], s["px_um"], root / "sections" / variant / "S1", sec.params_for("base"), "S1",
                      source="synthetic", segment=FixedLabels(s["labels_ds"]), variant=variant)
    pd.DataFrame({"sample": ["S1"], "path": ["scan.nd2"]}).to_csv(root / "samples.csv", index=False)
    return root, s


def test_review_writes_its_tables_and_pictures(run, monkeypatch):
    root, s = run
    monkeypatch.setattr(sec, "open_scan", lambda path: (Reader(), s["frame"], s["px_um"]))
    rv.main([str(root), "--out", str(root / "review"), "--crops", "1", "--per-row", "2"])
    out = root / "review"
    assert {p.name for p in out.iterdir()} >= {"summary.csv", "agreement.csv", "objects_all.csv", "distributions.png",
                                               "overview_S1.png", "crops_S1.png", "variants_S1.png"}
    summary, agree = pd.read_csv(out / "summary.csv"), pd.read_csv(out / "agreement.csv")
    assert summary[["sample", "variant"]].values.tolist() == [["S1", "alt"], ["S1", "base"]]
    assert agree[["sample", "variant", "matched_iou50", "merges", "splits"]].values.tolist() == [["S1", "alt", 1.0, 0, 0]]


def test_review_needs_a_base_variant(run, tmp_path):
    root, _ = run
    (tmp_path / "sections").mkdir()
    (tmp_path / "samples.csv").write_text((root / "samples.csv").read_text())
    with pytest.raises(SystemExit, match="no base variant"):
        rv.main([str(tmp_path), "--out", str(tmp_path / "review")])
