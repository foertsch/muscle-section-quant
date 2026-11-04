"""The shared per-section driver (msq.core.section), on a fake assay: folders written whole, the two layouts, what the
section's record is stamped with, and the writers."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import tifffile

from msq.core import runinfo, section
from msq.core.section import Layout, Outputs, Prepared, SectionFolder, run_section
from msq.core.segment import CellposeSegmenter, FixedLabels

ONE = Layout(seconds_in="seconds")      # psr-like: one variant in the section folder
MANY = Layout(shared_json="prep.json")  # necrosis-like: a folder per variant and the section's own record


class Fake:
    """An assay in three methods: the scan is an image, a variant's labels come from the segmenter."""

    def __init__(self):
        self.calls = []

    def params(self, variant):
        return SimpleNamespace(cp_flow_threshold=0.4, cp_cellprob_threshold=0.0, cp_batch_size=16)

    def prepare(self, scan, variants):
        self.calls.append(("prepare", tuple(variants)))
        return Prepared(scan, {"shared.tif": scan.astype(np.uint16)}, {"sample": "S", "seconds_read": 0.1})

    def run_variant(self, prepared, variant, segment):
        self.calls.append(("variant", variant))
        labels = segment(prepared.state)
        return Outputs({"variant": variant, "n": int(labels.max()), "seconds": {"segment": 0.1}},
                       {"labels.tif": labels, "table.csv": pd.DataFrame({"label": [1, 2], "x": [0.5, 1.5]}),
                        "h.npz": {"counts": np.arange(4)}})


class StandInCellpose(CellposeSegmenter):
    """Counts as Cellpose for the record, but needs no weights."""

    def __call__(self, img):
        return (img > 0).astype(np.uint32)


@pytest.fixture
def scan():
    img = np.zeros((6, 6), np.uint8)
    img[1:3, 1:3] = 1
    return img


@pytest.fixture
def weights(tmp_path, monkeypatch):
    """A fake weights file that segment.weights_file finds, and a count of how often it is hashed."""
    f = tmp_path / "weights" / "cpsam_v2"
    f.parent.mkdir()
    f.write_bytes(b"abc")
    hashed = []
    real = runinfo.weights
    monkeypatch.setattr(section, "weights_file", lambda model: f)
    monkeypatch.setattr(runinfo, "weights", lambda path: hashed.append(path) or real(path))
    monkeypatch.setattr(section, "_WEIGHTS", {})
    return SimpleNamespace(file=f, hashed=hashed)


def test_one_variant_goes_into_the_section_folder(scan, tmp_path):
    out = run_section(Fake(), scan, tmp_path / "s", ["base"], FixedLabels(scan), ONE)
    assert out == tmp_path / "s" and not (tmp_path / "s.partial").exists()
    assert {p.name for p in out.iterdir()} == {"shared.tif", "labels.tif", "table.csv", "h.npz", "section.json"}
    info = json.loads((out / "section.json").read_text())
    assert list(info) == ["variant", "n", "seconds", "peak_memory_gb", "code_version", "cellpose_weights"]
    assert info["variant"] == "base" and info["n"] == 1 and list(info["seconds"]) == ["segment", "total"]
    assert info["code_version"] == runinfo.code_version() and info["peak_memory_gb"] > 0
    assert info["cellpose_weights"] is None  # the labels were given, so no weights played a part


def test_a_folder_per_variant_and_the_sections_own_record(scan, tmp_path):
    assay = Fake()
    seg = {"a": FixedLabels(scan), "b": FixedLabels(scan * 2)}
    out = run_section(assay, scan, tmp_path / "s", ["a", "b"], seg, MANY)
    assert {p.name for p in out.iterdir()} == {"shared.tif", "prep.json", "a", "b"}
    for v in "ab":
        assert {p.name for p in (out / v).iterdir()} == {"labels.tif", "table.csv", "h.npz", "section.json"}
        assert json.loads((out / v / "section.json").read_text()) == {"variant": v, "n": 1 if v == "a" else 2,
                                                                      "seconds": {"segment": 0.1}}
    prep = json.loads((out / "prep.json").read_text())
    assert list(prep) == ["sample", "seconds_read", "seconds_total", "peak_memory_gb", "code_version",
                          "cellpose_weights"]
    assert assay.calls == [("prepare", ("a", "b")), ("variant", "a"), ("variant", "b")]
    assert np.array_equal(tifffile.imread(out / "b" / "labels.tif"), scan * 2)


def test_the_one_variant_layout_refuses_more_before_anything_is_read(scan, tmp_path):
    assay = Fake()
    with pytest.raises(ValueError, match="one variant"):
        run_section(assay, scan, tmp_path / "s", ["a", "b"], FixedLabels(scan), ONE)
    assert assay.calls == [] and list(tmp_path.iterdir()) == []


def test_a_failed_variant_leaves_the_finished_folder_and_a_partial_one(scan, tmp_path):
    out = tmp_path / "s"
    run_section(Fake(), scan, out, ["a"], FixedLabels(scan), MANY)
    before = (out / "prep.json").read_bytes()

    def boom(img):
        raise RuntimeError("no memory")

    with pytest.raises(RuntimeError, match="no memory"):
        run_section(Fake(), scan, out, ["a", "b"], {"a": FixedLabels(scan), "b": boom}, MANY)
    assert (out / "prep.json").read_bytes() == before
    partial = tmp_path / "s.partial"
    assert (partial / "shared.tif").exists() and (partial / "a" / "labels.tif").exists()
    assert not list(partial.rglob("*.json"))  # the records are written last, so an interrupted run has none


def test_a_rerun_replaces_the_folder(scan, tmp_path):
    out = tmp_path / "s"
    out.mkdir()
    (out / "stale.txt").write_text("old")
    run_section(Fake(), scan, out, ["a"], FixedLabels(scan), MANY)
    assert not (out / "stale.txt").exists() and (out / "a" / "labels.tif").exists()


def test_cellpose_weights_are_recorded_when_cellpose_ran(scan, tmp_path, weights):
    out = run_section(Fake(), scan, tmp_path / "s", ["base"], StandInCellpose(), ONE)
    got = json.loads((out / "section.json").read_text())["cellpose_weights"]
    assert got == {"file": "cpsam_v2", "bytes": 3, "sha256": runinfo.weights(weights.file)["sha256"]}


def test_the_default_segmenter_is_cellpose_with_the_params_of_the_variant(scan, tmp_path, weights, monkeypatch):
    monkeypatch.setattr(CellposeSegmenter, "__call__", lambda self, img: (img > 0).astype(np.uint32))
    out = run_section(Fake(), scan, tmp_path / "s", ["a", "b"], None, MANY)
    assert json.loads((out / "prep.json").read_text())["cellpose_weights"]["bytes"] == 3


def test_one_cellpose_variant_among_stand_ins_is_enough(scan, tmp_path, weights):
    seg = {"a": FixedLabels(scan), "b": StandInCellpose()}
    out = run_section(Fake(), scan, tmp_path / "s", ["a", "b"], seg, MANY)
    assert json.loads((out / "prep.json").read_text())["cellpose_weights"]["file"] == "cpsam_v2"


def test_weights_are_hashed_once_per_file(scan, tmp_path, weights):
    for name in ("s1", "s2"):
        run_section(Fake(), scan, tmp_path / name, ["a", "b"], StandInCellpose(), MANY)
    assert weights.hashed == [weights.file]
    weights.file.write_bytes(b"abcd")  # another file under the same name: hashed again
    out = run_section(Fake(), scan, tmp_path / "s3", ["a"], StandInCellpose(), MANY)
    assert weights.hashed == [weights.file] * 2
    assert json.loads((out / "prep.json").read_text())["cellpose_weights"]["bytes"] == 4


def test_weights_not_downloaded_yet_are_null(scan, tmp_path, monkeypatch):
    monkeypatch.setattr(section, "weights_file", lambda model: tmp_path / "nowhere")
    out = run_section(Fake(), scan, tmp_path / "s", ["a"], StandInCellpose(), MANY)
    assert json.loads((out / "prep.json").read_text())["cellpose_weights"] is None


# ---------------------------------------------------------------- the writers


def test_files_are_written_by_their_suffix(tmp_path):
    folder = SectionFolder(tmp_path)
    classes = np.arange(4 * 4 * 4, dtype=np.uint8).reshape(4, 4, 4)
    folder.write({"t.csv": pd.DataFrame({"a": [1, 2]}), "plain.tif": np.eye(3, dtype=np.uint16),
                  "classes.tif": (classes, {"photometric": "minisblack", "planarconfig": "contig"}),
                  "h.npz": {"counts": np.arange(3), "edges": np.arange(4.0)}})
    assert (tmp_path / "t.csv").read_text() == "a\n1\n2\n"  # no index column
    with tifffile.TiffFile(tmp_path / "plain.tif") as t:
        assert t.pages[0].compression.name == "ADOBE_DEFLATE" and t.pages[0].dtype == np.uint16
    with tifffile.TiffFile(tmp_path / "classes.tif") as t:
        assert t.pages[0].shape == (4, 4, 4) and t.pages[0].photometric.name == "MINISBLACK"
    assert np.array_equal(tifffile.imread(tmp_path / "classes.tif"), classes)
    z = np.load(tmp_path / "h.npz")
    assert z.files == ["counts", "edges"] and z["edges"].dtype == np.float64
    with pytest.raises(ValueError, match="no writer"):
        folder.write({"notes.txt": "x"})


def test_a_variant_folder_is_a_folder_of_its_own(tmp_path):
    inner = SectionFolder(tmp_path).variant("laminin")
    inner.write_json("section.json", {"a": 1})
    assert (tmp_path / "laminin" / "section.json").read_text() == '{\n "a": 1\n}'  # indent 1, as always
