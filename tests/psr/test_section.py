import colorsys
import hashlib
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import tifffile

from msq.core.segment import FixedLabels
from msq.core.synthetic.psr import synthetic_psr_section
from msq.psr import section as sec

GLASS = np.array([200.0, 190.0, 210.0])  # empty glass, RGB, a slightly blue lamp like the real scans
YELLOW = np.array([190.0, 175.0, 40.0])  # PSR fiber under that lamp
RED = np.array([170.0, 30.0, 70.0])      # PSR collagen
BLUE = np.array([60.0, 60.0, 190.0])     # neither


def as_uint8(rgb: np.ndarray) -> np.ndarray:
    return np.clip(np.round(rgb), 0, 255).astype(np.uint8)


def as_file(rgb: np.ndarray) -> np.ndarray:
    """An RGB image as the ND2 reader returns it (BGR, uint8)."""
    return as_uint8(rgb)[..., ::-1].copy()


# ---------------------------------------------------------------- colour


@pytest.mark.parametrize("rgb", [(255, 0, 0), (255, 255, 0), (255, 0, 64), (30, 200, 90), (12, 40, 230), (90, 90, 90),
                                 (0, 0, 0), (250, 120, 10)])
def test_hsb_bytes_match_imagej(rgb):
    h, s = sec.hue_sat_bytes(np.array([[rgb]], np.float32))
    eh, es, _ = colorsys.rgb_to_hsv(*(np.array(rgb) / 255))
    assert h[0, 0] == int(eh * 255) and s[0, 0] == int(es * 255)  # ImageJ truncates to bytes


def test_classify_pure_colours():
    img = np.stack([GLASS, YELLOW, RED, BLUE])[None]
    assert sec.classify(as_uint8(img), GLASS, sec.Params()).tolist() == [[sec.BG, sec.YELLOW, sec.RED, sec.OTHER]]


def test_classify_is_independent_of_exposure():
    rng = np.random.default_rng(0)
    tissue = np.where(rng.random((40, 40, 1)) < 0.3, RED, YELLOW) * rng.uniform(0.85, 1.0, (40, 40, 1))
    bright = sec.classify(as_uint8(tissue), GLASS, sec.Params())
    dark = sec.classify(as_uint8(tissue * 0.45), GLASS * 0.45, sec.Params())  # same tissue, darker scan
    assert (bright == dark).mean() > 0.99


# ---------------------------------------------------------------- reading


def test_background_colour_comes_from_glass():
    small = np.tile(GLASS, (200, 200, 1)).astype(np.float32)
    small[60:140, 50:150] = YELLOW
    bg, tissue = sec.background_colour(small, px_um=1.3, margin_um=10)
    assert np.allclose(bg, GLASS) and tissue[100, 100] and not tissue[5, 5]


def test_background_colour_needs_glass_away_from_the_tissue():
    small = np.tile(GLASS, (200, 200, 1)).astype(np.float32)
    small[10:190, 10:190] = YELLOW  # glass only in a 10 px frame, which the 40 um margin around the tissue takes
    with pytest.raises(ValueError, match="too few glass pixels"):
        sec.background_colour(small, px_um=1.3, margin_um=40)
    assert np.allclose(sec.background_colour(small, px_um=1.3, margin_um=5)[0], GLASS)


# ---------------------------------------------------------------- zones and counts


def test_count_classes_on_the_full_grid():
    labels = np.array([[1, 0], [2, 2]], np.uint32)  # ds grid, f = 2
    classes = np.array([[sec.RED, sec.YELLOW, sec.BG, sec.BG],
                        [sec.YELLOW, sec.YELLOW, sec.BG, sec.OTHER],
                        [sec.RED, sec.RED, sec.YELLOW, sec.YELLOW],
                        [sec.RED, sec.RED, sec.YELLOW, sec.YELLOW]], np.uint8)
    counts = np.zeros((3, 4), np.int64)
    sec.add_counts(counts, labels, classes, f=2, y0=0)
    assert counts[1].tolist() == [0, 1, 3, 0]   # bg, red, yellow, other
    assert counts[0].tolist() == [3, 0, 0, 1]
    assert counts[2].tolist() == [0, 4, 4, 0]


def test_object_table_gives_counts_and_fractions_per_zone_and_nan_without_pixels():
    labels = np.zeros((4, 4), np.uint32)
    labels[0:2, 0:2], labels[2:4, 2:4] = 1, 2
    section = np.ones((4, 4), bool)
    section[2:, 2:] = False  # object 2's centroid lies outside the main section
    counts = {"core": np.array([[0, 0, 0, 0], [1, 6, 2, 1], [0, 0, 0, 0]]),  # bg, red, yellow, other
              "ring": np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 3, 1, 0]])}  # object 1 has no ring pixels
    df = sec.object_table(labels, section, counts, px_ds=2.0, f=3)
    assert df["label"].tolist() == [1, 2] and df["in_section"].tolist() == [True, False]
    assert df["core_red"].tolist() == [6, 0] and df["ring_yellow"].tolist() == [0, 1] and df["core_other"].tolist() == [1, 0]
    assert df["red_frac_core"][0] == 0.6 and df["yellow_frac_core"][0] == 0.2 and np.isnan(df["red_frac_core"][1])
    assert np.isnan(df["red_frac_ring"][0]) and df["red_frac_ring"][1] == 0.75
    assert sec.object_table(np.zeros((4, 4), np.uint32), section, counts, 2.0, 3).empty


def test_a_section_without_objects_still_gets_its_images_and_record(tmp_path):
    p = sec.Params(downsample=6, bg_margin_um=6, section_close_um=5)
    out = sec.run_frame(synthetic_section(), px_um=0.2167, out_dir=tmp_path / "s", p=p, sample="E", source="synthetic",
                        segment=lambda img: np.zeros(img.shape[:2], np.uint32))
    info = json.loads((out / "section.json").read_text())
    assert info["n_objects"] == info["n_objects_in_section"] == 0 and info["coverage_yellow"] == 0
    assert list(info) == SECTION_KEYS and (out / "objects.csv").exists()
    assert all((out / name).exists() for name in ["labels_ds6.tif", "rgb_ds6.tif", "classes_ds6.tif"])


# ---------------------------------------------------------------- end to end


def synthetic_section(f: int = 6) -> np.ndarray:
    """Glass with a block of tissue: yellow fibers on a red background, as the file stores it (BGR)."""
    rgb = np.tile(GLASS, (240, 300, 1))
    rgb[30:222, 30:258] = RED
    for y in range(36, 200, 36):
        for x in range(36, 230, 36):
            rgb[y:y + 30, x:x + 30] = YELLOW  # 30 px fibers with 6 px of collagen between them
    return as_file(rgb)


def fake_segment(img: np.ndarray) -> np.ndarray:
    """Stand-in for Cellpose: bright connected regions of the segmentation input are fibers."""
    from scipy import ndimage as ndi

    lab, _ = ndi.label((img > 0.6) & (img < 0.99))
    return lab.astype(np.uint32)


SECTION_KEYS = [  # section.json: the keys other code reads (the last two say where the numbers came from)
    "sample", "source", "variant", "params", "pixel_size_um", "grid_um", "edge_px", "image_px", "background_rgb",
    "section_area_mm2", "n_objects", "n_objects_in_section", "section_class_px", "red_fraction",
    "collagen_outside_fraction", "coverage_yellow", "red_in_masks", "unsegmented_class_px", "seconds",
    "peak_memory_gb", "code_version", "cellpose_weights"]


def test_run_frame_writes_a_section(tmp_path):
    p = sec.Params(downsample=6, bg_margin_um=6, section_close_um=5)  # the synthetic section is small
    out = sec.run_frame(synthetic_section(), px_um=0.2167, out_dir=tmp_path / "s", p=p, sample="TEST",
                        source="synthetic", segment=fake_segment)
    objects = pd.read_csv(out / "objects.csv")
    info = json.loads((out / "section.json").read_text())
    assert len(objects) == 30
    assert list(info) == SECTION_KEYS and set(info["seconds"]) == {"read", "segment", "classify_count", "total"}
    assert info["cellpose_weights"] is None  # the segmentation was a stand-in, no weights played a part
    assert objects["red_frac_core"].max() < 0.05          # cores of the fake fibers are yellow
    assert objects["red_frac_ring"].min() > 0.9           # the ring around them is collagen
    assert info["coverage_yellow"] > 0.95
    assert 0.3 < info["red_fraction"] < 0.6
    assert abs(info["collagen_outside_fraction"] - info["red_fraction"]) < 0.01  # all collagen lies between fibers
    assert info["red_in_masks"] < 0.01
    assert np.allclose(info["background_rgb"], GLASS, atol=1)
    for name in ["labels_ds6.tif", "rgb_ds6.tif", "classes_ds6.tif"]:
        assert (out / name).exists()
    hist = np.load(out / "hue_sat_hist.npz")["counts"]
    assert hist.shape == (256, 256) and hist.sum() == sum(info["section_class_px"].values())


def test_yellow_input_is_bright_only_in_fibers():
    small = np.stack([GLASS, YELLOW, RED, GLASS * 0.5])[None].astype(np.float32)  # glass, fiber, collagen, dark spot
    got = sec.segmentation_input(small, GLASS, "yellow")[0]
    assert got[1] > 0.6 and got[0] == 0 and got[2] == 0 and got[3] == 0


def test_fiber_input_keeps_holes_bright_and_collagen_dark():
    pink = np.array([200.0, 150.0, 185.0])  # pale collagen under the glass colour
    small = np.stack([GLASS, YELLOW, RED, pink])[None].astype(np.float32)  # glass or hole, fiber, collagen, pale
    got = sec.segmentation_input(small, GLASS, "fiber_raw")[0]
    assert got[0] == 1 and got[1] > 0.8 and got[2] == 0 and got[3] < 0.5


def test_fiber_input_darkens_large_white_but_not_holes():
    small = np.tile(YELLOW, (40, 40, 1)).astype(np.float32)
    small[:, :10] = GLASS                 # glass along the left edge: 400 px at 1 um/px
    small[20:22, 25:27] = GLASS           # a 4 px hole inside the fiber
    got = sec.segmentation_input(small, GLASS, "fiber", grid_um=1.0)
    assert got[:, :10].max() == 0 and got[20, 25] == 1 and got[30, 30] > 0.8


def test_variants_change_only_their_setting():
    base = sec.Params()
    assert sec.params_for("ds4") == replace(base, downsample=4)
    assert sec.params_for("od").seg_input == "green_od" and sec.params_for("green").seg_input == "green_t"
    assert sec.Params().seg_input == "fiber" and sec.params_for("yellow").seg_input == "yellow"
    with pytest.raises(KeyError):
        sec.params_for("nope")


def image_digests(folder) -> dict:
    """First 16 hex digits of the sha256 of each image's pixels (decoded, so the compression does not matter)."""
    out = {}
    for f in sorted(folder.rglob("*.tif")):
        a = tifffile.imread(f)
        out[str(f.relative_to(folder))] = hashlib.sha256(f"{a.dtype}{a.shape}".encode() + a.tobytes()).hexdigest()[:16]
    return out


def test_the_images_of_a_seeded_section_are_pinned(tmp_path):
    """Images are not compared by the goldens: labels, normalised colour, class fractions and the section mask."""
    s = synthetic_psr_section(21, "high")
    out = sec.run_frame(s["frame"], s["px_um"], tmp_path / "s", sec.params_for("base"), "S1", "synthetic:S1",
                        FixedLabels(s["labels_ds"]), "base")
    assert image_digests(out) == {"classes_ds6.tif": "0d6c4c152da2ea52", "labels_ds6.tif": "db980ea84f2c16d0",
                                  "rgb_ds6.tif": "1d618bed92a0ae00", "section_ds6.tif": "c23c6be4ff3cf9f6"}
