"""The synthetic PSR demo and the PSR mode of make_crops, on a small run in tmp_path. No R output is read."""

import importlib.util
import struct
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

DEMO = Path(__file__).resolve().parents[2] / "demo"


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, DEMO / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


psr_demo = load("run_psr_demo")
mc = load("make_crops")
SAMPLE = "QUAD_sec1_B01_group_a"


@pytest.fixture(scope="module")
def out(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("psr_demo")
    got = psr_demo.run_demo(out, sections=1, r_figures=False)
    assert got["r_figures"] is False
    return out


def test_the_demo_writes_the_tables_with_the_primary_readout(out):
    for name in ("sections.csv", "animals.csv", "genotypes.csv", "palette.csv"):
        assert (out / "tables" / name).is_file(), name
    sections = pd.read_csv(out / "tables" / "sections.csv", dtype={"animal": str})
    animals = pd.read_csv(out / "tables" / "animals.csv", dtype={"animal": str})
    for table in (sections, animals):
        assert "collagen_outside_fraction" in table.columns
        assert table["collagen_outside_fraction"].between(0, 1).all()
    assert sorted(animals["animal"]) == ["A01", "A02", "B01", "B02"]
    assert "freezing_artifact_area" in sections.columns
    mean = animals.groupby("group")["collagen_outside_fraction"].mean()
    assert mean["group_a"] > mean["control"] + 0.05  # the synthetic treated group has more collagen between fibers
    assert "collagen outside the fiber outlines" in psr_demo.report(out)


def test_the_demo_keeps_the_per_section_outputs_the_crops_read(out):
    d = out / "run" / "sections" / "base" / SAMPLE
    for name in ("section.json", "rgb_ds6.tif", "classes_ds6.tif", "labels_ds6.tif", "section_ds6.tif"):
        assert (d / name).is_file(), name


def test_the_design_uses_the_example_cohorts_groups():
    from msq.psr.settings import load_cohort

    groups = {d["group"] for d in psr_demo.build_design()}
    assert groups <= set(load_cohort("example").order)
    assert len(psr_demo.build_design(2)) == 8


def test_without_rscript_the_r_step_is_skipped(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(psr_demo.shutil, "which", lambda name: None)
    assert psr_demo.draw_r_figures(tmp_path, tmp_path / "figures") is False
    assert "skipping the R figures" in capsys.readouterr().out
    assert not (tmp_path / "figures").exists()


def chunks(path: Path) -> set[str]:
    data = path.read_bytes()
    out, i = set(), 8
    while i < len(data):
        n, kind = struct.unpack(">I4s", data[i:i + 8])
        out.add(kind.decode())
        i += 12 + n
    return out


def test_psr_crops_write_four_bare_pngs_of_one_window(out, tmp_path):
    crops = tmp_path / "crops"
    got = mc.make_crops({"assay": "psr", "run": out / "run", "sample": SAMPLE, "window_um": 300, "out": crops})
    assert sorted(p.name for p in crops.iterdir()) == sorted(mc.PSR_NAMES)
    side = got["size"] // 6 * mc.PSR_ZOOM
    for p in crops.iterdir():
        assert chunks(p) == {"IHDR", "IDAT", "IEND"}, p.name
        assert Image.open(p).size == (side, side)
    assert 0 < got["collagen_outside"] < 1


def test_the_psr_crops_show_what_they_say(out, tmp_path):
    crops = tmp_path / "crops"
    mc.make_crops({"assay": "psr", "run": out / "run", "sample": SAMPLE, "window_um": 300, "out": crops})
    img = {n: np.asarray(Image.open(crops / n).convert("RGB")) for n in mc.PSR_NAMES}
    has = lambda name, colour: (img[name] == mc.rgb(mc.PSR_COLOURS[colour])).all(-1).sum()  # noqa: E731
    assert has("psr_classes.png", "red") > 1000 and has("psr_classes.png", "yellow") > 1000
    assert has("psr_outlines.png", "outline") > 500  # fiber outlines
    assert has("psr_collagen_outside.png", "outside") > 1000
    assert has("psr_collagen_outside.png", "outline") > 500
    assert has("psr_colour.png", "outside") == 0  # the plain colour image is not annotated
    assert (img["psr_colour.png"][-30:-20, 20:60] == 255).all()  # the scale bar


def test_the_highlighted_collagen_is_red_in_the_section_outside_every_outline(out):
    sec = mc.load_psr_section(out / "run", SAMPLE)
    red = mc.psr_red_outside(sec)
    assert red.any()
    assert not (red & (sec["labels"] > 0)).any()
    assert not (red & ~sec["section"]).any()
    assert (sec["classes"].argmax(-1)[red] == 1).all()


def test_the_pick_moves_the_window_along_collagen_outside(out, tmp_path):
    share = {pick: mc.make_crops({"assay": "psr", "run": out / "run", "sample": SAMPLE, "window_um": 300, "pick": pick,
                                   "out": tmp_path / pick})["collagen_outside"] for pick in ("lower", "higher")}
    assert share["lower"] <= share["higher"]


def test_psr_mode_errors_are_clear(out, tmp_path):
    base = {"assay": "psr", "run": out / "run", "out": tmp_path / "c"}
    with pytest.raises(ValueError, match="no PSR section 'nope'"):
        mc.make_crops({**base, "sample": "nope"})
    with pytest.raises(ValueError, match="scan is not used"):
        mc.make_crops({**base, "sample": SAMPLE, "scan": "x.nd2"})
    with pytest.raises(ValueError, match="300 to 600"):
        mc.make_crops({**base, "sample": SAMPLE, "window_um": 100})


def test_the_assay_key_is_read_and_checked(tmp_path):
    p = tmp_path / "crops.local.toml"
    p.write_text('run = "/x"\nsample = "S"\n', encoding="utf-8")
    assert mc.read_config(p)["assay"] == "dystrophin"
    p.write_text('run = "/x"\nsample = "S"\nassay = "psr"\n', encoding="utf-8")
    assert mc.read_config(p)["assay"] == "psr"
    p.write_text('run = "/x"\nsample = "S"\nassay = "he"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="assay must be"):
        mc.read_config(p)


def test_the_example_config_has_a_psr_block_with_the_demos_sample():
    text = (DEMO / "crops.example.toml").read_text(encoding="utf-8")
    assert 'assay = "psr"' in text and SAMPLE in text
