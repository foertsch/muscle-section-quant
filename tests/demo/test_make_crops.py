"""The README crops: images only, from a finished run, with nothing in them or their names that points to a section."""

import importlib.util
import struct
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from scipy import ndimage as ndi

DEMO = Path(__file__).resolve().parents[2] / "demo"


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, DEMO / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


run_demo = load("run_demo")
mc = load("make_crops")


@pytest.fixture(scope="module")
def run(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("demo")
    assert run_demo.run_demo(out, sections=1) == 0
    return out / "run"


def treated(run: Path) -> str:
    samples = pd.read_csv(run / "samples.csv", dtype=str)
    return samples.loc[samples["group"] == "treated", "sample"].iloc[0]


def chunks(path: Path) -> list[str]:
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    out, i = [], 8
    while i < len(data):
        n, kind = struct.unpack(">I4s", data[i:i + 8])
        out.append(kind.decode())
        i += 12 + n
    return out


def test_writes_the_crops_as_bare_pngs(run, tmp_path):
    out = tmp_path / "crops"
    got = mc.make_crops({"run": run, "sample": treated(run), "calls": True, "out": out})
    assert sorted(p.name for p in out.iterdir()) == ["calls.png", "ring_sectors.png", "segmentation.png"]
    for p in out.iterdir():
        assert set(chunks(p)) == {"IHDR", "IDAT", "IEND"}, p.name  # no text, time or profile chunks
    size = got["size"]
    assert size == 1000  # 360 um at 0.3606 um/px, on the 4x grid
    for name in ("segmentation.png", "calls.png"):
        assert Image.open(out / name).size == (size, size)


def test_without_calls_there_is_no_calls_image(run, tmp_path):
    out = tmp_path / "crops"
    mc.make_crops({"run": run, "sample": treated(run), "out": out})
    assert sorted(p.name for p in out.iterdir()) == ["ring_sectors.png", "segmentation.png"]


def test_outlines_take_the_colour_of_each_fiber_status(run, tmp_path):
    out = tmp_path / "crops"
    mc.make_crops({"run": run, "sample": treated(run), "calls": True, "out": out})
    seg = np.asarray(Image.open(out / "segmentation.png"))
    assert (seg == mc.rgb(mc.STATUS["included"])).all(-1).sum() > 1000
    calls = np.asarray(Image.open(out / "calls.png"))
    assert (calls == mc.rgb(mc.CALL["positive"])).all(-1).sum() > 100
    assert (calls == mc.rgb(mc.CALL["negative"])).all(-1).sum() > 100


def test_a_window_can_be_given(run, tmp_path):
    got = mc.make_crops({"run": run, "sample": treated(run), "at": [403, 798], "out": tmp_path / "c"})
    assert (got["y0"], got["x0"]) == (400, 796)  # floored to the segmentation grid


def test_the_picks_move_the_window_along_local_dystrophin(run, tmp_path):
    med = {}
    for pick in ("lower", "higher"):
        got = mc.make_crops({"run": run, "sample": treated(run), "pick": pick, "out": tmp_path / pick})
        med[pick] = got["local_median"]
    assert med["lower"] < med["higher"]


def test_the_ring_image_draws_the_band_and_core_the_measurement_used(run):
    """Sector values recomputed from the drawn band and core equal the stored ones (sectors.csv.gz, 0.1 a.u.)."""
    sec = mc.load_section(run, treated(run))
    p = mc.dq.params_for(sec["px_um"])
    for label in sec["fibers"].query("include")["label"].iloc[:5]:
        g = mc.fiber_rings(sec, int(label))
        gs = ndi.gaussian_filter(g["gfp"], p.smooth_px)
        yy, xx = np.nonzero(g["band"])
        ridge = mc.dq.sector_ridge(gs[g["band"]], yy, xx, g["cy"], g["cx"], p.n_sectors, p.sector_pct)
        stored = sec["sectors"].query("label == @label").sort_values("sector")["ridge_minus_cyto"].to_numpy()
        assert np.allclose(ridge - g["gfp"][g["core"]].mean(), stored, atol=0.051, equal_nan=True), label
        assert np.array_equal(g["values"], stored, equal_nan=True)


def test_a_chosen_fiber_must_be_included(run, tmp_path):
    sec = mc.load_section(run, treated(run))
    excluded = int(sec["fibers"].query("not include")["label"].iloc[0])
    with pytest.raises(ValueError, match="not an included fiber"):
        mc.make_crops({"run": run, "sample": treated(run), "fiber": excluded, "out": tmp_path / "c"})


def test_window_size_follows_the_image_rules(run, tmp_path):
    with pytest.raises(ValueError, match="300 to 600"):
        mc.make_crops({"run": run, "sample": treated(run), "window_um": 1000, "out": tmp_path / "c"})


def test_an_unknown_sample_is_named_in_the_error(run, tmp_path):
    with pytest.raises(ValueError, match="no section 'TA_sec9' in"):
        mc.make_crops({"run": run, "sample": "TA_sec9", "out": tmp_path / "c"})


def test_the_config_reads_paths_and_defaults(tmp_path):
    p = tmp_path / "crops.local.toml"
    p.write_text('run = "/x/run"\nsample = "S"\nat = [10, 20]\n', encoding="utf-8")
    cfg = mc.read_config(p)
    assert cfg["run"] == Path("/x/run") and cfg["at"] == [10, 20]
    assert cfg["pick"] == "typical" and cfg["window_um"] == 360 and cfg["calls"] is False
    assert cfg["out"] == DEMO / "crops"


def test_the_example_config_names_every_key():
    text = (DEMO / "crops.example.toml").read_text(encoding="utf-8")
    for key in mc.DEFAULTS:
        assert f"{key} =" in text, key
