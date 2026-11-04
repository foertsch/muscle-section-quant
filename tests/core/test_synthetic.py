"""The builders reproduce the goldens' inputs exactly: the same seeds give the same arrays the old repos stored."""

import json
import zlib
from pathlib import Path

import numpy as np
import pytest

from msq.core.synthetic import dystrophin, necrosis, psr

GOLDEN = Path(__file__).resolve().parents[1] / "golden"


def golden_inputs(pipeline: str):
    if not (GOLDEN / pipeline / "golden.json").exists():
        pytest.skip(f"no {pipeline} golden in this tree (it is published with its assay)")
    meta = json.loads((GOLDEN / pipeline / "golden.json").read_text())
    with np.load(GOLDEN / pipeline / "inputs.npz") as z:
        return meta, {k: z[k] for k in z.files}


def test_dystrophin_builder_reproduces_the_golden_inputs():
    meta, arrays = golden_inputs("dystrophin")
    for d in meta["design"]:
        s = dystrophin.synthetic_section(zlib.crc32(d["sample"].encode()), d["group"])
        for key, name in (("gfp", "gfp"), ("act", "act"), ("labels_ds", "labels_ds4")):
            assert np.array_equal(s[key], arrays[f"{d['sample']}/{name}"]), (d["sample"], key)
    assert set(dystrophin.synthetic_section(3, "ctrl", size=512)["truth"]["kind"]) >= {"fiber", "hollow", "debris"}


def test_psr_builders_reproduce_the_golden_inputs():
    meta, arrays = golden_inputs("psr")
    for d in meta["design"]:
        s = psr.synthetic_psr_section(zlib.crc32(d["sample"].encode()), d["fibrosis"])
        assert np.array_equal(s["frame"], arrays[f"{d['sample']}/frame_bgr"])
        assert np.array_equal(s["labels_ds"], arrays[f"{d['sample']}/labels_ds6"])
    field = psr.synthetic_he_field(zlib.crc32(b"he_field"))
    assert np.array_equal(field["rgb"], arrays["he_field/rgb"])
    assert np.array_equal(field["fiber_labels"], arrays["he_field/fiber_labels"])


def test_necrosis_builder_reproduces_the_golden_inputs():
    meta, arrays = golden_inputs("necrosis")
    for d in meta["design"]:
        s = necrosis.synthetic_necrosis_section(zlib.crc32(d["sample"].encode()), d["group"])
        assert np.array_equal(s["frame"], arrays[f"{d['sample']}/frame"])
        assert np.array_equal(s["labels_ds"], arrays[f"{d['sample']}/labels_ds4"])
        assert s["channels"] == list(arrays["channels"])


def test_fake_section_tables_are_seeded(tmp_path):
    a = dystrophin.write_section(tmp_path / "a", "TA_sec1_0001_20x_ctrl", n_fibers=50)
    b = dystrophin.write_section(tmp_path / "b", "TA_sec1_0001_20x_ctrl", n_fibers=50)
    assert (a / "fibers.csv").read_text() == (b / "fibers.csv").read_text()
    assert (a / "sectors.csv.gz").exists() and json.loads((a / "run_info.json").read_text())["params"]["n_sectors"] == 36
    assert psr.header(exposure_ms=22.0)["imaging"]["exposure_ms"] == 22.0 and psr.HEADER["imaging"]["exposure_ms"] == 11.0
    assert necrosis.header("2026-09-01 10:00:00", gain=2.0)["imaging"]["nec"]["gain"] == 2.0
