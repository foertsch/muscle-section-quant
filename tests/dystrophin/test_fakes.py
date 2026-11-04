import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from msq.dystrophin import measure as dq  # noqa: E402
from msq.core.synthetic.dystrophin import synthetic_section  # noqa: E402


def test_synthetic_section_is_seeded_and_labelled():
    a, b = synthetic_section(7, "treated", size=512), synthetic_section(7, "treated", size=512)
    assert np.array_equal(a["gfp"], b["gfp"]) and np.array_equal(a["labels_ds"], b["labels_ds"])
    assert not np.array_equal(a["gfp"], synthetic_section(8, "treated", size=512)["gfp"])
    assert a["labels_ds"].shape == (128, 128) and a["labels_ds"].max() == len(a["truth"])
    kinds = set(a["truth"]["kind"])
    assert {"fiber", "hollow", "bright_cytoplasm", "debris", "border", "fragment"} <= kinds
    assert (a["gfp"][:32] == 0).all()  # zero padding, as in edited scans


def test_synthetic_section_runs_through_the_measurement(tmp_path):
    s = synthetic_section(3, "ctrl", size=512)
    f = dq.Params().downsample
    act_ds, gfp_ds = dq.downsample(s["act"], f), dq.downsample(s["gfp"], f)
    tissue, off = dq.tissue_mask(act_ds, s["px_um"] * f, dq.Params().tissue_margin_um)
    bg = dq.background_level(gfp_ds, off)
    assert bg == 500.0 and dq.background_level(act_ds, off) == 500.0
    df = dq.measure_all(s["labels_ds"], s["gfp"], s["act"], s["px_um"], dq.Params(), bg, bg, s["sat"], progress=False)
    df["in_main_section"] = True
    df = dq.add_flags(df, dq.Params())
    truth = s["truth"].set_index("label")
    assert df.loc[df["label"].map(truth["kind"]) == "hollow", "excl_hollow"].all()
    assert df.loc[df["label"].map(truth["kind"]) == "debris", "excl_size"].all()
    assert df.loc[df["label"].map(truth["kind"]) == "border", "touches_border"].all()
    assert df.loc[df["label"].map(truth["kind"]) == "fiber", "include"].mean() > 0.9
