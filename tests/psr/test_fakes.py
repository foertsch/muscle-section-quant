import numpy as np

from msq.core.synthetic.psr import synthetic_psr_section
from msq.psr import section as sec


def test_synthetic_psr_section_is_seeded_and_classifies():
    a, b = synthetic_psr_section(5), synthetic_psr_section(5)
    assert np.array_equal(a["frame"], b["frame"]) and a["labels_ds"].max() == len(a["truth"])
    assert not np.array_equal(a["frame"], synthetic_psr_section(6)["frame"])
    rgb = a["frame"][..., ::-1]
    small = sec.downsample_rgb(a["frame"], 6)
    bg, tissue = sec.background_colour(small, 1.3, 200.0)
    assert np.allclose(bg, (200, 190, 210), atol=1) and tissue[300, 300] and not tissue[20, 20]
    classes = sec.classify(rgb[1750:1850, 1750:1850], bg, sec.Params())
    assert {sec.RED, sec.YELLOW} <= set(np.unique(classes))
