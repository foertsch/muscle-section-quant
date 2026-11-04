"""What a run records about its code and weights (version.py in dystrophin-quant, msq.core.runinfo here)."""


import numpy as np

from msq.core import segment
from msq.dystrophin import measure as dq
from msq.dystrophin import run_pipeline as rp


def test_segmentation_loads_the_named_weights(monkeypatch):
    """The model is named in the code, so a new cellpose default cannot switch weights unnoticed."""
    from cellpose import models

    seen = {}

    class Model:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def eval(self, img, **kwargs):
            seen["eval"] = kwargs
            return np.zeros(img.shape, np.int32), None, None

    monkeypatch.setattr(models, "CellposeModel", Model)
    monkeypatch.setattr(segment, "_MODELS", {})  # no model cached from another test
    out = dq.cellpose_for(dq.Params())(np.zeros((8, 8), np.float32))
    assert out.dtype == np.uint32
    assert seen["pretrained_model"] == segment.DEFAULT_MODEL == "cpsam_v2"
    p = dq.Params()
    assert seen["eval"] == {"batch_size": p.cp_batch_size, "flow_threshold": p.cp_flow_threshold,
                            "cellprob_threshold": p.cp_cellprob_threshold, "normalize": True}
    assert rp.weights_file().name == "cpsam_v2"
