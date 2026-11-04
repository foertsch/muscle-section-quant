import numpy as np
import pytest
import tifffile

from msq.core import segment
from msq.core.segment import CellposeSegmenter, CellposeSettings, FixedLabels, labels_from_file


def test_fixed_labels_return_the_known_segmentation():
    labels = np.array([[0, 1], [2, 2]], np.int64)
    seg = FixedLabels(labels)
    out = seg(np.zeros((2, 2), np.float32))
    assert out.dtype == np.uint32 and (out == labels).all()
    assert (seg(np.zeros((2, 2, 3))) == labels).all()  # an RGB image on the same grid
    with pytest.raises(ValueError):
        seg(np.zeros((3, 3)))


def test_labels_from_file(tmp_path):
    labels = np.arange(6, dtype=np.uint32).reshape(2, 3)
    tifffile.imwrite(tmp_path / "labels_ds4.tif", labels, compression="zlib")
    assert (labels_from_file(tmp_path / "labels_ds4.tif")(np.zeros((2, 3))) == labels).all()


def test_cellpose_segmenter_passes_the_shared_settings(monkeypatch):
    calls = []

    class Model:
        def eval(self, img, **kw):
            calls.append(kw)
            return np.ones(img.shape[:2], np.int32), None, None

    monkeypatch.setattr(segment, "cellpose_model", lambda model, gpu: Model())
    seg = CellposeSegmenter(CellposeSettings(flow_threshold=0.6, batch_size=4))
    out = seg(np.zeros((8, 8), np.float32))
    assert out.dtype == np.uint32 and out.shape == (8, 8)
    assert calls == [{"batch_size": 4, "flow_threshold": 0.6, "cellprob_threshold": 0.0, "normalize": True}]
    assert CellposeSettings() == CellposeSettings(0.4, 0.0, 16) and segment.DEFAULT_MODEL == "cpsam_v2"


def test_model_is_loaded_once_per_process(monkeypatch):
    made = []

    class Models:
        @staticmethod
        def CellposeModel(gpu, pretrained_model):
            made.append((gpu, pretrained_model))
            return object()

    import sys
    monkeypatch.setitem(sys.modules, "cellpose", type(sys)("cellpose"))
    monkeypatch.setitem(sys.modules, "cellpose.models", Models)
    sys.modules["cellpose"].models = Models
    monkeypatch.setattr(segment, "_MODELS", {})
    a = segment.cellpose_model("demo_weights", gpu=False)
    b = segment.cellpose_model("demo_weights", gpu=False)
    assert a is b and made == [(False, "demo_weights")]


def test_cellpose_for_takes_the_thresholds_and_batch_size_of_any_params_object():
    class Params:
        cp_flow_threshold, cp_cellprob_threshold, cp_batch_size = 0.7, -1.0, 3

    seg = segment.cellpose_for(Params())
    assert isinstance(seg, CellposeSegmenter) and seg.settings == CellposeSettings(0.7, -1.0, 3)
    assert seg.model == segment.DEFAULT_MODEL


def test_segmenter_for_one_variant_all_variants_or_cellpose():
    class Params(CellposeSettings):
        cp_flow_threshold = cp_cellprob_threshold = 0.1
        cp_batch_size = 2

    one, other = FixedLabels(np.ones((2, 2))), FixedLabels(np.zeros((2, 2)))
    assert segment.segmenter_for(one, "laminin", Params()) is one
    assert segment.segmenter_for({"laminin": one, "inverted": other}, "inverted", Params()) is other
    with pytest.raises(KeyError):
        segment.segmenter_for({"laminin": one}, "inverted", Params())
    assert segment.segmenter_for(None, "laminin", Params()).settings == CellposeSettings(0.1, 0.1, 2)
