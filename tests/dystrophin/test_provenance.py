"""What a section records about the code and weights that made it, and the warnings when a run's
sections were not all made the same way."""

import json
from pathlib import Path


from msq.core.segment import FixedLabels
from msq.core.synthetic.dystrophin import synthetic_section
from msq.dystrophin import checks
from msq.dystrophin import measure as dq


def problems(infos, code="v1.1.0", weights=None):
    return {p.code: p for p in checks.check_provenance(infos, code, weights)}


def test_one_code_and_one_set_of_weights_give_no_warning():
    w = {"sha256": "a" * 64}
    infos = {s: {"code_version": "v1.1.0", "cellpose_weights": w} for s in ("s1", "s2")}
    assert checks.check_provenance(infos, "v1.1.0", w) == []
    assert checks.check_provenance({}, "v1.1.0", w) == []


def test_other_code_no_record_and_other_weights_each_warn():
    a, b = {"sha256": "a" * 64}, {"sha256": "b" * 64}
    infos = {"s1": {"code_version": "v1.1.0", "cellpose_weights": a},
             "s2": {"code_version": "v1.0.0-3-gabc", "cellpose_weights": b},
             "s3": {"params": {}},                                       # dystrophin-quant v1.0.0 kept no record
             "s4": {"code_version": "v1.1.0", "cellpose_weights": None}}  # a segmentation cached from elsewhere
    ps = checks.check_provenance(infos, "v1.1.0", a)
    texts = [p.message for p in ps]
    assert [p.code for p in ps] == [checks.MIXED_CODE, checks.MIXED_CODE, checks.MIXED_WEIGHTS]
    assert all(p.level == "warning" for p in ps)
    assert texts[0].startswith("1 section(s) were measured by other code (v1.0.0-3-gabc) than this run (v1.1.0): s2")
    assert texts[1].startswith("1 section(s) carry no record of the code that measured them") and "s3" in texts[1]
    assert texts[2].startswith("1 section(s) were segmented with other Cellpose weights") and ": s2." in texts[2]
    assert not [p for p in checks.check_provenance(infos, "v1.1.0", None) if p.code == checks.MIXED_WEIGHTS]


def test_long_lists_are_cut():
    infos = {f"s{i}": {"code_version": "old"} for i in range(8)}
    (p,) = checks.check_provenance(infos, "new", None)
    assert p.message.startswith("8 section(s)") and "s4, …" in p.message and "s5" not in p.message


def run_one(tmp_path: Path, monkeypatch, out_root: Path, provenance, resegment=False, labels=True):
    s = synthetic_section(5, "treated", size=512)
    scan = tmp_path / "scan.nd2"
    scan.write_bytes(b"")
    monkeypatch.setattr(dq, "read_nd2", lambda path: {k: s[k] for k in ("gfp", "act", "px_um", "sat", "channels")})
    return dq.run(scan, out_root, name="S1", resegment=resegment, provenance=provenance,
                  segment=FixedLabels(s["labels_ds"]) if labels else None)


def weights(path: Path, content: bytes) -> Path:
    path.write_bytes(content)
    return path


def test_measure_records_code_and_weights_and_keeps_the_record_of_a_cached_segmentation(tmp_path, monkeypatch,
                                                                                         capsys):
    monkeypatch.setattr(dq.runinfo, "code_version", lambda: "v1.1.0")
    wa, wb = weights(tmp_path / "a", b"A"), weights(tmp_path / "b", b"BB")
    out = tmp_path / "out"
    run_one(tmp_path, monkeypatch, out, dq.Provenance(lambda: wa))  # segments now with weights A
    info = json.loads((out / "S1" / "run_info.json").read_text())
    assert info["code_version"] == "v1.1.0" and info["cellpose_weights"]["bytes"] == 1

    monkeypatch.setattr(dq.runinfo, "code_version", lambda: "v1.2.0")
    run_one(tmp_path, monkeypatch, out, dq.Provenance(lambda: wb))  # measured again from the cached segmentation
    info = json.loads((out / "S1" / "run_info.json").read_text())
    assert info["code_version"] == "v1.2.0" and info["cellpose_weights"]["bytes"] == 1  # still segmented with A

    run_one(tmp_path, monkeypatch, out, dq.Provenance(lambda: wb), resegment=True)  # segmented again, with B
    assert json.loads((out / "S1" / "run_info.json").read_text())["cellpose_weights"]["bytes"] == 2


def test_a_cached_segmentation_from_elsewhere_has_unknown_weights_and_no_provenance_records_nothing(tmp_path,
                                                                                                   monkeypatch,
                                                                                                   capsys):
    import tifffile

    s = synthetic_section(5, "treated", size=512)
    out = tmp_path / "out"
    (out / "S1").mkdir(parents=True)
    tifffile.imwrite(out / "S1" / "labels_ds4.tif", s["labels_ds"])  # copied in, without a run_info.json
    run_one(tmp_path, monkeypatch, out, dq.Provenance(lambda: weights(tmp_path / "w", b"W")), labels=False)
    info = json.loads((out / "S1" / "run_info.json").read_text())
    assert info["cellpose_weights"] is None and info["code_version"]
    run_one(tmp_path, monkeypatch, tmp_path / "plain", None)
    assert "code_version" not in json.loads((tmp_path / "plain" / "S1" / "run_info.json").read_text())


def test_provenance_reads_the_weights_once_they_exist(tmp_path, monkeypatch):
    monkeypatch.setattr(dq.runinfo, "code_version", lambda: "v1.1.0")
    w = tmp_path / "cpsam_v2"
    prov = dq.Provenance(lambda: w)
    assert prov.weights is None  # not downloaded yet
    w.write_bytes(b"now here")
    assert prov.weights["bytes"] == 8
    w.write_bytes(b"changed afterwards")
    assert prov.weights["bytes"] == 8  # read once
