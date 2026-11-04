"""Layer 1 for dystrophin: msq.dystrophin reproduces tests/golden/dystrophin/expected/ from its inputs.npz.

The run is made exactly as golden.json's "how" says the golden was made by dystrophin-quant v1.0.0: a fake scan
folder from the design, every section's labels_ds4.tif written into the run folder beforehand (so Cellpose never
runs), the ND2 reader and header returning the stored arrays, no GPU probe, no weights file, and the two HTML report
builders replaced by no-ops (reports are not compared).
"""

from pathlib import Path

import numpy as np
import pytest
import tifffile

from msq.dystrophin import checks
from msq.dystrophin import measure as dq
from msq.dystrophin import run_pipeline as rp

HEADER_GFP = {"exposure": "2 s", "gain": "1.0x", "light": "10.0"}  # the same for every scan, as in the golden run


def stage_run(golden: Path, root: Path, monkeypatch) -> tuple[Path, Path]:
    """The scan folder and the run folder of the golden design, with the stubs the golden run used."""
    import json

    meta = json.loads((golden / "golden.json").read_text(encoding="utf-8"))
    with np.load(golden / "inputs.npz") as z:
        arrays = {k: z[k] for k in z.files}
    scans, run = root / "scans", root / "run"
    by_path = {}
    for d in meta["design"]:
        p = root / d["scan"]  # "scans/<group folder>/<animal>/<file>"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")
        by_path[p.resolve()] = d["sample"]
        (run / "sections" / d["sample"]).mkdir(parents=True)
        tifffile.imwrite(run / "sections" / d["sample"] / "labels_ds4.tif", arrays[f"{d['sample']}/labels_ds4"],
                         compression="zlib")
    px_um, sat, channels = float(arrays["px_um"]), int(arrays["sat"]), [str(c) for c in arrays["channels"]]

    def read_nd2(path):
        s = by_path[Path(path).resolve()]
        return {"gfp": arrays[f"{s}/gfp"], "act": arrays[f"{s}/act"], "px_um": px_um, "sat": sat, "channels": channels}

    def nd2_header(path):
        gfp = arrays[f"{by_path[Path(path).resolve()]}/gfp"]
        return {"sizes": {"C": 3, "Y": gfp.shape[0], "X": gfp.shape[1]}, "channels": channels, "px_um": px_um,
                "gfp": HEADER_GFP}

    monkeypatch.setattr(dq, "read_nd2", read_nd2)
    monkeypatch.setattr(checks, "nd2_header", nd2_header)
    monkeypatch.setattr(rp, "gpu_device", lambda: "synthetic golden run (no GPU)")
    monkeypatch.setattr(rp, "weights_file", lambda: root / "no weights here")
    monkeypatch.setattr(rp.dr, "build", lambda results, tables, nd2, muscles, out: None)
    monkeypatch.setattr(rp.cr, "build", lambda results, tables, nd2, muscle, out: None)
    return scans, run


def test_dystrophin_reproduces_its_golden(golden_dirs, check_golden, tmp_path, monkeypatch, capsys):
    if "dystrophin" not in golden_dirs:
        pytest.skip("no dystrophin golden")
    golden = golden_dirs["dystrophin"]
    scans, run = stage_run(golden, tmp_path, monkeypatch)
    assert rp.run(scans, run) == 0, (run / "run.log").read_text(encoding="utf-8")[-3000:]
    check_golden(golden, run)


def test_a_changed_number_fails_the_golden(golden_dirs, tmp_path, monkeypatch, capsys):
    """The comparison is not vacuous: one setting moved by a little changes the tables and the golden says so."""
    from tests.golden.conftest import compare_golden

    if "dystrophin" not in golden_dirs:
        pytest.skip("no dystrophin golden")
    golden = golden_dirs["dystrophin"]
    scans, run = stage_run(golden, tmp_path, monkeypatch)
    monkeypatch.setattr(rp.fs, "FPR", 0.03)  # the threshold at the 97th instead of the 98th percentile
    monkeypatch.setattr(rp.settings, "FPR", 0.03)
    assert rp.run(scans, run) == 0
    diffs = compare_golden(golden, run)
    assert any(d.startswith("tables/thresholds.json") for d in diffs)
    assert any(d.startswith("tables/animals.csv") for d in diffs)
