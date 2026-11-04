"""The golden regenerator replaces only the expected files whose content changed."""

import json
import shutil

from tests.golden import regenerate
from tests.golden.conftest import GOLDEN_ROOT, compare_golden


def test_regenerate_rewrites_only_changed_files_and_records_why(tmp_path, monkeypatch):
    golden = tmp_path / "dystrophin"
    shutil.copytree(GOLDEN_ROOT / "dystrophin", golden)
    animals = golden / "expected" / "tables" / "animals.csv"
    animals.write_text(animals.read_text().replace(",ctrl,", ",ctrl_x,", 1))  # one changed file
    samples = golden / "expected" / "samples.csv"
    before = samples.read_bytes()
    monkeypatch.setattr(regenerate, "GOLDEN_ROOT", tmp_path)
    changed = regenerate.regenerate("dystrophin", "test: a planted change")
    assert changed == ["tables/animals.csv"]
    assert samples.read_bytes() == before  # equal after the normaliser: untouched
    assert compare_golden(golden, golden / "expected", actual_renamed=True) == []  # old and regenerated names mixed
    meta = json.loads((golden / "golden.json").read_text())
    assert meta["produced"]["reason"] == "test: a planted change" and meta["produced"]["replaces"]["reason"]


def test_a_regenerated_file_keeps_no_temp_path(tmp_path):
    tmp = str(tmp_path / "golden_abc")
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "run_settings.json").write_text(json.dumps({"scans": f"{tmp}/scans", "fpr": 0.02}))
    (tmp_path / "run" / "calls.csv.gz").write_bytes(b"\x1f\x8b binary")
    out = tmp_path / "expected"
    regenerate.copy_expected(tmp_path / "run" / "run_settings.json", out / "run_settings.json", tmp)
    regenerate.copy_expected(tmp_path / "run" / "calls.csv.gz", out / "calls.csv.gz", tmp)
    assert json.loads((out / "run_settings.json").read_text()) == {"scans": "/tmp/golden/scans", "fpr": 0.02}
    assert (out / "calls.csv.gz").read_bytes() == b"\x1f\x8b binary"
    shipped = [*GOLDEN_ROOT.joinpath("dystrophin", "expected").rglob("*.json"),
               *GOLDEN_ROOT.joinpath("dystrophin", "expected").rglob("*.csv")]
    assert shipped and all("/var/folders" not in p.read_text(encoding="utf-8") for p in shipped)
