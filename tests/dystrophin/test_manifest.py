import sys
from pathlib import Path

import pandas as pd
import pytest

from msq.dystrophin import manifest as mf  # noqa: E402


@pytest.mark.parametrize("path, expected", [
    ("scans/1104 TA/0001 1104 L TA sec 2 20xfull.nd2", ("treated", "TA", "1104", 2, "20x", False)),
    ("scans/1104 QUAD/Edited sec1 QUAD 1104 20xfull.nd2", ("treated", "QUAD", "1104", 1, "20x", True)),
    ("scans/Neg-ctrl-TA/1003/0001 1106 L TA sec1 20xfull.nd2", ("ctrl", "TA", "1106", 1, "20x", False)),
    ("scans/Neg-ctrl-QUAD/1107/0001 1108 L QUAD sec 3 4x.nd2", ("ctrl", "QUAD", "1108", 3, "4x", False)),
    ("scans/Neg-ctrl-QUAD/1107/0001 1107 L QUAD sec 3 4xl.nd2", ("ctrl", "QUAD", "1107", 3, "4x", False)),
    ("scans/Neg-ctrl-TA/1107/0001 1107 L TA sec2 4x.nd2", ("ctrl", "TA", "1107", 2, "4x", False)),
])
def test_parse_handles_naming_variants(path, expected):
    r = mf.parse(Path(path))
    assert (r["group"], r["muscle"], r["animal"], r["section"], r["mag"], r["edited"]) == expected


def test_lower_exposure_is_noted():
    r = mf.parse(Path("scans/Neg-ctrl-TA/1108/0001 1108 L TA sec 1 20xfull_lower_exposure.nd2"))
    assert r["note"] == "lower_exposure" and r["mag"] == "20x" and r["animal"] == "1108"


def test_sample_id_keeps_qc_name_parts():
    r = mf.parse(Path("scans/1105 QUAD/0001 1105 L QUAD sec 3 20xfull.nd2"))
    assert mf.sample_id(r) == "QUAD_sec3_1105_20x_treated"


def test_main_writes_the_project_manifest(tmp_path, monkeypatch, capsys):
    d = tmp_path / "1104 TA"
    d.mkdir()
    for n in ["Edited sec3 TA 1104 20xfull.nd2", "0001 1104 L TA sec 3 20xfull.nd2",
              "0001 1104 L TA sec 2 20xfull.nd2", "0001 1104 L TA sec 2 4x.nd2"]:
        (d / n).write_bytes(b"")
    out = tmp_path / "manifest_20x.csv"
    monkeypatch.setattr(sys, "argv", ["manifest.py", str(tmp_path), "--out", str(out)])
    mf.main()
    df = pd.read_csv(out)
    assert list(df.columns) == mf.MANIFEST_COLUMNS
    assert list(df["sample"]) == ["TA_sec2_1104_20x_treated", "TA_sec3_1104_20x_treated"]
    assert not df["edited"].any()  # the raw file wins over the edited copy
    assert "2 files describe this section" in capsys.readouterr().out


def test_underscores_read_as_spaces():
    r = mf.parse(Path("scans/1104 TA/TA_sec2_1104_20x.nd2"))
    assert mf.sample_id(r) == "TA_sec2_1104_20x_treated"
    assert mf.animal_numbers(Path("x/TA_sec2_1104_20x_1500.nd2")) == ["1104", "1500"]


def test_group_comes_from_folders_only():
    assert mf.parse(Path("scans/Neg-ctrl-TA/1107/TA sec1 1107 20x.nd2"))["group"] == "ctrl"
    assert mf.parse(Path("scans/1104 TA/Neg-ctrl TA sec1 1104 20x.nd2"))["group"] == "treated"  # file name, not folder
    root = Path("lab/Neg-ctrl comparison/batch2")
    assert mf.parse(root / "1104 TA/TA sec2 1104 20x.nd2", root)["group"] == "treated"
    assert mf.parse(root / "1104 TA/TA sec2 1104 20x.nd2")["group"] == "ctrl"  # without the root, every folder counts


def test_unparseable_name_raises():
    with pytest.raises(ValueError):
        mf.parse(Path("scans/misc/overview.nd2"))
