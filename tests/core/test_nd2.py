import pytest

from msq.core.nd2 import acquired, channel_index, channel_settings, to_ms
from msq.core.synthetic.necrosis import DESCRIPTION


def test_channel_index_by_name_not_position():
    assert channel_index(["GFP", "DAPI", "Cy5"], "Cy5") == 2
    assert channel_index(["DAPI", "GFP", "Cy5"], "gfp") == 1
    with pytest.raises(ValueError, match="exactly one channel named 'TRITC'"):
        channel_index(["DAPI", "Cy5"], "TRITC")
    with pytest.raises(ValueError):
        channel_index(["GFP", "GFP"], "GFP")


def test_channel_settings_are_the_raw_strings_of_one_plane():
    assert channel_settings(DESCRIPTION, "Cy5") == {"exposure": "2 s", "gain": "1.0x", "light": "10.0"}
    assert channel_settings(DESCRIPTION, "gfp")["exposure"] == "500 ms"
    assert channel_settings(DESCRIPTION, "TRITC") == {}
    assert channel_settings("no planes here", "GFP") == {}
    indented = DESCRIPTION.replace("\nPlane #", "\n  Plane #")
    assert channel_settings(indented, "DAPI") == channel_settings(DESCRIPTION, "DAPI")
    assert channel_settings(DESCRIPTION, "Cy5", {"binning": "Binning"}) == {"binning": "1.0x1.0"}


def test_exposure_units_and_acquisition_time():
    assert to_ms("500 ms") == 500.0 and to_ms("2 s") == 2000.0 and to_ms("1.5 min") == 90000.0 and to_ms("7") == 7.0
    with pytest.raises(ValueError):
        to_ms("fast")
    assert acquired("23.07.2026  13:46:15") == "2026-07-23 13:46:15"
    assert acquired("01.01.2026  12:00:00", "%Y-%m-%d %H:%M") == "2026-01-01 12:00"
    assert acquired("  ") is None
