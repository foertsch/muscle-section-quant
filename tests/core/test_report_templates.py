"""The report templates, two per assay package (dystrophin compare and decision, necrosis threshold and section viewer,
psr overlay and click review; the public tree has only dystrophin's), carry the shared assets and one data marker, and
their page scripts parse together with base.js."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

import msq
from msq.core.report import render
from msq.core.report.render import ASSETS, DATA_MARKER

TEMPLATES = sorted(Path(msq.__file__).parent.glob("*/*_template.html"))
IDS = [f"{t.parent.name}/{t.stem}" for t in TEMPLATES]


def test_each_assay_package_has_its_two_templates():
    assays = {t.parent.name for t in TEMPLATES}
    assert "dystrophin" in assays
    for assay in assays:
        assert sum(t.parent.name == assay for t in TEMPLATES) == 2, assay


@pytest.mark.parametrize("template", TEMPLATES, ids=IDS)
def test_template_carries_the_asset_markers_and_one_data_marker(template):
    text = template.read_text(encoding="utf-8")
    assert "<!--ASSET base.css-->" in text and "<!--ASSET base.js-->" in text
    assert text.count(f"const DATA = {DATA_MARKER};\n") == 1 and text.count(DATA_MARKER) == 1
    for name in re.findall(r"<!--ASSET ([\w.-]+)-->", text):
        assert (ASSETS / name).is_file(), name
    assert not re.search(r"const (\$|fmt|seg|tip|pctl|countGE|niceStep|rowHist|PanZoom|persist) =", text.split("<!--ASSET base.js-->")[1])


def rendered(template: Path, tmp_path: Path) -> str:
    slots = {name: "x" for name in re.findall(r"<!--([A-Z]+)-->", template.read_text(encoding="utf-8"))}
    return render(template, tmp_path / "page.html", slots=slots).read_text(encoding="utf-8")


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
@pytest.mark.parametrize("template", TEMPLATES, ids=IDS)
def test_page_scripts_parse_together_with_the_shared_script(template, tmp_path):
    """One file per page: a name the page declares again that base.js declares is a SyntaxError in the browser."""
    scripts = re.findall(r"<script>\n?(.*?)</script>", rendered(template, tmp_path), re.S)
    assert any("function rowHist" in s for s in scripts)
    f = tmp_path / "page.js"
    f.write_text("\n".join(scripts), encoding="utf-8")
    r = subprocess.run(["node", "--check", str(f)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize("template", TEMPLATES, ids=IDS)
def test_rendered_page_has_no_marker_left(template, tmp_path):
    html = rendered(template, tmp_path)
    assert "<!--ASSET" not in html and not re.search(r"<!--[A-Z]+-->", html)
