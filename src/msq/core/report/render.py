"""Fill a report template: the page's data at its one marker, text at its slots, the shared assets inlined, so the
page is one self-contained file.

A template holds `const DATA = /*DATA*/null;` (the page's data, written as one JSON line), `<!--NAME-->` slots for
text (HTML-escaped) and `<!--ASSET name-->` where a shared stylesheet or script goes (base.css, lightbox.css,
base.js; see assets/)."""

from __future__ import annotations

import json
import re
from html import escape
from pathlib import Path

ASSETS = Path(__file__).parent / "assets"
DATA_MARKER = "/*DATA*/null"
_ASSET = re.compile(r"<!--ASSET ([\w.-]+)-->")
_WRAP = {".css": "<style>\n{}</style>", ".js": "<script>\n{}</script>"}


def render(template: Path, out: Path, data=None, slots: dict[str, str] | None = None,
           assets: dict[str, str | Path] | None = None) -> Path:
    """Write the template to out with its assets, slots and data filled in; returns out.

    data: any JSON-able value, put at the data marker (compact JSON, which the page reads as `const DATA`).
    slots: name -> text for `<!--NAME-->`, every occurrence.
    assets: name -> text or file, for assets the package does not ship (a page's own shared code) or to replace one."""
    html = Path(template).read_text(encoding="utf-8")
    extra = assets or {}

    def inline(m: re.Match) -> str:
        name = m.group(1)
        src = extra.get(name, ASSETS / name)
        text = src if isinstance(src, str) else Path(src).read_text(encoding="utf-8")
        return _WRAP[Path(name).suffix].format(text.rstrip("\n") + "\n")

    html = _ASSET.sub(inline, html)
    for name, text in (slots or {}).items():
        if f"<!--{name}-->" not in html:
            raise ValueError(f"template has no <!--{name}-->")
        html = html.replace(f"<!--{name}-->", escape(text))
    if data is not None:
        if html.count(DATA_MARKER) != 1:
            raise ValueError(f"template needs exactly one {DATA_MARKER}")
        html = html.replace(DATA_MARKER, json.dumps(data, separators=(",", ":")))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out
