"""Build the click review page: one self-contained HTML file to open in a browser (no server, no internet needed
beyond the web fonts, which fall back to system fonts).

Takes the windows written by click_windows.py (windows.json, wNN.jpg, wNN_outlines.png) and embeds them into
click_review_template.html. Reviewers mark missed, merged, split and non-fiber outlines; marks stay in the
browser until they press "Download results", which saves fiber_outline_check_<initials>_<date>.json.

The review is done by eye, by one reviewer. The reviewer's results are not stored in the repository.

Usage:
    python -m msq.psr.build_click_report WINDOWS_DIR --out REPORT.html [--run NAME]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from msq.core import report
from msq.core.report.images import data_uri
from msq.psr import paths

TEMPLATE = paths.CODE / "click_review_template.html"


def build(windows_dir: Path, out: Path, run: str) -> Path:
    windows = json.loads((windows_dir / "windows.json").read_text(encoding="utf-8"))
    images = {w["id"]: {"base": data_uri(windows_dir / f"{w['id']}.jpg", "image/jpeg"),
                        "outlines": data_uri(windows_dir / f"{w['id']}_outlines.png", "image/png")}
              for w in windows}
    return report.render(TEMPLATE, out, data={"run": run, "windows": windows, "images": images}, slots={"RUN": run})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("windows_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--run", default=None, help="run name shown on the page and in the results (default: folder)")
    args = ap.parse_args()
    out = build(args.windows_dir, args.out, args.run or args.windows_dir.parent.name)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
