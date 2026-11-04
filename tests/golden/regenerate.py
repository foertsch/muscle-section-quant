"""Regenerate a pipeline's golden with this repository's code, after a change that is meant to change its numbers.

    uv run python tests/golden/regenerate.py dystrophin --reason "C3: control animals by their held-out calls"

Keeps inputs.npz, the design and the list of compared files; reruns the pipeline the way its golden test does,
replaces the expected files whose content changed (the others keep their bytes) and golden.json's "produced" (with
the reason and the record it replaces). Prints which compared
files changed, so the commit can name them; its CHANGELOG entry must say which results change.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import platform
import shutil
import sys
import tempfile
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from msq.core import runinfo  # noqa: E402
from tests.golden.conftest import GOLDEN_ROOT, compare_golden  # noqa: E402


def run_dystrophin(golden: Path, work: Path) -> Path:
    from msq.dystrophin import run_pipeline as rp
    from tests.golden.test_golden_dystrophin import stage_run

    with pytest.MonkeyPatch.context() as mp:
        scans, run = stage_run(golden, work, mp)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = rp.run(scans, run)
    if code != 0:
        raise SystemExit(f"the run exited with status {code}; see {run / 'run.log'}")
    return run


RUNNERS = {"dystrophin": run_dystrophin}  # psr and necrosis join with their ports
PLACEHOLDER = "/tmp/golden"  # stands for the run's temp folder, which only appears in values the comparison ignores


def copy_expected(src: Path, dst: Path, tmp: str) -> None:
    """One expected file; in a text file the run's temp folder becomes PLACEHOLDER, so no machine's path is kept."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.suffix not in {".json", ".csv", ".txt"}:
        shutil.copy2(src, dst)
        return
    text = src.read_text(encoding="utf-8")
    for path in sorted({tmp, str(Path(tmp).resolve())}, key=len, reverse=True):
        text = text.replace(path, PLACEHOLDER)
    dst.write_text(text, encoding="utf-8")


def regenerate(pipeline: str, reason: str) -> list[str]:
    golden = GOLDEN_ROOT / pipeline
    meta = json.loads((golden / "golden.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="golden_") as tmp:
        run = RUNNERS[pipeline](golden, Path(tmp))
        changed = sorted({d.split(":", 1)[0] for d in compare_golden(golden, run)})
        for rel in changed:  # only these: the others are equal after the normaliser, so their bytes stay put
            copy_expected(run / rel, golden / "expected" / rel, tmp)
    previous = meta["produced"]
    meta["produced"] = {"date": date.today().isoformat(), "code_version": runinfo.code_version(),
                        "repo": "muscle-section-quant-lab", "platform": runinfo.platform_key(),
                        "python": platform.python_version(), "packages": runinfo.packages(), "reason": reason,
                        "replaces": {k: previous.get(k) for k in ("date", "code_version", "repo", "reason")
                                     if previous.get(k) is not None}}
    (golden / "golden.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return changed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pipeline", choices=sorted(RUNNERS))
    ap.add_argument("--reason", required=True, help="why the numbers change (the CHANGELOG entry)")
    args = ap.parse_args()
    changed = regenerate(args.pipeline, args.reason)
    print(f"{args.pipeline} golden regenerated; {len(changed)} compared file(s) changed:")
    for rel in changed:
        print(f"  {rel}")


if __name__ == "__main__":
    main()
