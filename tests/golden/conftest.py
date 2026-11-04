"""Layer 1 of the test harness: synthetic golden runs, and the normaliser their comparison goes through.

Each pipeline folder (tests/golden/<pipeline>/) was produced by that pipeline's old repository at v1.0.0
(tests/golden_run.py there) and holds inputs.npz (the synthetic scans and their labels), expected/ (every table and
JSON file the run wrote, as written) and golden.json (which files are compared, which fields and columns are
volatile, the generic group settings and how the run was made). The ported code must write the same files from the
same inputs; compare_golden() says where it does not.

What "the same" means: tables (.csv, .csv.gz) and JSON are parsed, so number formatting, column order and
indentation may change, but every value must be equal (exactly, on the platform that made the golden; NaN equals
NaN). On another platform (golden.json "produced" names it) numbers may differ by OTHER_PLATFORM_RTOL, relative:
another CPU and maths library round the last bits of a float differently, which changes no count, call or table row.
Volatile fields are dropped before the comparison: timestamps, durations, code and package versions, the weights'
checksum, host, device, paths (any key named in golden.json's volatile_keys, and any string that looks like a path).
Arrays in .npz files must be equal element by element. Images and HTML are never compared.

The goldens, and the baselines of the private real-data check, were written when the value a fiber is compared with
was still called "bar" (and, in dystrophin, "cutoff"). The pipelines now write "threshold". The old files stay as
they are; the expected side of every comparison is read through the names below, so the new output must equal the
old one value by value. The map renames columns, keys, file names and the listed wording, nothing else: no
tolerance, no column or key dropped.
"""

from __future__ import annotations

import gzip
import io
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from msq.core import runinfo

GOLDEN_ROOT = Path(__file__).resolve().parent
PIPELINES = ("dystrophin", "psr", "necrosis")

# dropped from every JSON object, whichever pipeline wrote it; golden.json may add to this
VOLATILE_KEYS = frozenset({
    "started", "finished", "paused", "updated", "heartbeat", "made", "date",
    "runtime_s", "seconds", "seconds_read", "seconds_total", "eta_s", "peak_memory_gb",
    "code_version", "software", "versions", "cellpose_weights", "weights", "host", "hostname", "device", "platform",
    "scans", "run_folder", "run", "source", "source_size", "path", "reference", "title", "img",
})
VOLATILE_COLUMNS = frozenset({"path", "source"})
PATH_LIKE = re.compile(r"^(?:[A-Za-z]:\\|/)|^\\\\|/Users/|/home/|\\Users\\")

# Old name -> name now written, for the rename of "bar" and "cutoff" to "threshold" (28 Oct 2026).
RENAMED_COLUMNS = {  # dystrophin sections, calls, animals, leave-one-animal-out; necrosis necrosis, sections, fibers
    "bar": "threshold",
    "bar_raw": "threshold_raw",
    "bar_from": "threshold_from",
    "sector_cutoff": "sector_threshold",
}
RENAMED_KEYS = {  # JSON: thresholds.json, run_settings.json, run_info.json, necrosis bar.json
    **RENAMED_COLUMNS,
    "bar_source": "threshold_source",
    "bar_coverage": "threshold_coverage",
    "bars": "thresholds",                                                  # run_settings.json
    "bar_percentile_of_controls": "threshold_percentile_of_controls",      # run_settings.json, "rule"
    "inverted_cutoff": "inverted_threshold",                               # measurement parameters
    "bright_cytoplasm_cutoff_bgsub": "bright_cytoplasm_threshold_bgsub",   # run_info.json
}
RENAMED_FILES = {"bars.csv": "thresholds.csv", "bar.json": "threshold.json"}  # necrosis calls/<variant>/, by file name
RENAMED_TEXT = (  # wording that compared files hold, each phrase exactly as the old code wrote it
    ("cutoff_decision.html", "threshold_decision.html"),  # run_settings.json: the reports written
    ("one bar per muscle = (1 - fpr) percentile of pooled control fibers' 18th brightest ring piece; positive at "
     "level c when at least c of the 36 pieces reach the bar",  # thresholds.json: rule
     "one threshold per muscle = (1 - fpr) percentile of pooled control fibers' 18th brightest ring piece; positive "
     "at level c when at least c of the 36 pieces reach the threshold"),
    ("borrowed bar", "borrowed threshold"),  # sections.csv, animals.csv: bar_from
    # messages in checks.json and run_settings.json (dystrophin) and checks.txt (necrosis)
    ("The bar set on the controls only fits scans", "The threshold set on the controls only fits scans"),
    ("but a bar is not borrowed between runs", "but a threshold is not borrowed between runs"),
    ("no untreated controls in this run; the bar (", "no untreated controls in this run; the threshold ("),
    ("or borrow the bar of an earlier run.", "or borrow the threshold of an earlier run."),
    ("; the bar rests on it alone", "; the threshold rests on it alone"),
    ("no cutoff decision report: it needs at least", "no threshold decision report: it needs at least"),
    ("the necrosis bar is set on them", "the necrosis threshold is set on them"),
)


def renamed_text(s: str) -> str:
    for old, new in RENAMED_TEXT:
        s = s.replace(old, new)
    return s


def renamed_path(rel: str) -> str:
    """The path of a file under the name it is written with now (the last part only)."""
    head, sep, name = rel.rpartition("/")
    return head + sep + RENAMED_FILES.get(name, name)


def current_file(folder: Path, rel: str) -> Path:
    """rel (a path under the names written now) in a run folder; the old name where the run was made before."""
    old = {new: o for o, new in RENAMED_FILES.items()}
    head, sep, name = rel.rpartition("/")
    before = folder / (head + sep + old.get(name, name))
    return folder / rel if (folder / rel).exists() or not before.exists() else before


OTHER_PLATFORM_RTOL = 1e-12  # relative (and absolute, near zero); Linux x86_64 differs from darwin-arm64 by about 1e-16


# ---------------------------------------------------------------- normalising


def looks_like_path(s: str) -> bool:
    return bool(PATH_LIKE.search(s))


def normalise_json(obj, volatile_keys: frozenset[str] = VOLATILE_KEYS):
    """The JSON object without its volatile keys, path-like strings replaced by '<path>'."""
    if isinstance(obj, dict):
        return {k: normalise_json(v, volatile_keys) for k, v in obj.items() if k not in volatile_keys}
    if isinstance(obj, list):
        return [normalise_json(v, volatile_keys) for v in obj]
    if isinstance(obj, str) and looks_like_path(obj):
        return "<path>"
    return obj


def renamed_json(obj):
    """obj with the keys and wording of the map above under their present names."""
    if isinstance(obj, dict):
        out = {RENAMED_KEYS.get(k, k): renamed_json(v) for k, v in obj.items()}
        assert len(out) == len(obj), "the rename map merged two keys"
        return out
    if isinstance(obj, list):
        return [renamed_json(v) for v in obj]
    return renamed_text(obj) if isinstance(obj, str) else obj


def renamed_table(df: pd.DataFrame) -> pd.DataFrame:
    """df with the columns and the cell wording of the map above under their present names, columns in name order."""
    df = df.rename(columns=RENAMED_COLUMNS)
    assert df.columns.is_unique, "the rename map merged two columns"
    for c in df.columns:
        if any(isinstance(v, str) and renamed_text(v) != v for v in df[c]):
            df[c] = df[c].map(lambda v: renamed_text(v) if isinstance(v, str) else v)
    return df[sorted(df.columns)]


def read_table(path: Path, volatile_columns: frozenset[str] = VOLATILE_COLUMNS, renamed: bool = False) -> pd.DataFrame:
    """A CSV (plain or gzip) without its volatile columns, columns in name order. renamed: under present names."""
    data = path.read_bytes()
    if path.suffix == ".gz":
        data = gzip.decompress(data)
    df = pd.read_csv(io.BytesIO(data))
    df = df.drop(columns=[c for c in df.columns if c in volatile_columns])
    return renamed_table(df) if renamed else df[sorted(df.columns)]


def read_json(path: Path, volatile_keys: frozenset[str] = VOLATILE_KEYS, renamed: bool = False):
    obj = normalise_json(json.loads(path.read_text(encoding="utf-8")), volatile_keys)
    return renamed_json(obj) if renamed else obj


def read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


# ---------------------------------------------------------------- comparing


def _same_scalar(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return (math.isnan(a) and math.isnan(b)) or a == b
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    return a == b


def json_diffs(expected, actual, where: str = "$", rtol: float = 0.0) -> list[str]:
    """Every place the two normalised JSON objects differ, as 'key path: what'. rtol: numbers that close are equal."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        out = []
        for k in sorted(set(expected) | set(actual)):
            if k not in actual:
                out.append(f"{where}.{k}: missing")
            elif k not in expected:
                out.append(f"{where}.{k}: unexpected")
            else:
                out += json_diffs(expected[k], actual[k], f"{where}.{k}", rtol)
        return out
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f"{where}: {len(expected)} items expected, {len(actual)} found"]
        return [d for i, (e, a) in enumerate(zip(expected, actual)) for d in json_diffs(e, a, f"{where}[{i}]", rtol)]
    if type(expected) is not type(actual) and not (isinstance(expected, (int, float)) and isinstance(actual, (int, float))
                                                   and not isinstance(expected, bool) and not isinstance(actual, bool)):
        return [f"{where}: {type(expected).__name__} expected, {type(actual).__name__} found"]
    if not _same_scalar(expected, actual):
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            if rtol and math.isclose(expected, actual, rel_tol=rtol, abs_tol=rtol):
                return []
            return [f"{where}: differs by {abs(expected - actual):.6g}"]
        return [f"{where}: differs"]
    return []


def table_diffs(expected: pd.DataFrame, actual: pd.DataFrame, rtol: float = 0.0) -> list[str]:
    """Where two tables differ: columns, row count, then per column the number of differing cells and, for numbers,
    the largest absolute difference. rtol: numbers that close are equal."""
    out = []
    missing, extra = set(expected.columns) - set(actual.columns), set(actual.columns) - set(expected.columns)
    if missing:
        out.append(f"columns missing: {', '.join(sorted(missing))}")
    if extra:
        out.append(f"columns unexpected: {', '.join(sorted(extra))}")
    if len(expected) != len(actual):
        out.append(f"{len(expected)} rows expected, {len(actual)} found")
        return out
    for c in sorted(c for c in expected.columns if c in actual.columns):
        e, a = expected[c], actual[c]
        if pd.api.types.is_numeric_dtype(e) and pd.api.types.is_numeric_dtype(a) and \
                not pd.api.types.is_bool_dtype(e) and not pd.api.types.is_bool_dtype(a):
            ev, av = e.to_numpy(dtype=float), a.to_numpy(dtype=float)
            same = (ev == av) | (np.isnan(ev) & np.isnan(av))
            if rtol:
                same |= np.isclose(ev, av, rtol=rtol, atol=rtol)
            if not same.all():
                finite = np.isfinite(ev) & np.isfinite(av)
                largest = np.abs(ev - av)[finite & ~same].max() if (finite & ~same).any() else float("nan")
                out.append(f"{c}: {int((~same).sum())} of {len(same)} cells differ, largest difference {largest:.6g}")
        else:
            ee, aa = e.astype(object).where(e.notna(), None), a.astype(object).where(a.notna(), None)
            same = (ee.to_numpy() == aa.to_numpy())
            if not same.all():
                out.append(f"{c}: {int((~same).sum())} of {len(same)} cells differ")
    return out


def npz_diffs(expected: dict[str, np.ndarray], actual: dict[str, np.ndarray], rtol: float = 0.0) -> list[str]:
    out = []
    for k in sorted(set(expected) | set(actual)):
        if k not in actual:
            out.append(f"{k}: missing")
        elif k not in expected:
            out.append(f"{k}: unexpected")
        elif expected[k].shape != actual[k].shape:
            out.append(f"{k}: shape {expected[k].shape} expected, {actual[k].shape} found")
        elif not (np.array_equal(expected[k], actual[k], equal_nan=np.issubdtype(expected[k].dtype, np.floating))
                  or (rtol and np.allclose(expected[k], actual[k], rtol=rtol, atol=rtol,
                                           equal_nan=True))):
            e, a = expected[k].astype(float), actual[k].astype(float)
            out.append(f"{k}: {int((e != a).sum())} of {e.size} elements differ, largest difference "
                       f"{np.nanmax(np.abs(e - a)):.6g}")
    return out


def kind(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".csv") or name.endswith(".csv.gz"):
        return "table"
    if name.endswith(".json"):
        return "json"
    if name.endswith(".npz"):
        return "npz"
    return "text"


def file_diffs(expected: Path, actual: Path, volatile_keys: frozenset[str] = VOLATILE_KEYS,
               volatile_columns: frozenset[str] = VOLATILE_COLUMNS, actual_renamed: bool = False,
               rtol: float = 0.0) -> list[str]:
    """Differences between one expected and one actual output file, after normalising both. The expected file is read
    through the rename map (it may have been written before the rename); the actual one is, only with actual_renamed
    (a file that was written before the rename too, as when a golden is compared with itself)."""
    if not actual.exists():
        return ["missing"]
    k = kind(expected)
    if k == "table":
        return table_diffs(read_table(expected, volatile_columns, renamed=True),
                           read_table(actual, volatile_columns, renamed=actual_renamed), rtol)
    if k == "json":
        return json_diffs(read_json(expected, volatile_keys, renamed=True),
                          read_json(actual, volatile_keys, renamed=actual_renamed), rtol=rtol)
    if k == "npz":
        return npz_diffs(read_npz(expected), read_npz(actual), rtol)
    e = [renamed_text(x) for x in expected.read_text(encoding="utf-8").splitlines()]
    a = [renamed_text(x) if actual_renamed else x for x in actual.read_text(encoding="utf-8").splitlines()]
    return [] if e == a else [f"{sum(x != y for x, y in zip(e, a)) + abs(len(e) - len(a))} lines differ"]


def golden_meta(golden_dir: Path) -> dict:
    return json.loads((golden_dir / "golden.json").read_text(encoding="utf-8"))


def compare_golden(golden_dir: Path, actual_dir: Path, actual_renamed: bool = False) -> list[str]:
    """Every compared file of a golden folder against the same path under actual_dir (under its present name, see
    RENAMED_FILES): a list of 'file: difference' lines, empty when the run reproduces the golden. actual_renamed:
    actual_dir was written before the rename too (a golden against itself): old names, read through the map."""
    meta = golden_meta(golden_dir)
    volatile_keys = VOLATILE_KEYS | frozenset(meta.get("volatile_keys", []))
    volatile_columns = VOLATILE_COLUMNS | frozenset(meta.get("volatile_columns", []))
    rtol = 0.0 if meta.get("produced", {}).get("platform") == runinfo.platform_key() else OTHER_PLATFORM_RTOL
    out = []
    for rel in meta["compare"]:
        now = rel if actual_renamed else renamed_path(rel)
        for d in file_diffs(golden_dir / "expected" / rel, actual_dir / now, volatile_keys, volatile_columns,
                            actual_renamed, rtol):
            out.append(f"{now}: {d}")
    return out


def assert_matches_golden(golden_dir: Path, actual_dir: Path) -> None:
    diffs = compare_golden(golden_dir, actual_dir)
    assert not diffs, f"{len(diffs)} difference(s) from the {golden_dir.name} golden:\n" + "\n".join(diffs[:40])


# ---------------------------------------------------------------- fixtures


@pytest.fixture
def golden_dirs() -> dict[str, Path]:
    """The golden folder of every pipeline that has one."""
    return {p: GOLDEN_ROOT / p for p in PIPELINES if (GOLDEN_ROOT / p / "golden.json").exists()}


@pytest.fixture
def check_golden():
    return assert_matches_golden
