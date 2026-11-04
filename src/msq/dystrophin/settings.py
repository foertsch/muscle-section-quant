"""What the pipeline accepts and the rule it applies, in one place.

The checks (checks.py), the sample table (manifest.py), the measurement (measure.py) and the calls
(full_summary.py) read these values, so changing the scope here changes all of them. Segmentation and
measurement parameters stay in measure.Params.
"""

from msq.core.groups import Groups

MUSCLES = ("QUAD", "TA")            # muscle names read from file names; also the plotting order
DYSTROPHIN_CHANNEL = "GFP"          # ND2 channel names (the channel order differs between files)
ACTIN_CHANNEL = "Cy5"
MAGNIFICATION = "20x"               # scans of other magnifications are skipped
PIXEL_UM_RANGE = (0.33, 0.40)       # 20x scans are 0.3606 µm/px; outside this range is another objective
CONTROL_FOLDER_PREFIX = "neg-ctrl"  # scans in a folder whose name starts with this are untreated controls

FPR = 0.02                          # threshold = 98th percentile of pooled control fibers' score at CALL_LEVEL
CALL_LEVEL = 0.5                    # a fiber is positive when at least half of its ring reaches the threshold
CALL_COLUMN = f"pct_positive_{round(100 * CALL_LEVEL)}"  # the call's column in sections.csv and animals.csv
# The tables also hold the 60 and 70% levels (full_summary.COVERAGES), and both report templates assume the three
# levels 50 / 60 / 70% (18 / 22 / 26 of 36 ring pieces): changing CALL_LEVEL needs those changed too.
GROUPS = Groups(order=("ctrl", "treated"), control="ctrl",  # the two groups: untreated controls and treated
                labels={"ctrl": "negative control (mdx)", "treated": "treated"},  # in figures
                colours={"ctrl": "#898781", "treated": "#2a78d6"},  # controls grey, treated blue
                animal=r"\b(1\d{3})\b")  # an animal is a number from 1000 to 1999 in the file name
GROUP_NAMES = {"ctrl": "control", "treated": "treated"}  # the two groups as people read them
# The reference Cellpose weights, the ones the lab's runs were made with: cpsam_v2, 1,233,586,851 bytes, downloaded on
# 16 Oct 2026. Other weights give a warning, and a threshold is not borrowed between runs with different weights.
REFERENCE_WEIGHTS_SHA256 = "0f1cc3f7ecdd8a037a57c6c48d9d8921391be4cbce3fa9f13c3e3a2e1253c667"
MIN_CONTROL_ANIMALS = 2             # fewer control animals: warn, and no held-out check is possible

# A run made before the rename of "bar" and "cutoff" to "threshold" (28 Oct 2026) wrote the old keys in its
# thresholds.json and run_settings.json; the code that lets a muscle borrow such a run's threshold reads them as the
# new ones.
OLD_KEYS = {"bar": "threshold", "bar_source": "threshold_source", "bar_coverage": "threshold_coverage",
            "sector_cutoff": "sector_threshold", "inverted_cutoff": "inverted_threshold"}


def current_keys(obj):
    """A loaded JSON object with every key of OLD_KEYS under its present name (nested objects too)."""
    if isinstance(obj, dict):
        return {OLD_KEYS.get(k, k): current_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [current_keys(v) for v in obj]
    return obj
