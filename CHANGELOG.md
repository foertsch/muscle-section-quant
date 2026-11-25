# Changelog

## 0.2.0 (2026-11-25)

Adds the picrosirius red (PSR) fibrosis assay on the shared core. Dystrophin results are unchanged.

- `msq.psr`: whole-section PSR scans to tables per section and animal. The primary readout is collagen outside the fiber outlines (`collagen_outside_fraction`); `red_fraction`, fiber size, fiber count and density are in the tables.
- `uv run msq-psr SCANS RUN --cohort NAME` runs it; cohorts are defined in `cohorts.local.toml`, copied from `cohorts.example.toml`.
- R figure and export drivers (`R/figures.R`, `R/export.R`): Dunnett's test of each group against control on animal means, with no ANOVA first.
- PSR statistics differ from earlier private runs: no ANOVA gate, and the primary readout is collagen outside the fiber outlines.
- A synthetic PSR demo (`demo/run_psr_demo.py`), a PSR mode in `demo/make_crops.py`, and synthetic golden runs for the PSR assay.
- The R tests run in CI, and both demos.
- Limits are stated in the README: the readout depends on the Cellpose-SAM outlines, the glass-colour division assumes a linear camera (checked on synthetic images only), nothing is masked, and no section is excluded.

## 0.1.0 (2026-11-04)

First public release: the shared core (`msq.core`) and the dystrophin assay (`msq.dystrophin`).

- `uv run msq-dystrophin` takes a folder of ND2 scans to tables, figures and interactive reports; `uv run msq-dystrophin-app` runs it from a local browser page.
- Control animals are reported by their held-out calls (each against the threshold from the other control animals).
- The value a fiber is compared with is called the threshold in the tables (`threshold`, `threshold_from`, `thresholds.json`), the run record and the reports (`reports/threshold_decision.html`).
- Every run, and every section in it, records the code version and the Cellpose weights that made it.
- A synthetic demo (`demo/run_demo.py`) and a synthetic golden run in the tests.
- `demo/make_crops.py` draws method crops (segmentation outlines, one fiber's ring sectors, the calls) from a finished run.
