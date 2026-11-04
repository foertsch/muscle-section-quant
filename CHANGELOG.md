# Changelog

## 0.1.0 (2026-11-04)

First public release: the shared core (`msq.core`) and the dystrophin assay (`msq.dystrophin`).

- `uv run msq-dystrophin` takes a folder of ND2 scans to tables, figures and interactive reports; `uv run msq-dystrophin-app` runs it from a local browser page.
- Control animals are reported by their held-out calls (each against the threshold from the other control animals).
- The value a fiber is compared with is called the threshold in the tables (`threshold`, `threshold_from`, `thresholds.json`), the run record and the reports (`reports/threshold_decision.html`).
- Every run, and every section in it, records the code version and the Cellpose weights that made it.
- A synthetic demo (`demo/run_demo.py`) and a synthetic golden run in the tests.
- `demo/make_crops.py` draws method crops (segmentation outlines, one fiber's ring sectors, the calls) from a finished run.
