"""Per-fiber dystrophin on immunofluorescence whole sections.

Fibers are segmented with Cellpose-SAM on the actin channel (Cy5) after 4x block-mean downsampling (measure.py);
dystrophin (GFP) is measured at full resolution in 36 ring sectors around each fiber's edge; each muscle's threshold is
the 98th percentile of the pooled control fibers' 50% score and a fiber is positive when half its ring reaches it
(calibrate.py, full_summary.py). run_pipeline.py takes a folder of scans to a run folder; app.py drives it from a
browser on this computer.
"""
