# Dystrophin per fiber: method

How `msq.dystrophin` turns a whole-section scan into one call per fiber and one value per animal. Every number below is a setting in the code (`src/msq/dystrophin/settings.py`, `Params` in `src/msq/dystrophin/measure.py`); some were tuned on the lab's first dataset (see the README's "Status and limits").

## Input

Stitched Nikon ND2 scans at 20x (0.33 to 0.40 µm per pixel), one image with channels found by name: dystrophin in `GFP`, actin in `Cy5`. The checks (`checks.py`) stop a run before it starts when a file does not open, holds more than one image, lacks either channel or has another pixel size, and warn when the scans of one muscle differ in GFP exposure, camera gain or light intensity.

## Tissue and background

The actin image is block-mean downsampled 4x (to about 1.44 µm per pixel). Tissue is the smoothed actin above an Otsu threshold on the log scale, holes filled, pieces smaller than 1% of the largest dropped. The main section is the largest tissue piece after bridging gaps of up to 50 µm, so neighbouring sections on the slide and torn-off fragments are left out. The background of each channel is the median of non-zero pixels about 50 µm or more from the tissue (the margin is grown as a diamond, so 50 µm along the axes and less on the diagonals), which is camera offset plus slide; zero pixels (blank padding in edited scans) count as neither tissue nor background.

## Segmentation

Cellpose-SAM (cellpose 4.2, weights `cpsam_v2`, named in the code) on the downsampled actin image, flow threshold 0.4, cell-probability threshold 0. The weights' size and SHA-256 are recorded with every run; other weights than the reference ones give a warning, and a threshold is not borrowed between runs whose recorded weights differ (a run without a record lends it with a warning). A segmentation saved in the run folder is reused, so a stopped run does not segment again.

## Measurement per fiber

At full resolution, around each fiber's outline:

- **Band**: pixels within 1.5 µm of the outline, inside and outside.
- **Ring pieces**: the band is split into 36 angular sectors around the fiber's centroid. Each piece's value is the 90th percentile of the GFP band pixels (after a Gaussian smoothing of 1 pixel) minus the fiber's own cytoplasm. The percentile keeps the membrane's ridge rather than the band's mean.
- **Cytoplasm**: the mean GFP of the fiber's pixels deeper than 3 µm from the outline (at most half the fiber's depth). Subtracting it removes the fiber's own background and haze.
- **Membrane-specific dystrophin**: the median ring piece, so half the ring lies above it.

Also recorded per fiber: area, minimum Feret diameter, actin in the core and at the edge, saturation shares, the spread of the ring pieces, and the centroid.

## Which fibers count

An object is left out when its area is outside 50 to 15,000 µm², it touches the image edge or blank padding, its core actin is below 0.8 of its edge actin (a hollow, vessel-like ring), its core actin is below 0.3 of the median fiber's, its ring is more than 150 a.u. below its own cytoplasm, or its centroid lies outside the main section. A fiber whose cytoplasm GFP is more than 5 scaled median absolute deviations (1.4826 × MAD) above the median of the counted fibers (bright cytoplasm, e.g. a damaged fiber) is flagged and left out of the calls. Fibers with fewer than 24 of 36 ring pieces measured get no score.

## Score, threshold and call

- **Score at coverage c**: the k-th brightest of a fiber's measured ring pieces, k = ceil(c × pieces), so a fiber's score at 50% is reached by at least half its ring. Coverage levels in the tables: 50, 60 and 70% (18, 22 and 26 of 36 pieces).
- **Threshold**: per muscle, the 98th percentile of the pooled control fibers' scores at 50%. About 2% of control fibers reach it by construction.
- **Call**: a fiber is positive at level c when its score at c reaches the threshold. The 60 and 70% calls use the same threshold, so each is nested in the one before.
- **Controls, held out**: each control animal's fibers are called against the threshold set on the other control animals of the muscle, the same rule applied without them. Its result is therefore what an unseen control animal would give. The leave-one-animal-out table lists each held-out threshold and false-positive rate. With a single control animal there is nothing to hold out; that animal is called against its own threshold and marked in the tables (`threshold_from`).
- **No controls in a run**: a muscle can borrow the threshold of an earlier run, which is valid only if staining and imaging were the same. The earlier run must use the same rule, measurement settings and Cellpose weights; the run records where each threshold came from.

## Sections and animals

Sections are weighted equally within an animal: an animal's % positive is the mean of its sections' (with minimum and maximum), and the pooled share over all its fibers is reported next to it. Animals are the unit of comparison; the tables are descriptive and include no statistical test.

## Fiber size

Minimum Feret diameter of the included fibers, pooled per animal, in 5 µm bins, with median and the variance coefficient (SD / mean × 1000). There is no fiber-type stain in these scans, so this is size only.

## Records

Every run writes `run_settings.json` (rule, scope, measurement settings, code version, package versions, GPU, Cellpose weights, thresholds and results), and every section records the code version that measured it and the weights that segmented it. A run that resumes over sections made by other code or weights says so.

## Caveats

- A fiber whose outline touches blank padding is left out, but a fiber where only the outer half of the band reaches padding stays in, with fewer band pixels in those sectors.
- At 20x the band of a fiber can include membrane signal of its neighbour where two fibers touch.
