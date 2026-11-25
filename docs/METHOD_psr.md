# Picrosirius red fibrosis: method

How `msq.psr` turns a whole-section picrosirius red (PSR) scan into collagen outside the fiber outlines per section and animal. Every number below is a setting in the code (`Params` in `src/msq/psr/section.py`, `src/msq/psr/settings.py`, `src/msq/psr/fiber_size.py`, `src/msq/psr/hole_survey.py`, `R/psr/measures.R`).

## Input

Stitched Nikon ND2 brightfield scans at 20x, one 8-bit colour image in one channel named `Brightfield`, 0.20 to 0.23 µm per pixel (0.2167 µm for the lab's whole-section scans). The file stores the colour planes as BGR; they are flipped to RGB on read. Whole-section scans have `full` in the file name; single fields do not and are left out. The muscle name (`QUAD`) must appear in the name of the scan's own folder or in its file name.

The checks (`checks.py`) stop a run before it starts when a file does not open, is not one stitched colour image, has another channel name, another pixel size or another bit depth, when a section appears twice, when a group is not in the cohort, or when there are no control sections. They warn when a group has fewer than 3 animals, and note when exposure, camera gain, white balance, transmitted light intensity or in-camera sharpening differ between scans, and on which days each group was imaged.

## Tissue, main section and glass colour

The scan is block-mean downsampled 6x (to 1.30 µm per pixel) for tissue finding and segmentation. Tissue is the smoothed saturation (Gaussian, sigma 2 pixels of that grid) above an Otsu threshold, holes filled, pieces smaller than 1% of the largest dropped. The main section is the largest tissue piece after bridging gaps of up to 50 µm, so neighbouring sections on the slide and torn-off fragments are left out. Pixels that are zero in all channels (blank padding) count as neither tissue nor glass.

The glass colour is the median RGB of the pixels at least 200 µm from the tissue (the margin is grown as a diamond, so 200 µm along the axes and less on the diagonals). Dividing each pixel by it gives transmittance, with the glass at 1. This is meant to remove differences in exposure, gain, lamp and white balance between scans. It does so only if the camera responds linearly to light. That is assumed in the code; it was checked on synthetic images only.

## Pixel classes

At full resolution each pixel is divided by the glass colour (scaled so glass is 255, clipped to 0 to 255), then converted to hue and saturation bytes the way ImageJ's HSB stack gives them. The rule is that of Hildyard et al. (see Reference):

| Class | Rule |
|---|---|
| background | saturation 40 or below (checked last, so it overrides the others) |
| red (collagen) | hue below 23, or 234 and above |
| yellow (fiber) | hue 23 to 50 |
| other | every remaining hue |

Per section the pixels of the main section are also counted by hue and saturation (`hue_sat_hist.npz`).

## Segmentation

[Cellpose-SAM](https://github.com/MouseLand/cellpose) (cellpose 4.2, weights `cpsam_v2`) outlines the fibers on the downsampled image. Its input is `3 × green − 2 × red` transmittance, clipped to 0 to 1, with white areas set to 0 when they are larger than 150 µm² (white: all three channels pass more than 0.8 of the light). Glass, tears and lumens go dark that way and fibers next to them keep an edge; the small white holes inside fibers stay bright. Flow threshold 0.4, cell-probability threshold 0. Other inputs and settings exist as variants (`--variants`) for comparison: `green`, `yellow`, `fiber_raw`, `od`, `ds4`, `ds8`, `flow06`. The tables use the first variant given (default `base`).

## Zones per fiber

On the segmentation grid each outline gets three zones: the core (the mask eroded by one grid pixel, 1.30 µm: the 1.5 µm edge width rounded to the grid), the edge band (mask minus core) and the ring just outside (one grid pixel wide, each pixel owned by the nearest fiber). The class counts of each zone come from the full-resolution classes, which lets "red inside a fiber" be told apart from "outline slightly over the collagen". They are written per object to `objects.csv`.

## Which objects count as fibers

For fiber size, fiber count and fiber density an object counts when its centroid is in the main section, it does not touch the image edge, its area is 50 to 15,000 µm², and less than half of its core and edge-band pixels are red (`fiber_size.py`). This rule is provisional. The collagen readout does not use it; see below.

## Readouts

Both are over the main section only. Nothing is masked: folds, dust, edge staining and tendon stay in the counts.

- **`collagen_outside_fraction`** (primary, "collagen outside the fiber outlines"): red pixels in the main section outside every Cellpose-SAM outline, divided by the red plus yellow pixels in the main section. All red outside the outlines counts, whatever it is (perimysium, epimysium, tendon, vessel walls). Red inside any outline does not count, including outlines of objects that are not fibers, so the readout depends on the segmentation.
- **`red_fraction`** (secondary): all red pixels in the main section, inside or outside outlines, divided by the red plus yellow pixels in the main section. This is the fibrosis fraction of Hildyard et al. It is in the tables and not plotted.

`section.json` also records `coverage_yellow` (the share of yellow pixels inside an outline) and `red_in_masks` (the share of red pixels inside an outline), as checks on the outlines.

**Fiber size.** Per fiber: minimum Feret diameter, maximum Feret diameter and cross-sectional area (`fiber_size.py`). Per section: the number of fibers and the median of each. Per animal: the median of all its fibers (minimum and maximum Feret; the animal table has no area column). The R step also draws size distributions: a Gaussian kernel density per animal (bandwidth 2 µm for minimum Feret, 3 µm for maximum Feret, 150 µm² for area), averaged over each group's animals so every animal counts once.

**Exploratory measures.** Fibers per section, and fibers per mm² of the main section. They are plotted and tested like the primary readout and are labelled exploratory in the figures.

**Freezing-artifact mark.** `hole_survey.py` looks for small white spots inside yellow fibers on a 3x downsampled grid (0.65 µm per pixel): a white spot of 3 to 150 µm² whose 2-pixel ring is at least 70% yellow is a hole. The survey reports `holey_area`, the share of the fiber area (yellow plus hole area) that lies in 200 µm tiles where holes are more than 2% of that area. `msq.psr.summary --holes` takes a CSV with the columns `sample` and `holey_area` and writes it to `sections.csv` as `freezing_artifact_area`. The R figures mark a section when this share is above 0.3, and an animal when at least half of its sections are marked. The 0.3 is a threshold chosen on the lab's sections. The mark excludes nothing: every section stays in the tables and the statistics.

## Sections, animals and groups

An animal's value is the mean of its sections, each section weighted equally. A group's value is the mean of its animals. The animal is the unit of comparison; sections are shown in the by-animal figures.

## Statistics

`R/figures.R` and `R/export.R` compare each group with the control by Dunnett's test (`multcomp::glht`), on the animal means. There is no ANOVA first. The one-way ANOVA p value is still written to `stats_vs_control.csv`, with the difference to control, its confidence interval, the adjusted p value, the residual SD, and the smallest difference to the control that a two-sample t-test with these animal numbers and this spread would detect (80% power, alpha 0.05; not adjusted for the several comparisons). The primary readout is collagen outside the fiber outlines. Fiber count and density are exploratory. `red_fraction` is not tested.

## Records

Every section folder holds `section.json`: the glass colour, the class counts, both fractions, all settings, run time, peak memory, the code version and the Cellpose weights that made it. The run folder holds `run_settings.json` (scans, versions, weights, summary settings) and `run.log`.

## Caveats

- The readout depends on the outlines. An outline around something that is not a fiber hides the red inside it; a fiber that is missed leaves its yellow in the denominator and adds none to the numerator.
- Perimysium, epimysium, tendon and vessel walls are collagen outside the fibers and count.
- Folds, dust and edge staining are not masked.
- If scans of different groups differ in brightness after the division by the glass colour, a remaining effect would sit on the group contrast.
- The mark for freezing artifacts needs the hole survey table; `msq-psr` does not run the survey.

## Reference

Hildyard J, Foster E, Wells D, Piercy R. Histological image quantification of picrosirius red stained skeletal muscle sections. protocols.io, 2022. [doi:10.17504/protocols.io.81wgb6j1olpk/v1](https://doi.org/10.17504/protocols.io.81wgb6j1olpk/v1)
