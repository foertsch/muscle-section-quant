# Glossary

Terms as `msq.dystrophin` and `msq.psr` use them. Method: [METHOD_dystrophin.md](METHOD_dystrophin.md), [METHOD_psr.md](METHOD_psr.md).

| Term | Meaning |
|---|---|
| band | Pixels within 1.5 µm of a fiber's outline, inside and outside. |
| ring piece | One of 36 angular sectors of the band. Its value is the 90th percentile of the smoothed GFP in it minus the fiber's cytoplasm. |
| cytoplasm | Mean GFP of the fiber's pixels deeper than 3 µm from the outline. |
| membrane-specific dystrophin | The median ring piece of a fiber. |
| coverage | The share of a fiber's ring that must reach the threshold: 50, 60 or 70% (18, 22 or 26 of 36 pieces). |
| score | A fiber's k-th brightest ring piece at a coverage; the fiber reaches the threshold at that coverage when its score does. |
| threshold | Per muscle, the 98th percentile of the pooled control fibers' scores at 50% coverage. |
| call | Positive when the score reaches the threshold, negative when it does not, not called when fewer than 24 ring pieces were measured. |
| held out | A control animal's fibers called against the threshold set on the other control animals. |
| threshold_from | Which threshold a section was called against: all controls, other control animals, this animal alone, or a borrowed threshold. |
| main section | The largest piece of tissue after bridging gaps up to 50 µm. |
| excluded | An object that is not counted as a fiber (size, edge, hollow, low actin, ring below cytoplasm, outside the main section). |
| flagged | A fiber with bright cytoplasm, left out of the calls. |
| transmittance | PSR: a pixel's colour divided by the glass colour, per channel, so that glass is 1. |
| glass colour | PSR: the median RGB of the pixels at least 200 µm from the tissue. |
| red, yellow, other, background | PSR pixel classes from hue and saturation after the division by the glass colour: background is saturation 40 or below, red (collagen) hue below 23 or 234 and above, yellow (fiber) hue 23 to 50, other the rest. Rule of Hildyard et al., protocols.io `81wgb6j1olpk`. |
| collagen outside the fiber outlines | PSR primary readout (`collagen_outside_fraction`): red pixels in the main section outside every Cellpose-SAM outline, divided by the red plus yellow pixels of the main section. Red inside any outline does not count. |
| red_fraction | PSR secondary readout: all red pixels of the main section divided by its red plus yellow pixels. |
| freezing artifact | PSR: small white holes inside fibers. A section is marked in the R figures when more than 0.3 of its fiber area lies in 200 µm tiles with such holes. The mark excludes nothing. |
