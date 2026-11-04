# Glossary

Terms as `msq.dystrophin` uses them. Method: [METHOD_dystrophin.md](METHOD_dystrophin.md).

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
