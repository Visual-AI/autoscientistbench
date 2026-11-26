# Frozen I/O specification

`train.sh --out DIR` creates a model directory; the baseline writes `NO_TRAINING`.
`predict.sh --model DIR --input PATH --output FILE` reads a directory of NPZ core images and writes one CSV at exactly FILE.

Each input filename is an opaque 20-character hexadecimal core ID followed by `.npz`.
Load with `numpy.load(..., allow_pickle=False)`. Arrays:

| Key | Shape / dtype | Meaning |
| --- | --- | --- |
| image | (56, 3000, 3000), uint8 | Original fluorescence intensities, channel / y / x |
| segmentation | (3000, 3000), integer | Published cellpose mask; 0 is background, positive values identify cells within this core |
| channels | (56,), Unicode | Marker names, in image channel order |
| y, x | (3000,), integer | Pixel coordinates |
| qc_cells | (N,), int64 | Fixed scoring population, computed solely from the mask using released outlier-cell removal |

No source core names, patient metadata, reviewed thresholds, expression matrices or annotations are supplied in inputs.
The baseline recomputes observations and QC, then quantile thresholding, mean intensity, arcsinh and argmax.
Do not alter the frozen mask or the set of cells being scored. The baseline computes quantiles over the whole image; candidates may use other feature transformations.

Output CSV has exactly the header `id,label`. One row is required for every fixed QC cell, with
ID formed by the opaque core ID, a colon, and the decimal segmentation cell number. Row order is arbitrary;
duplicate IDs and extra IDs are invalid. The validator also rejects missing rows and non-finite numeric tokens.
Write an empty label or `-1` to abstain; it counts as a mismatch and is never removed from the denominator.

| label | Cell type |
| --- | --- |
| 0 | B cell |
| 1 | T cell |
| 2 | Myeloid cell |
| 3 | Dendritic cell |
| 4 | Macrophage |
| 5 | Stromal cell |
| 6 | Endothelial cell |

Any other label is an out-of-class prediction and counts as a mismatch.
Public `data/val/labels.csv` uses the same schema and contains the majority label of three annotators.
Three different annotation labels are resolved to Annotator 1. Annotations are never prediction inputs.

`validate.py PRED INPUT_DIRECTORY` exits nonzero for format violations.
`scorer.py PRED LABELS` prints one JSON line with `pooled_kappa`, pooled over all fixed cells.
The scorer explicitly fills absent predictions with an invalid sentinel even when called without validation;
it never averages only the submitted rows. Invalid predictions and invalid truth use distinct sentinels.
Duplicated or extra IDs, malformed CSV, and non-finite tokens yield `Infinity` as a failed submission.

## Validation use (baseline and candidates)

Fit models and learned preprocessing only on the designated training observations.
Validation features and labels are for evaluation and method, hyperparameter or
checkpoint selection only. This restriction applies equally to the baseline and
every candidate, including final retraining. Do not merge validation into training,
fit normalization/encoders on validation, distill from validation predictions, or
relabel validation observations as training or calibration data.

For tasks defined as per-instance inference, optimization or online interaction,
use only the observation/training/calibration/feedback roles explicitly specified
for that instance. Do not use its evaluation targets or carry fitted state from
validation cases into other independent cases. A development replay directory may
contain separately designated training observations; its validation observations
remain excluded from fitting. Public visibility does not authorize training use.
