# Spatial isoform simulation I/O

## Observations

`input.npz`, loaded with `allow_pickle=False`, contains exactly four arrays:

| Array | Shape | Meaning |
|---|---|---|
| `id` | (N,) unicode | Opaque unique instance identifier; order has no meaning |
| `offsets` | (N+1,) int64 | Instance i occupies the half-open interval offsets[i]:offsets[i+1] |
| `coords` | (sum of cell counts, 2) float64 | Original Sample 1 x,y coordinates in DNB units (0.5 micrometers) |
| `values` | (sum of cell counts,) int8 | Observed binary relative isoform expression |

An instance is one independent testing problem. Nested cell samples may share observations;
use only the observations inside the current instance. No scenario, geometry, replicate identity,
null/alternative flag, or generation seed is supplied to prediction.

## Predictions

CSV text, including when the output filename has no extension. Exact header: `id,called`.
Exactly one row per input ID, any order, no duplicates or missing/extra IDs. `called` is the
literal integer `0` or `1`, indicating whether the procedure declares spatial variability.
No NaN, Infinity, blanks, extra columns or nonbinary values. A method-filtered instance is a
noncall, never a removed row. The baseline calls when its uncorrected permutation p <= 0.05
and Moran's I > 0.01. Alternative procedures may use other valid pattern-agnostic tests;
null rejection constraints apply to their submitted calls.

## Labels

`labels.npz` holds `id,scenario,n_cells,rep,simulation_seed,subsample_seed,geometry,n_ones,
n_mask,n_ones_full,n_mask_full`. Geometry is a JSON string. True pattern/null identity and
mask counts are for scoring and offline diagnostics only, never for prediction or fitting.
Replicate groups identify dependence among nested samples when estimating Monte Carlo SE.
Public `data/train/coordinates.npz` contains only the shared experimental coordinate design;
there are no cross-instance training examples and the baseline does not fit reusable weights.
Validation is only for evaluation and selection, not fitting reusable models or learned
preprocessing. A test may use the observations of its own instance.

## Scoring

`python bench/scorer.py PRED LABELS` prints one JSON line with `macro_recall` (higher is better),
raw recall, clustered simulation SE, and null rejection diagnostics. Recall is an equal-weight
mean across pattern scenarios and cell levels. Null pooled rejection must be <= 0.058 and each
null scenario/cell-count rejection rate <= 0.094; violation returns 0, as specified by this
task's score rule. Malformed predictions return Infinity and fail validation. Pattern-agnostic
behavior and subset/ID invariance are patch-audit rules.

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
