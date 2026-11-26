# I/O specification

Input is NPZ with `id` (N Unicode opaque identifiers), `x` (N,2048,50 float32 standardized observations) and `trial` (N integer algorithm replicate indices). Trial indices are public algorithm metadata, independent of data-generation seeds. Rows may be shuffled. Each graph estimate must use only that instance's observations and development-selected configuration.

Output is CSV: header `id,g0_0,g0_1,...,g0_49,g1_0,...,g49_49`, one row per input ID, any row order, exactly 2500 finite real-valued scores per ID. All entries are scored in row-major order. No symmetry, range, or diagonal restriction is imposed on submitted scores. Write exactly the requested filename, even without an extension.

Labels (`truth.npz`) hold `id`, `true_prec` (diagonal zero), `theta`, `sigma`, `seed`, and `trial`. Seeds and true parameters are not prediction inputs. Public labels are for offline diagnostics and configuration selection, never truth lookup or supervised memorization of the fixed graph.

`python bench/validate.py PRED INPUT` validates shapes, IDs, duplicates and finiteness. `python bench/scorer.py PRED LABELS` prints one JSON line with `edge_auc`: mean of original full-matrix ROC AUC across trials. The scorer takes absolute scores, uses NumPy's original argsort tie order, and trapezoidal integration. It does not symmetrize, drop the diagonal or use sklearn's tie aggregation. Invalid formats produce Infinity; the metric otherwise has no numerical constraints.

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
