# Bunny point-cloud interface

`train.sh --out DIR` fits a reusable DDPM to the frozen lifted points in `data/train/samples.npy` (float32, 100000 by 3). `mesh.obj` is the public geometric constraint, in the original coordinates; the author projector performs its own centering and normalization.

`predict.sh --model DIR --input PATH --output FILE` receives a NumPy NPZ containing only `id`, a string vector of opaque sample slots. Slots disclose no reference coordinates or seeds. Return an NPZ with exactly `id` and `samples`; `samples` must be a floating array of shape `(len(id), 3)` in normalized mesh coordinates. The ID set must match exactly, with no duplicates; order is immaterial for this distribution metric. Write via an open binary file handle so an output path without a suffix remains unchanged.

The sampler submits raw outputs, including failures; do not discard rows, replace failed samples or resample them. At most ten rows may contain NaN or infinity. The scorer applies its frozen `MeshConstraintProjector` and requires at most ten projected failures too. Coverage uses the finite projected rows. Fixed histograms retain failures in one overflow bin.

Validation uses a 10000-slot request. The input determines the number of slots. The trusted scorer receives `truth.npz` with `id`, `reference`, `hist_range`, and frozen `baseline_jsd`, `baseline_tvd`, `baseline_jsd_sd`, `baseline_tvd_sd`. Public validation labels expose these fields for evaluation and method, hyperparameter or checkpoint selection only. Prediction must never read reference labels.

`python bench/validate.py PRED INPUT` exits zero for a valid file. `python bench/scorer.py PRED LABELS` prints one JSON line with `cov` (higher is better); any failed guard returns `Infinity`, which the harness treats as an invalid submission. Other keys are diagnostics. The plotting convention is deliberately `coverage(projected_generated, reference, k=5)`: the neighbor balls are centered on generated points and include self among the five neighbors.

The task-specific Research contract in README is binding: output validity alone does not permit substituting a different research object or objective.

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
