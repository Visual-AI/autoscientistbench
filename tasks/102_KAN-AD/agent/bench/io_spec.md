# KAN-AD I/O

All data and predictions are NumPy NPZ archives, without pickled objects. File paths may have no extension. Write predictions with `np.savez` through an open binary file handle.

Input archives contain `ids` (unique string curve ids), `offsets` (integer vector, length number of curves plus one, starting at zero), and `values` (concatenated raw univariate observations). Slice `values[offsets[i]:offsets[i+1]]` for curve i. Optional `dtypes` restores the original per-curve NumPy dtype. Every series is an independent segment; do not prepend another split's history. Training and validation inputs use the same layout.

Labels contain the same ids and offsets, `y` (concatenated binary point anomaly labels), and `dataset` (one of AIOPS, TODS, WSD, UCR per curve). Labels describe raw observations, including the initial context. Curve ids are opaque and have no semantic meaning.

Predictions contain exactly `ids`, `offsets`, and `scores`. The ids must match the supplied input exactly, without duplicates; any order is accepted. Prediction offsets count scored points, so each curve has input length minus 96 scores. `scores` is a finite floating tensor of shape `(5, offsets[-1])`: one independently fitted trial per row, aligned to raw observations starting at index 96. Larger scores indicate more anomalous observations. Do not output thresholds, binary decisions, labels, or self-reported aggregate metrics.

The frozen scorer retains EasyTSAD's alignment, margins (AIOPS/TODS 5, WSD 3, UCR 50), all-normal curve skipping, stable tie behavior, event squeeze F1 and 5-delay point F1. It averages curves within datasets, then equally averages the four datasets and five trials. `avg_f1e` equals the resulting event F1 when mean delay F1 is at least 0.727 on test or 0.59 on val, otherwise zero. Invalid tensors receive Infinity. Threshold sweeps are internal evaluation operations; they do not select a submitted model or deployment threshold. Any deployment threshold or model selection must use training/validation data only.

A dataset without any anomalous curves contributes zero and is listed in `empty_datasets`. This convention makes development scoring finite without replacing the fixed four-dataset denominator. Diagnostics include ungated F1e, F1d, per-trial and per-dataset values.

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
