# 052_AnisoBivFunc observation and prediction contract

The `--input` argument is a directory. `manifest.csv` has exactly `id,file,offset,N,side` columns. IDs and filenames use independent random 128-bit hexadecimal tokens. `file` ends in `.bin`; `offset` is a nonnegative byte offset. N is 100 or 150 and side is 51 or 101. Read N × side × side consecutive IEEE754 binary64 little-endian values. Each surface is a side × side matrix in R column-major order; surfaces follow consecutively. Its coordinate vector is `seq(0,1,length.out=side)`.

The manifest and observations contain no generation seed, true angle, sigma, H1/H2, cell, or replication number. Storage order and IDs do not identify a configuration. The frozen generator is for operator materialization; prediction consumes observations directly.

Prediction is a UTF-8 CSV with exactly `id,alpha_hat_adj` columns, one unique row for every input ID, no extra IDs. Angles are finite radians in [0,pi]. Row order is irrelevant. Duplicate IDs, missing IDs, extra IDs, malformed/nonfinite/out-of-range values are invalid.

Labels are JSON with a frozen protocol identifier and records containing `id`, `cell`, and `alpha`. Full labels declare `figure3-full-v1`: 60 cells and 200 replications each. The metric is the mean of the 60 medians of absolute angle errors. Only the separate verification fixture declares `smoke-v1` (two complete learning sets); full labels cannot select a smaller design. The scorer uses standard-library arithmetic and returns 1e100 for invalid predictions or malformed labels.

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
