# Input and output

Input JSON has `version: 1` and a `datasets` array. Each item contains only an opaque `id` and `y`, a 500-by-2 matrix of observed bivariate normal coordinates. No generating seed, true cluster count, component label, mixture mean or covariance is provided in input.

Output JSON has `version: 1` and `predictions`, exactly one item per input ID, in any order. Each item contains exactly `id` and `p_kn`: 500 finite probabilities for K=1,...,500. Values must lie in [0,1] and sum to 1 within 1e-8. Missing, unknown, duplicate IDs, duplicate keys, extra fields, nonfinite values and malformed JSON are rejected.

Truth labels are stored separately as ID/true-K pairs. The scorer averages each true-K posterior probability within K=2,4,6, then averages the three cells only if their means meet 0.835, 0.550 and 0.458 respectively; otherwise score is zero.

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
