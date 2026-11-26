# Geodesic DiD input and prediction contract

The input directory contains `index.csv` (`id,file,n`) and one observation CSV
per replication. Each observation row has columns
`group,pre_location,pre_scale,post_location,post_scale,lower,upper`.
Group is 0 for control and 1 for treated. The location, scale and support give
a lossless description of each observed truncated-normal outcome distribution;
these are not population treatment-effect labels. Evaluate quantiles at
0,0.01,...,1 to recover the original 101-point observations (support limits at
the endpoints). No generator seed or population reference curve is in input.

Output CSV has one row per input id in index order and exactly columns
`id,start_000,...,start_100,end_000,...,end_100`. Start/end arrays describe the
estimated geodesic effect map and must be finite, strictly increasing quantile
curves with endpoints -3 and 3 (absolute tolerance 1e-7 for each endpoint). The baseline outputs `gdd(...)$gdd$start` and
`$end`, never a self-reported error statistic.

The frozen scorer transports this map to the population reference using the
original `gtm` and `pracma::spinterp`, computes root-mean-square discrepancy from
the target over 101 points, then averages the distances across replications.
Metric `mean_geodesic_error` is lower-is-better. A separate labels directory
contains `labels.json` (ids) and `truth.csv` (population reference/target curves).
Test labels are judge-only.

Validation contains 500 independent n=1000 datasets. This is per-instance inference with no training split. Validation is only for evaluation and selection. Do not fit reusable models or learned preprocessing on it. Estimate each instance only from its own observed distributions.

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
