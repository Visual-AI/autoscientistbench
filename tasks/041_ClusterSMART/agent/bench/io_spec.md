# 041_ClusterSMART I/O

Input is a directory with `index.csv` (`id,group,file`) and RDS batches. Each batch is a list of independent observed SMART study datasets; each entry contains `id`, `group`, `observations` (cluster_id, person_id, t, a_1, r, a_2, y), minimal `driver` fitting metadata and the original fit-settings table. No true effect, simulation parameter, generator seed or reference prediction is present.

Output is CSV with exactly one `id,estimate,lower,upper` row per input ID. Estimates and interval endpoints must be finite; `lower <= estimate <= upper`. The interval targets the end-of-study (1,1) versus (-1,-1) contrast. The scorer computes rejection and strict interval coverage itself.

Public validation contains 600 studies per environment. Each study is an independent inference instance. There is no training split; validation supports evaluation and method or hyperparameter selection only. Fit a study-specific estimator only to that study's observations. Do not fit reusable models or preprocessing across validation cases.

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
