# ALSE input/output contract

Input is a directory containing `manifest.json` (`format=alse-observations-v1`, ordered `cases` with `case_id` and integer `n`) and `observations.rds`, a named R list mapping each case ID to a tibble with exactly X1,X2,X3,A,Y. Each column is numeric and finite. These are observed samples only: no seeds, true parameters, fitted models or cached estimates.

Output is CSV with the exact header `case_id,n,learner,estimate,std_error`. Write exactly one row for each case and each output slot (`gam`, `ranger`); order is arbitrary. These names identify the two fixed nuisance-regression families: generalized additive models and random forests. Their implementations may change, with tuning as part of such a change, but an unrelated or identical replacement learner cannot be submitted under both labels. Estimates and nonnegative standard errors must be finite. The Wald CI is fixed by the scorer as estimate ± 1.959964*std_error; candidates cannot submit a different interval convention.

Labels are separate JSON containing `protocol` and `cases` (case_id,n,truth). Full validation uses every n=500,1000,...,4000, 40 datasets per size, and both learners. The scorer averages n*MSE equally over (n,learner) cells and returns +Infinity unless pooled coverage >=0.92 and every-cell coverage >=0.85.

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
