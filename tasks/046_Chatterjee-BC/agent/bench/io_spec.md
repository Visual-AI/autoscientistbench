# 046_Chatterjee-BC estimator interface
`--input` is a directory with `manifest.json` and observed-only binary files. The JSON has
format `acbc-observations-v1` and a `settings` list with `setting_id,n,d,replications,file`.
Each binary is little-endian float64. Its first dataset is the shared setting calibration input (bootstrap variance in the baseline); the next
`replications` datasets are scored. Each dataset contains column-major X (n*d doubles),
followed by Y (n doubles). Covariates and response are normal-CDF transformed. No rho, true correlation, generation seed, or coverage label is present in input.
The full protocol has 1000 scored replications per setting. Group IDs are shuffled opaque IDs.

Output is CSV with exactly `setting_id,replicate,estimate,lower,upper`. Replicate is 1-based.
Every scored dataset must have one finite estimate and a finite ordered interval enclosing
it. Do not emit a row for the initial bootstrap dataset. Row order is irrelevant.

The scorer matches IDs to separate labels, computes per-setting RMSE and inclusive ECP,
then averages RMSE only over positive-rho settings. Coverage guards average ALL settings,
including rho=0: mean ECP >=0.94 and every ECP >=0.85. Failure scores +Infinity.

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
