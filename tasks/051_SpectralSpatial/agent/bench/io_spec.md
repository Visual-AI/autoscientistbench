# Observed spatial simulation input and posterior effect output

Input is a directory containing `manifest.csv` with exactly `id,n,grid_side,observed_file`. Each referenced RDS is a list with only numeric matrices `X` and `Y`; `X` includes the intercept, and `Y` has five observed outcomes. Original lattice order is `expand.grid(1:grid_side,1:grid_side)`. No generation seed, latent variable, effect truth, or loading matrix is embedded in an observed input. Default inputs contain the complete five-setting, 12-replicate, n=400 protocol.

Output CSV header is exactly `id,exposure,outcome,mean,lower,upper`. There must be one row for each input case, each exposure 1–9 (corresponding to X columns 2–10), and each outcome 1–5. `mean` is the MSM posterior mean at the first spatial frequency; `lower` and `upper` are its 2.5% and 97.5% posterior quantiles. These summaries must be computed for the fixed MSM likelihood, tensor model and priors. Posterior solvers and numerical approximations may change; unrelated effect estimators or confidence intervals are not substitutes. All numbers must be finite and `lower <= upper`; duplicate, missing, extra or malformed rows are invalid. No intercept rows are emitted.

The scorer compares these summaries with labels using strict interval inclusion, computes per-setting MSE and coverage, and emits `mean_mse = +inf` unless every setting has coverage at least 0.93. It always reports the raw MSE and per-setting coverage for diagnosis when the prediction format is valid.

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
