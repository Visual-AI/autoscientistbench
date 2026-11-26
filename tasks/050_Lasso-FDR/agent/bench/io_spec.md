# hFDR observation and prediction contract

`--input` is a directory with `manifest.json` and opaque `.rds` files. The manifest gives version 1 and `instances`, each containing `id`, `file`, `design`, `n`, `p`, and `n_lambda`. The IDs are independent random tokens. Each RDS object contains exactly `X`, `y`, and `lambda`: observed design matrix, observed Gaussian response, and the frozen regularization grid. There are no generating seeds, coefficients, null masks, FDPs, covariance truth, latent values, or cached estimates in these objects.

The full validation set contains four designs, 120 observations per design, 1500 rows and 500 columns per observation, and ten lambdas per design. Labels are separate and contain the observations plus beta solely so the frozen scorer can recompute original Lasso selections and oracle FDP without trusting candidate outputs. Labels must never be read in prediction. No training corpus is supplied. Validation supports evaluation and method selection, not fitting reusable models or learned preprocessing. Per-instance estimation may use only that instance's X/y observations and supplied grid; fitted state must not pass between instances.

Prediction is JSON `{"version":1,"estimates":[{"id":"opaque-token","hFDR":[0.1,0.2],"hFDR_se":[0.01,0.02]}]}`. Include exactly one entry per input ID and one finite nonnegative hFDR estimate per lambda, in the given lambda order. hFDR estimates may exceed one. `hFDR_se` is optional diagnostic output and is not used as the Monte Carlo standard error guard. Duplicate IDs, duplicate JSON keys, missing/extra IDs, nonnumeric values, infinities, NaNs, and wrong lengths are rejected.

The scorer computes original fixed-X Gaussian Lasso selections independently with glmnet (intercept and standardization enabled), and realized FDP = selected null variables / max(1, selected variables). For each design/lambda cell, bias is mean(hFDR - FDP), and `se_cell = sd(hFDR-FDP, ddof=1) / sqrt(120)`. MAD is the equally weighted mean of the absolute 40 cell biases. Every cell must satisfy bias >= -3*se_cell. Each design's mean of ten sample SDs of hFDR must be <= 1.2 times its archived original baseline mean SD. These constants are in `calibration.json`.

All four designs, 120 replicates per design and ten lambdas are required for a finite score. Any protocol or statistical guard failure gives +Infinity. `validate.py PRED INPUT_DIR`; `scorer.py PRED LABEL_DIR`.

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
