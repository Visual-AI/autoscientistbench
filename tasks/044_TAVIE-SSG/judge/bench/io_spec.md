# I/O specification

`train.sh --out DIR` writes config.json. There is no cross-instance fitting stage.
`predict.sh --model DIR --input PATH --output FILE` fits each observed dataset independently.

Input is an NPZ file, loaded with allow_pickle=False:

- `id`: unique strings, one per observed dataset; no parameter identities or generator seeds.
- `offsets`: integer vector of length len(id)+1, starting at zero. Dataset j occupies rows offsets[j]:offsets[j+1].
- `X`: float64 matrix with eight observed covariates. The model adds an intercept.
- `y`: float64 response vector, sharing the concatenated row axis with X.

Prediction is an NPZ file written through a binary file handle (do not append an extension), with exactly these keys:

- `id`: the same id set, without duplicates. Any order is allowed.
- `z`: finite float64 tensor (number of instances, 8000, 10). Coordinates are nine beta coefficients (intercept first), followed by log(tau2). No standardization.
- `seconds`: positive finite float64 vector, one elapsed wall time per instance for all model construction, fitting and 8000-sample generation. Exclude dependency import, input/output and SW scoring. Include all instance-dependent computation and any method setup that depends on observations.

Each instance must use only its own X/y and the trained configuration. Sampling must honor BENCH_SEED; use a content-derived seed so shuffling or subsetting instances does not change predictions. The baseline defines this in bench_run.sample_seed.

For independent timing replay, code must retain `bench_run.fit_sample(X, y, seed, config)` returning the complete (8000,10) sample matrix and `config.json` written by train. `seed` is the content-derived algorithm seed; the callback must include every instance-dependent computation. Import-time fitting and cached predictions are forbidden. The replay process recomputes samples and measures elapsed time outside the callback.

Labels are NPZ with `id,n,z_ref,beta,tau2_true,seed,nuts_seed,projection_seed,baseline_seconds`. They contain the NUTS reference posterior, true beta and auxiliary seeds; predict never needs labels. The scorer computes 500-projection SW against z_ref, takes medians within n then the mean over n; per-n median elapsed times are reported as diagnostics. Output is one JSON line with `sw`, `sw_raw`, and per-n diagnostics. Invalid formats produce Infinity. The supplied point-parameter or collapsed-distribution predictions are not special-cased by the literal SW score: their distributional error is scored normally.

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
