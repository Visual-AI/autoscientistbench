# Frozen I/O specification

## Observation input
`--input PATH` is an NPZ archive, read with `allow_pickle=False`. `id` is a one-dimensional string array with opaque experiment IDs. For each ID, `o_ID` is an int8 array of shape `(180000,8)` with columns `X1,X2,X3,X4,X5,S,A,Y`. X and S are binary. A and factual Y are binary only for selected patients (`S=1`), otherwise the sentinel is -1. The unmeasured sixth covariate, counterfactual outcomes, probabilities, mechanism names and generator seeds are absent.

Within an experiment, rows 0:50000 are the RCT fitting cohort; 50000:100000 the observational fitting cohort; 100000:140000 fresh RCT evaluation candidates; 140000:180000 fresh observational evaluation candidates. Cohort membership is observed. Patient order is frozen because each Pearson statistic uses the first 2000 eligible observational evaluation candidates. There are five measured binary covariates, encoded by the authors as 32 one-hot strata in lexicographic order (`group = 16*X1+8*X2+4*X3+2*X4+X5`). Experiment order carries no meaning.

## Prediction CSV
Exactly one row per input ID, in any order. Header is exactly `format.COLS` from `bench/format.py`:

1. `id`.
2. 256 probabilities: for cohort `rct,obs`, then target `S,A,Y0,Y1`, then stratum `00` through `31`, use column `COHORT_TARGET_GROUP`, for example `rct_S_00` and `obs_Y1_31`.
3. Eight statistics, in order `SE_Y0_r,SE_Y0_p,SE_Y1_r,SE_Y1_p,SE_A_r,SE_A_p,SE_S_r,SE_S_p`.

Probabilities and p-values must be finite in [0,1], correlations finite in [-1,1]. No missing or duplicate IDs, no extra fields. The model table describes fitted conditional probabilities for every measured covariate pattern. Each patient's prediction depends on their measured X and cohort, using the model fitted to that experiment's fitting samples. The table permits any estimator of those probabilities; it does not require logistic regression.

The scorer reconstructs `abs((mu1_rct-mu0_rct)-(mu1_obs-mu0_obs))` and squared errors, then computes scipy's two-sided Pearson tests on the first 2000 eligible OS evaluation rows: all for S, selected for A, selected with matching treatment for Y0/Y1. This preserves the author's implementation of b1, including its difference of treatment effects. Reported statistics must match this reconstruction within absolute/relative tolerance 2e-7 and must produce identical positive-significance decisions.

## Labels and scoring
Public `truth.npz` contains IDs, mechanism, generator seed and trial, latent probability tables, transportability probabilities, observed evaluation candidates, and unchanged baseline per-instance probability tables/statistics. Only observations are available to predict.

`python bench/validate.py PRED INPUT` checks the output format. `python bench/scorer.py PRED LABELS` prints one JSON object with metric `rho_b1_A_detection`. The metric is the positive significant A-correlation rate in confounding if all four fixed false-positive constraints pass, otherwise 0. Malformed certificates, undefined correlations and inconsistent claimed statistics receive `Infinity` (invalid submission). Diagnostics include the raw detection rate, all mechanism rates and each guard result.

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
