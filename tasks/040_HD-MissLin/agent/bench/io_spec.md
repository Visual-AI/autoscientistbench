# I/O specification

Each instance is an independent missing-outcome linear regression experiment. All arrays use float64 unless noted.

## Input

`--input PATH` is an NPZ file containing only:

| Key | Shape | Meaning |
|---|---|---|
| id | (M,) Unicode | Opaque unique identifiers; order is arbitrary |
| X | (M,900,1000) | Fully observed design matrices |
| y | (M,900) | Observed responses; missing responses replaced by zero |
| R | (M,900) | 1 for observed response, 0 for missing response |
| query | (M,1000) | Query vectors for conditional-mean inference |

No instance seed, complete missing responses, coefficients, or RNG state is supplied. Never treat zero-filled missing outcomes as observed responses. Every result must be computed from its own instance only.

## Output

CSV, exact header `id,m_deb,sigma_hat,asym_se`, one row for every input id, arbitrary row order. All three numerical columns must be finite. `sigma_hat` and `asym_se` must be strictly positive and their product finite.

The reported interval is `m_deb +/- Z * sigma_hat * asym_se`, where `Z = scipy.stats.norm.ppf(0.975)`. A candidate may use a different interval estimator, representing its center and standard error in these columns. The baseline reports its scaled-Lasso noise estimate and debiasing asymptotic standard-error factor.

## Labels and scoring

Public validation labels `truth.npz` contain `id,seed,m_true,beta` and the unchanged baseline's measured coverage on that same split, `baseline_coverage` (with `baseline_coverage_seeds`, its value per algorithm seed). The seed is a data-generation seed, never an input to prediction. The coefficients and conditional means are for offline evaluation only. The frozen generator matches the source simulation's RNG draw order, including the unused MCAR draw before MAR.

`python bench/validate.py PRED INPUT` checks the prediction format and exact id set.
`python bench/scorer.py PRED TRUTH` prints one JSON line with `ci_length` (lower is better), raw length, empirical coverage, the coverage floor actually applied, the baseline coverage and tolerance it was built from, the nominal threshold as a diagnostic, Monte Carlo standard error and mean absolute bias.

The coverage floor is **baseline-relative**: `baseline_coverage - 2*max(sqrt(p*(1-p)/M), sqrt(0.0475/M))` with `p = baseline_coverage`. It is the same two-standard-error width as the nominal `0.95 - 2*sqrt(0.0475/M)` rule, re-centred on what the unchanged baseline actually attains on these instances, and it reduces to the nominal rule when the baseline attains 0.95. Coverage below the floor yields `Infinity`. Malformed predictions also yield `Infinity` when passed directly to the scorer.

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
