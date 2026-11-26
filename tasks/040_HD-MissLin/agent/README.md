# Efficient inference on high-dimensional linear models with missing outcomes

The judge rebuilds your code from scratch, predicts new observed instances and compares the score with the archived baseline.

## 1. Goal

Construct short 95% confidence intervals for m(x)=E[Y|X=x] in a high-dimensional linear model with missing outcomes. Each instance has n=900, d=1000, circulant symmetric covariance (rho=0.1), Gaussian noise and missing-at-random outcomes. The response model is Y=X.T @ beta + noise. The observation probability is P(R=1|X)=1/(1+exp(-1+X_7-X_8)), with covariates numbered from one. The fixed dense-coefficient experiment uses query e100, the unit vector selecting the hundredth coefficient. Compare interval estimators on the same supplied observations and target. Improve `ci_length`, lower is better, subject to empirical coverage. You may change the inference method within the rules below.

## 2. Data

- `data/train/README.md`: no cross-instance training split. Validation is only for evaluation and method or hyperparameter selection. Do not fit reusable models or learned preprocessing on validation. Per-instance fitting uses only that instance's own X, R and observed y; fitted state must not carry between instances.
- `data/val/input.npz` and `data/val/truth.npz`: eight public validation instances, data seeds 1 through 8. Labels are only for evaluation and method or hyperparameter selection, never fitting.
- `bench/generator.py` defines the frozen simulation. Use the supplied materialized arrays as the authoritative observations.
- Inputs contain only opaque ids, X, zero-filled observed y, R and query vectors. The coefficient vector, conditional mean and generation seed live in labels. See `bench/io_spec.md` for shapes.
- Do not generate additional instances, use outside data, re-split or relabel the supplied data. Every prediction must depend only on its own observed instance. No cross-instance fitting or sharing predictions.

## 3. Metric and scoring

`bench/scorer.py` reports the mean length of the submitted 95% intervals, `2 * norm.ppf(.975) * sigma_hat * asym_se`. Empirical coverage counts instances for which the true conditional mean lies in the interval centered on `m_deb`. The score is the mean length if coverage clears the floor below, otherwise `Infinity`.

The floor is relative to the unchanged baseline's own coverage on the same instance set, which is frozen in that set's labels as `baseline_coverage`:

```
floor = baseline_coverage - 2 * max(sqrt(p*(1-p)/M), sqrt(0.0475/M)),  p = baseline_coverage
```

M is the number of supplied instances, so the floor is computed separately for each evaluated instance set. This is the same two-standard-error width as the nominal `0.95 - 2*sqrt(0.0475/M)` rule, re-centred on what the baseline attains; it reduces to the nominal rule when the baseline attains 0.95. On the validation set the baseline covers 8 of 8, so the floor there is 0.8459 and 7 of 8 is the smallest passing count. The scorer reports `baseline_coverage`, `coverage_tolerance`, the applied `coverage_threshold` and the `nominal_threshold` alongside the score, so you can see how much room a submission has.

| Guard | Class | Enforcement |
|---|---|---|
| Coverage floor relative to the baseline | scorer | Formula above, from the frozen `baseline_coverage` |
| Complete unique ids, finite center, positive finite standard-error factors | scorer | Validator rejects invalid files; scorer also returns Infinity |
| Predictions respond to observed outcomes | patch audit | Review dependence on observed outcomes and reject fixed answers |
| No truth constants, truth table, paper answers, generation seeds or additional simulations | patch audit | Review submitted code; the public distribution is not permission to use its true mean |
| Each instance inferred independently from its own observations | patch audit | Review data access, state and cross-instance dependencies |

Validation baseline: `ci_length = 0.8833581145`, standard deviation across the three algorithm repetitions `6.77191921e-09`; coverage `1.000, 1.000, 1.000`. `bench status` also reports the baseline. The relative improvement threshold is `max(0.03, 2 * reference_std / abs(reference_mean))`. A guard violation fails the submission.

Coverage alone cannot detect an oracle that copies the known true mean and attaches an extremely small positive interval. This is explicitly prohibited and requires patch audit. A zero-width interval is invalid; shrinking a data-based interval lowers coverage.

## 4. Baseline

Official code: https://github.com/zhangyk8/Debias-Infer at `82497ff07d4c3d3ec1548614f681459d7f26ae0c`. The numerical routines are in `code/Paper_Code/Cir_Sym/debias_prog.py`. `code/repro/method.py` tries SCS, then ECOS and CVXOPT for the same debiasing program.

The baseline computes a scaled-Lasso pilot with `lam0='univ'`, 5-fold L1-logistic propensity fitting over 40 C values, then 5-fold debiasing CV over 41 gamma/n values. It computes the min-CV, 1SE and min-feas rules and submits the 1SE interval. With pilot beta_hat, propensities pi_hat and debiasing weights w, the center is `x.T @ beta_hat + sum(w * R * (y - X @ beta_hat)) / sqrt(n)`. The standard-error factors are the scaled-Lasso noise estimate and `sqrt(sum(pi_hat * w**2) / n)`.

On 32 CPUs, configuration setup took 5.57, 5.67 and 5.57 seconds across reference repetitions. Validation prediction took 621.28, 592.31 and 647.97 seconds, respectively. The declared memory budget is 48 GB; this is a budget, not a measured peak. All instances share one pool of `BENCH_CPUS` processes, and each cross-validation or rule fit in it uses one numerical-library thread.

## 5. Interface (frozen)

- `bash code/train.sh --out DIR` writes all reusable configuration into DIR.
- `bash code/predict.sh --model DIR --input PATH --output FILE` fits each supplied instance and writes CSV with exact header `id,m_deb,sigma_hat,asym_se`.
- Environment: `BENCH_SEED`, `BENCH_CPUS`, `BENCH_DATA_DIR`, `BENCH_BENCH_DIR`, `BENCH_CODE_DIR`. Algorithm seeds must come from `BENCH_SEED`, not from private data-generation seeds. The baseline combines `BENCH_SEED` with a hash of the observed design for order-independent instance streams.
- `bench status`, `bench eval-dev [--seed S]`, `bench validate-output FILE`, `bench submit [--final] [-m MSG]`.
- Dependencies: Python 3.12; install packages with `uv pip install`. The submission records `name==version` pins in `code/env.lock`. Only packages reproducible from PyPI are supported; no editable, git or local-path dependencies.

## 6. Rules

The free surface is `code/`: conditional-mean estimators, nuisance models, fitting losses, hyperparameters and numerical methods may change, subject to the pipeline limit under Forbidden substitutions, while preserving the stated high-dimensional linear-model estimand, MAR observation boundary, interface and guards. Frozen surfaces are `data/`, `bench/`, this README, script signatures and the CSV contract.

The judge uses 32 CPUs and three algorithm seeds. Memory budget is 48 GB. No network is available during train or predict. Required artifacts must be available through the submitted code and pinned environment.

Disqualifying: reading or guessing hidden instances; hard-coded predictions; looking up the known true mean or coefficients from the paper, generator, validation labels or stored constants during inference; creating extra simulation instances; using an instance's generation seed or RNG state; pooling predictions or outcomes across instances. Public labels may be used to evaluate procedures and select methods or hyperparameters. Each procedure must infer the conditional mean from its own observed instance, without fitting reusable state on validation.

## 7. Known pitfalls

Full CV fitting is expensive. Plan experiments around validation cost; reducing the process count will reduce simultaneous memory use but usually increases wall time. Each worker must stay single-threaded to avoid oversubscription.

R=0 means y is unavailable, even though the file carries a zero. The baseline's final residual sum multiplies by R and its pilot uses only R=1 rows. Do not fit missing zeros as real outcomes.

`PYTHONSAFEPATH=1` is enabled. Keep the scripts' explicit PYTHONPATH. Prediction paths have no required extension: write to the exact output path. Do not silently drop failed or slow instances. The scorer checks coverage on every supplied id.

Seed-to-seed algorithm variation and Monte Carlo uncertainty are different: the scorer reports Monte Carlo standard error across instances, while reference standard deviation is across algorithm seeds.

## Research contract

**Objective:** Construct shorter valid 95% confidence intervals for the fixed conditional mean m(e100) in the dense high-dimensional missing-outcome linear-model experiment.

**Fixed research objects:** The conditional-mean estimand, query e100, dense-coefficient generating design, n=900/d=1000, MAR mask and supplied observed outcomes are fixed. The normal-multiplier interval convention, mean length, the baseline-relative coverage floor, full instance set and independent-instance inference remain fixed.

**Editable methods:** Outcome/propensity nuisance estimators, working models, surrogate losses, debiasing and uncertainty methods, regularization and numerical solvers may change. The scaled-Lasso pilot, propensity logistic model, CV grids and 1SE rule are baseline estimators, not a different posterior/parameter target.

**Forbidden substitutions:** Do not change the target to another query, marginal mean or alternative coefficient model; treat missing zeros as observed responses; or substitute known true coefficients/mean. No extra simulated datasets, hidden seeds, cross-instance fitting or modified data/metric. When retaining the supplied scaled-Lasso, logistic-propensity and debiasing pipeline, changing only lam0='univ', the 40-point propensity grid, the 41-point gamma/n grid, the 5-fold splits or the submitted 1SE rule does not count as a method improvement. Multiplying `sigma_hat` or `asym_se` by a constant, or any other fixed rescaling of interval width, also does not count.

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
