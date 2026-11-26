# Bias correction for Chatterjee's graph-based correlation coefficient

## 1. Goal
Estimate the population Chatterjee graph-based dependence functional from observed X,Y and return a 95% confidence interval for each dataset. The target is
`T(Y,X) = integral Var(E[I(Y >= t) | X]) dF_Y(t) / integral Var(I(Y >= t)) dF_Y(t)`,
where I is an indicator and F_Y is the marginal distribution of Y. Improve mean RMSE under the coverage requirements below. The experiment compares dependence estimation with and without nearest-neighbor bias correction as dimension and dependence vary.

## 2. Data
There is no training split; `data/train/` contains only a note. Validation supports evaluation and method selection, not fitting reusable models or learned preprocessing. Per-instance fitting may use only that scored replicate and its supplied calibration dataset. `data/val/input/` contains 75
settings on the validation grid d=2,4,6,8,10; n=300,600,900; rho=0,0.3,0.5,0.7,0.9, with
1000 scored replicates and one independent bootstrap dataset per setting. Inputs contain
observations and dimensions only; rho and true correlation are in `data/val/labels.json`.
The frozen `bench/generator.R` defines the validation design. Independent Gaussian covariates and a Gaussian response correlated with the first covariate are transformed by the standard normal CDF. The public validation seed is 16103101.
Do not resplit or use external data. Binary encoding is documented in `bench/io_spec.md`.

## 3. Metric and scoring
Lower `mean_rmse` is better: average of per-setting RMSE over the 60 positive-rho settings.
Coverage uses inclusive endpoints and all 75 settings. The scorer requires mean ECP >=0.94
and every setting ECP >=0.85; either failure gives +Infinity. Complete IDs, finite numbers
and interval ordering are enforced by the validator. Candidate estimators of this same dependence functional, nuisance
models, surrogate losses, regularization and interval methods may be replaced;
the degree-two basis, ridge n^-0.85, bootstrap m=floor(sqrt(n)), B=10000 and z=1.96
are choices of the supplied baseline. Reference validation mean RMSE is 0.04602472105559376, with standard deviation 0.0 across reference repetitions. All reference validation runs passed both coverage guards.

## 4. Baseline
Official source: https://github.com/chenleihaomars/Simulation-for-ACbc at 985cb1c074bf5c1eba937e0ea5e2f50d8bb4e357.
`code/bench_predict.R` computes the nearest-neighbor rank statistic and subtracts a regression estimate of its bias. Quadratic ridge regressions fit conditional indicator probabilities within each scored replicate. The independent calibration dataset supplies the bootstrap variance. Prediction uses BENCH_SEED and BENCH_CPUS; training only saves the run seed. Across reference repetitions, training took 0.11, 0.11, and 0.07 seconds. Validation prediction took 798.84, 877.43, and 706.02 seconds. Memory cap is 80 GB.

## 5. Interface (frozen)
`bash code/train.sh --out DIR` writes a seed marker. `bash code/predict.sh --model DIR
--input PATH --output FILE` produces CSV described by `bench/io_spec.md`.
Environment: BENCH_SEED, BENCH_CPUS, BENCH_DATA_DIR, BENCH_BENCH_DIR. Use `bench eval-dev`
and `bench validate-output` for local development. R 4.5.3, MASS, FNN, FOCI and jsonlite
come from the fixed supplied environment; scorer/validator use system Python standard library. Runtime dependencies are fixed.

## 6. Rules
Only `code/` is editable; `data/`, `bench/`, README, I/O signatures and the supplied runtime are frozen. Preserve the scientific settings above. No network, memorized answers, truth reconstruction from hidden seeds, or reading hidden labels. Resources are 16 CPUs and 80 GB memory. Research-scope and information-use rules are checked by patch audit.

## 7. Known pitfalls
The first binary dataset is a separate shared calibration input for its setting;
the baseline uses it for bootstrap variance. Output rows start with the following
dataset. Estimate each scored replicate from its own observations and this supplied
calibration input; do not use other scored replicates as training data. R matrices use column-major encoding. Do not average
squared errors across settings before taking square roots. Include null settings when
checking ECP but exclude them from mean RMSE. A failed coverage guard yields +Infinity with coverage diagnostics.

## Research contract

**Objective:** Estimate the fixed population Chatterjee graph-based dependence functional and its 95% intervals, minimizing positive-rho cell RMSE under coverage constraints.

**Fixed research objects:** The population dependence functional defined above, observed X/Y and normal-CDF experimental design remain fixed; rho itself is not the estimand. All 75 coverage cells, 60 positive-rho RMSE cells, their aggregation, supplied independent calibration dataset and coverage floors remain fixed. These keep the estimand and evaluation population comparable across methods.

**Editable methods:** Bias correction, nuisance regression/working models, basis, regularization, bootstrap/other variance estimation and fitting losses may change for the same population functional. Degree-two basis, ridge exponent, bootstrap sizes and z-based interval implementation are baseline estimation choices that may be retuned as part of a new estimator or variance method.

**Forbidden substitutions:** Do not substitute Pearson correlation, a latent rho estimate or another dependence measure; infer hidden rho/cell truth as predictions; or modify RMSE/coverage definitions. Only the scored replicate and its supplied calibration dataset may inform its estimate; no pooling other scored replicates. When retaining the supplied estimator, bootstrap and normal-interval construction, changing only lambda=n^(-0.85), m=floor(sqrt(n)), B=10000 or z=1.96 does not count as a method improvement. Do not fit bias-correction tables or other constants to extra simulations, including ones drawn from `bench/generator.R`.

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
