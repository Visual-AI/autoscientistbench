# Estimating the false discovery rate of variable selection

## 1. Goal
Estimate the false discovery rate (FDR) of a fixed Lasso selector from each observed X/y dataset. The model is y = X beta + Gaussian noise. A null variable has a zero coefficient. Realized FDP is the fraction of selected variables that are null, or zero when none are selected. FDR is its expectation over repeated responses conditional on X. Improve MAD (lower is better) over IID normal, X-AR, Coef-AR and Sparse designs, with n=1500, p=500, 30 nonzero coefficients, ten original lambda values and 120 replicates per design. These designs cover independent Gaussian features, correlated Gaussian features, correlated regression coefficients and sparse binary features.

## 2. Data
No training corpus is supplied; `data/train/` documents this. `data/val/input/` contains 480 complete observed X/y instances and their lambda grids; `data/val/labels/` contains separate beta labels and a second observation copy for frozen oracle scoring. Validation uses the original public replicate seeds 1 through 120. Validation is only for evaluation and method, hyperparameter or checkpoint selection. Do not fit reusable models or learned preprocessing on validation. Per-instance estimation and nuisance fitting may use that instance's X/y observations, without carrying fitted state between instances. `bench/generator.R` and `bench/original_utils.R` freeze the observation-generating functions. Prediction input has only observations and public design/grid metadata. IDs are random tokens unrelated to generation seeds. Do not resplit, relabel or add outside data.

## 3. Metric and scoring
MAD is the mean absolute bias across all 40 design/lambda cells, where bias = mean_replicates(hFDR - realized FDP). The scorer independently fits the original Lasso to observed X/y before calculating FDP against the separate null labels.

Scorer-enforced guards: every cell bias must be at least -3 times the paired-difference Monte Carlo standard error, and every design's mean sample SD(hFDR) across its ten lambdas must be at most 1.2 times the archived baseline mean SD. Baseline SDs are IID_Normal 0.0442960371119491; X_AR 0.0498058847364376; Coef_AR 0.0456158418078810; Sparse 0.0469046668969414. The full 4×120×10 protocol is required; any failure gives +Infinity. Returned hFDR bootstrap standard errors do not determine this guard. Frozen protocol settings and absence of label access also require patch audit.

Measured validation MAD is 0.019046607923255434, with standard deviation 0.00003491967592533255 across reference repetitions. All validation repetitions passed the statistical guards.

## 4. Baseline
Official source: https://github.com/yixiangLuo/hFDR_expr at fc139b1736fd86a9ecf8dec517c1888b0c6b79e1; estimator package hFDR at aab14e171b16aa56ecc32b75a7c9eb7ea8562338. `code/R/methods.R::calc_hFDR` calls the Gaussian linear/Lasso estimator with psi=pval, standard errors enabled, n_sample.hfdr=20, n_sample.se=10 and one inner worker. The baseline sums null-conditional expected selection fractions weighted by normalized p-value indicators, with a default cutoff of 0.1. Its Lasso branch uses homotopy integration and randomized approximations for small contributions. The generic n_sample.hfdr argument is not used by this specialized branch; bootstrap diagnostics are computed separately.

The baseline reads the supplied observations and parallelizes across instances. `code/train.sh` records the seed and fits no global model. `code/predict.sh` computes each estimate from that instance's X/y and supplied lambda grid. BENCH_SEED controls method randomness independently of observation generation. The supplied calibrated signal strengths and lambda grids are fixed.

Across reference repetitions, seed recording took 0.16, 0.12 and 0.13 seconds. Validation prediction took 758.23, 815.42 and 767.34 seconds, respectively.

## 5. Interface
`bash code/train.sh --out DIR` and `bash code/predict.sh --model DIR --input PATH --output FILE` implement the frozen signatures. BENCH_SEED controls estimator randomness; BENCH_CPUS caps replicate workers. Each worker uses one BLAS thread. BENCH_DATA_DIR and BENCH_BENCH_DIR locate the data and frozen benchmark. See `bench/io_spec.md` for exact prediction JSON.

The supplied runtime includes R 4.5.3, glmnet 5.0 and hFDR 0.1.0. System Python runs the standard-library wrappers. The runtime snapshot and empty Python env.lock are fixed; no installation or downloads occur at runtime.

## 6. Rules
Only code is editable. Candidate FDR estimators, auxiliary working models and losses, uncertainty calculations and numerical methods may be replaced while estimating the FDR of the same frozen Lasso selector. The Lasso selection rule whose FDR is estimated is part of the fixed scientific target. Data, scorer, validator, generator, calibrated grids, baseline guards and documentation are frozen. Preserve script signatures and prediction schema. The CPU allocation is 32 and memory budget is 144 GB. Reading or reconstructing held-out labels, hard-coding estimates, storing cached predictions, or bypassing the observation boundary is disqualifying. Runtime dependencies are fixed. Network is unavailable during evaluation.

## 7. Known pitfalls
hFDR is an FDR estimator, not an oracle FDP predictor: its output can exceed one, and conservative bias matters. The paired Monte Carlo SE is computed across replicate errors; bootstrap hFDR_se is a different quantity. The baseline enables the expensive original standard-error calculation; `hFDR_se` is optional diagnostic output. A finite score requires the full protocol and every statistical guard.

## Research contract

**Objective:** Estimate the false discovery rate of the specified Lasso selector in the fixed-X Gaussian linear experiment, minimizing guarded bias MAD.

**Fixed research objects:** The selected-variable rule is the frozen original Lasso at the supplied lambda grid, including its scaling and selection convention; the target is its FDR, not another selector or per-realization oracle FDP. Fixed X/y, four designs, 120 replications/design, 40-cell MAD, paired MC bias guard and baseline-SD guard remain fixed.

**Editable methods:** FDR estimation theory/algorithm, auxiliary nuisance models, surrogate losses, homotopy integration, bootstrap/variance methods and numerical optimization may change. The method need not execute the original hFDR estimator or optional bootstrap diagnostic.

**Forbidden substitutions:** Do not switch the evaluated Lasso to ridge, elastic-net, a different lambda or another feature-selection rule; change null definitions or estimate a different error rate. No true coefficient labels, modified design matrices, cross-replicate truth pooling or altered scoring. When retaining the supplied hFDR estimator, changing only the p-value cutoff threshold=0.1 does not count as a method improvement. Do not fit correction tables or other constants to extra simulations, including ones drawn from `bench/generator.R`.

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
