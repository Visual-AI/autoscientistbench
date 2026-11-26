# Assumption-lean post-integrated inference with surrogate-control outcomes

## 1. Goal
Improve `power` (higher is better) for post-integrated inference of covariate effects on binary outcomes with latent confounding. Each instance uses the frozen author-code simulation in `bench/generator.py`: n=100 observations, p=1000 outcomes, 500 known surrogate controls, r=10 latent factors, effect sparsity s=0.2, and latent noise sigma_eps=0.8. Estimate effects and sampling variances for the remaining outcomes and report valid two-sided p values. You may change the method under the rules below.

## 2. Data
- `data/train/README.md`: there is no reusable training dataset. Validation is for evaluation and method, hyperparameter or checkpoint selection only. The task permits fitting separately to each instance using only its own observed X, Y and C, without evaluation targets or fitted state from other instances.
- `data/val/input.npz` and `data/val/truth.npz`: 20 public validation instances, generated with seeds 100 through 119; each input supplies observed X, Y, known control indices C and opaque identifiers. Labels supply coefficients, latent factors and public seeds.
- Final evaluation data are not available in this workspace.
- `bench/generator.py` freezes the authors' data-generating process. Predict receives observations only. Do not infer generating seeds, regenerate instances, simulate extra labeled data, re-split or relabel the supplied data. Each prediction must depend only on its own instance's observations.

## 3. Metric and scoring
`bench/scorer.py` computes mean replication power: the fraction of non-null tested outcomes with p_value < .05, averaged equally across instances. Mean type-I error uses null outcomes; mean BH FDR averages the false discovery proportion after BH adjusted p_value < .05, with zero FDP for no discoveries.

| Guard | Enforcement | Failure |
|---|---|---|
| Mean type-I error <= .069 | scorer | power = 0 |
| Mean BH FDR <= .085 | scorer | power = 0 |
| Finite beta, positive variance, valid p values, exact ID set | validator and scorer | invalid / Infinity |
| Each instance is processed independently; subset and order consistency | patch audit | invalid |
| No true parameters, hidden seed recovery, memorized answers or extra simulations in inference | patch audit | invalid |

The literal error-rate rule returns power when both thresholds hold, otherwise zero. Measured reference validation power is **0.5228495701**; type-I error 0.0651561223, BH FDR 0.0746747375. Both guards pass. Across-run standard deviation is 0 for this deterministic baseline; validation replication standard deviation is 0.112962 and standard error 0.025259. PASS requires relative improvement of at least 8%, or the larger variance-based threshold reported by `bench status`.

## 4. Baseline
Official code: https://github.com/jaydu1/PII at commit `7ee99bf1e352f875500b6def37bc35ac74a7ed1e`. `code/PII.py` implements PII, with PCA embedding of surrogate controls and three random-forest nuisance fits tuned by ECV. Author defaults are unchanged: crossfit=1, logit link, M=M0=25, M_max=100, depths 1/3/5, and subsampling fractions .25/.5/.75/1. The original Lasso alpha=1e-4 and variance regularization are retained.

`train.sh` saves configuration; all per-instance fitting occurs during `predict.sh`. The adapter loads observed data, saves CSV predictions and runs independent workers. Measured runtime in one reference run on 16 CPUs: train 3.42 s; validation predict 20.82 s. Memory allocation is 16 GB; peak memory was not measured.

## 5. Interface (frozen)
- `bash code/train.sh --out DIR` writes the configuration required for prediction.
- `bash code/predict.sh --model DIR --input PATH --output FILE` writes CSV `id,beta,var,p_value`; see `bench/io_spec.md`.
- Environment: `BENCH_SEED`, `BENCH_CPUS`, `BENCH_DATA_DIR`, `BENCH_BENCH_DIR`, `BENCH_CODE_DIR`. Paths passed by the harness are absolute.
- `bench status`, `bench eval-dev [--seed S]`, `bench validate-output FILE`, `bench submit [--final] [-m MSG]`.
- Dependencies are installed with `uv pip install`; submission freezes PyPI `name==version` pins in `code/env.lock`. Editable, git and local packages cannot be rebuilt.

## 6. Rules
Methods under `code/` are editable, including nuisance working models and surrogate losses, embedding/effect estimation, uncertainty estimation, test statistics and p-value calibration, while retaining the specified logit-link projected direct-effect estimand, the surrogate-control information boundary, interface and statistical guards. The baseline uses two-sided Student-t calibration with df=99; other valid calibration methods are allowed. `data/`, `bench/`, this README, script arguments and output format are frozen.

The CPU budget is 16 and memory allocation 16 GB. No network is available in train or predict. The judge uses one run for this deterministic baseline; honor BENCH_SEED for any introduced randomness. Do not read or guess hidden instances, hard-code true coefficients, memorize paper results, exploit validation labels at prediction time, recover generating seeds, or generate additional simulated training instances. No cross-instance fitting or sharing of information during prediction is allowed.

## 7. Known pitfalls
PII fitting is the prediction stage, so prediction is much more expensive than train. Use at most BENCH_CPUS worker processes and one BLAS/joblib thread per worker; uncapped nested ECV parallelism is slow. The original package fixes forest randomness internally; accepting BENCH_SEED does not override those author defaults. The baseline is deterministic given observations.

The control indices are provided in their original order; output hypotheses are in ascending complement-of-C order and must use the supplied hypothesis IDs. Never treat all 1000 outcomes as tested hypotheses. Use a file handle if writing NumPy output elsewhere, since the harness output path need not have an extension. Scoring guards are absolute and cannot be relaxed to validation-relative thresholds.

## Research contract

**Objective:** Improve testing power for the specified logit-link direct-effect hypotheses using only observed covariates, binary outcomes and known surrogate controls. For outcome j, the target is beta_j = E[Cov(X, logit(E[Y_j | X, U]) | U)] / E[Var(X | U)], where U denotes the unobserved latent factors. Test whether this effect is zero. Surrogate-control methods use an estimated embedding in place of U. Compare candidate methods with the supplied PII baseline under the same observation boundary.

**Fixed research objects:** The selected generalized partially linear/logit effect parameter and zero-effect hypotheses, observed X/Y and supplied 500 control indices define the scientific inference problem; latent U is not an input. The complementary 500 tested outcomes, finite beta/positive sampling variance, valid two-sided p-values, power/type-I/BH-FDR rules and independent-instance protocol remain fixed.

**Editable methods:** Surrogate-control embeddings, nuisance working models and surrogate losses, semiparametric effect estimators, variance estimation, cross-fitting and valid p-value calibrations may change. PCA, ECV-tuned forests and t(df=99) are baseline estimation/calibration choices; alternative methods must estimate/test the same logit-scale effect rather than change the estimand.

**Forbidden substitutions:** Do not change the link/scale to marginal mean association or another causal effect, use latent U/true coefficients, test the known controls as alternatives, or submit unrelated beta/variance/p-values. No hidden-seed recovery, additional labeled simulations, per-instance truth lookup or changed BH/type-I/power definitions. When retaining the supplied PII pipeline, changing only the variance ridge 1e-4, Lasso alpha=1e-4, r=10, crossfit=1, M=M0=25, M_max=100, max_depth 1/3/5, max_samples .25/.5/.75/1 or df=99 does not count as a method improvement.

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
