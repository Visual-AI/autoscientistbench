# Bayesian Nonparametric Modeling of Latent Partitions via Stirling-Gamma Priors

## 1. Goal
Infer the posterior distribution of the number of occupied clusters K_n from 500 observed bivariate normal measurements. The research object is the Dirichlet process mixture posterior with the alpha ~ Sg(1, 0.25, n) prior defined in the Research contract. Improve the guarded mean posterior probability assigned to the true cluster count.

## 2. Data
This is a synthetic task with no training split for reusable model fitting. `data/train/` is empty. Validation is only for evaluation and method, hyperparameter or checkpoint selection. Per-instance posterior inference may use its own observed coordinates, but never its true-K label or fitted state from another dataset. `data/val/input.json` contains 60 independent datasets, with 20 replicates for each true K in {2,4,6}; each dataset has n=500 rows. `data/val/labels.json` provides ID/true-K pairs and `data/val/seeds.json` records public generation seeds. `bench/generator.R` freezes the original equally weighted mixture generator, component means, and Sigma=diag(0.15,0.15). Prediction input contains only observations and opaque IDs, never seeds or true parameters. Do not change the fixed split or use external data.

## 3. Metric and scoring
The metric is `posterior_probability`, higher is better. For each true K=2,4,6 the scorer averages P(K_n=K|data) across replicates. The score is the mean of these three averages if all three are at least 0.835, 0.550, and 0.458 respectively; otherwise it is zero. These exact cell floors are scorer checks. The validator checks complete unique IDs and finite normalized posterior probabilities. Observed-input-only inference and absence of memorized predictions are patch-audit constraints. The submitted probabilities must approximate the fixed posterior defined in the Research contract, not an arbitrary calibrated cluster-count predictor. The likelihood and priors define that posterior and remain fixed; posterior algorithms and numerical approximations may change. The output validator cannot prove posterior fidelity, so the judge must audit the implementation as well as score the probabilities.

The measured validation reference has mean 0.6553133333 and standard deviation 0.0013668584. Across three reference repetitions, package-build times were 11.73, 11.29, 11.43 seconds and validation inference times were 35.62, 33.82, 34.28 seconds.

## 4. Baseline
The upstream package is [ConjugateDP](https://github.com/alessandrozito/ConjugateDP), commit `719a6804a4ee03c125d025de4fd59280a0d2b190`. Reproduction source `sim/common.R` and `sim/dpm_gibbs.cpp` implement the collapsed marginal Gibbs sampler (Neal Algorithm 3). The supplied baseline uses n=500, 15,000 iterations and 5,000 discarded burn-in draws, NIW base settings, initialization, and Sg(1,0.25,n). The observed n=500, Gaussian mixture likelihood, NIW base measure and Sg(1,0.25,n) precision prior remain fixed. The chain length, burn-in, initialization and collapsed Gibbs implementation are baseline numerical choices. The other two comparison priors remain in source but are not run for this metric.

`train.sh` rebuilds editable ConjugateDP R/C++ source into its model directory. `predict.sh` loads that build, recompiles the candidate Gibbs sampler, and performs independent posterior inference using only each dataset's observations. The copied installed package is never substituted for candidate source. `BENCH_SMOKE=1` uses 150 iterations and 50 burn-in draws for pipeline checks only.

## 5. Interface (frozen)
`bash code/train.sh --out DIR` builds the package. `bash code/predict.sh --model DIR --input PATH --output FILE` fits and emits posterior probabilities. Output format is specified in `bench/io_spec.md`. `BENCH_SEED` controls chain randomness, `BENCH_CPUS` caps process parallelism, and `BENCH_DATA_DIR` / `BENCH_BENCH_DIR` provide fixed data/tool locations. Useful commands are `bench status`, `bench eval-dev`, `bench validate-output`, and `bench submit`.

The fixed runtime supplies R 4.5.3, compilers and pinned R packages. Python tools use only the system standard library. `code/env.lock` and runtime dependencies are immutable.

## 6. Rules
Candidate changes are limited to `code/`, including method and hyperparameters within the Research contract, while preserving the interface. Frozen surfaces are `data/`, `bench/`, this README, runtime dependencies, script signatures and prediction format. Resources are 8 CPUs and 48 GB memory. Network access is disabled. Reading hidden truth, guessing IDs, hard-coding probabilities, or using cached inference outputs is disqualifying.

## 7. Known pitfalls
A short chain can fail one or more cell floors and score zero even when the pipeline works. Full validation evaluation requires results for all sixty public datasets. Do not treat the smoke score as a reference or reduce default replication counts. Candidate changes to package code are rebuilt during train; edits to `sim/dpm_gibbs.cpp` are rebuilt at predict. All posterior probabilities must describe the same normalized distribution over K=1,...,500.

## Research contract

**Objective:** Approximate the specified Stirling-gamma Dirichlet-process-mixture posterior of occupied cluster count K_n, improving its guarded true-count posterior probability.

**Fixed research objects:** Each observed y_i follows N(mu_i,Sigma_i), with (mu_i,Sigma_i) independently drawn from G and G ~ DP(alpha,G0). The base measure G0 is N(mu;0,Sigma/2) IW(Sigma;2,I). The precision prior is alpha ~ Sg(1,0.25,n). For positive alpha, its density is proportional to [Gamma(alpha)/Gamma(alpha+n)]^0.25. These define the fixed posterior. All 500 observed rows per independent dataset, the normalized K=1..500 output distribution, three cell floors, and equal cell aggregation remain fixed.

**Editable methods:** Any valid inference or posterior approximation algorithm, proposal, variational family, numerical solver, initialization and convergence strategy may replace the baseline Gibbs implementation. Auxiliary losses may improve approximation of this same posterior; Monte Carlo length and burn-in are numerical choices of such a replacement.

**Forbidden substitutions:** Do not replace the likelihood, base measure or precision prior, tune them against true K, or submit a generic classification distribution instead of the specified posterior. No hidden truth, data substitutions, cross-dataset pooling or ID-derived predictions. When retaining the supplied Gibbs transition kernel, changing only niter=15000, burnin=5000, init_type=1 or alpha_init=1 does not count as a method improvement.

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
