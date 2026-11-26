# The Bradley–Terry stochastic block model

## 1. Goal
Infer latent player skill groups and strength rankings under the specified Bayesian Bradley-Terry stochastic block model from directed paired-comparison win counts. Players in a group share a strength. A player wins with probability equal to its group strength divided by the sum of both groups' strengths. Improve mean adjusted Rand index (ARI) under the simulation design described below. The method under `code/` is editable; the observation-only interface and scoring rule are fixed.

## 2. Data
This is a synthetic per-network inference task with no population training split. `data/train/README.txt` explains that training builds the editable estimator. Fit each network only from its own W. Validation features and labels support evaluation and method or hyperparameter selection, not fitting reusable model state or preprocessing across networks. `data/val/input.json` contains 90 observed networks: the authors' conditional Gnedin-size generator, three ATP exposure designs (2000, 2015, 2024), K in {3,5,7}, ten replicates each, n=105, gamma_true=.71 and p_adj=.85. `data/val/labels.json` contains validation truth. Public generation seed base is 123 with the authors' formula; network order is shuffled.

`bench/generator.R` contains the frozen original generating functions; `bench/generator.py` documents the entry point. Prediction input contains only W, without truth, K, seed, replication metadata, or player names. Do not re-split, relabel, or import outside data.

## 3. Metric and scoring
`ari_minVI` is mean ARI, computed from output partitions and truth, exactly matching the installed `fossil::adj.rand.index`, including its absolute value and square contingency-table behavior when cluster counts differ. Strength quality is tie-aware Spearman correlation of estimated and true player strengths. The score is mean ARI if mean Spearman >= **0.8211555622531486**; otherwise it is 0. The ranking threshold is fixed. Fit times are reported as diagnostics.

Scorer checks: complete output coverage, valid finite partitions/strengths/times, exact ARI and ranking guard. Patch audit checks: estimates are computed from each network's observations, timing is honestly reported, no truth access or cached predictions, and the full dataset protocol. The Bradley-Terry likelihood, GN partition prior, block-strength prior and requested posterior summaries remain fixed. Posterior computation, approximation and numerical minimization of the fixed VI decision loss may be replaced. Partition output is the posterior minimum-VI estimate and strengths are the normalized posterior-strength summary, rather than arbitrary clustering/ranking predictions. These semantics require source audit in addition to numeric output checks. Reported timing and provenance are not authenticated by the prediction file.

Measured validation ARI is 0.7909380496766238 with standard deviation 0.0017127633386225679 across inference seeds. Improvement decisions use relative delta .005.

## 4. Baseline
Official simulation repository: https://github.com/laposanti/BT-SBM-Bradley-Terry-Stochastic-Block-Model at `1a96862a5c97baceb2417b4083de56d0642302d3`. Editable BTSBM package source is pinned to `069d31d134bcca8ad00e0a7f32fd49258c5ead60` from https://github.com/laposanti/BTSBM. The estimator is `code/BTSBM/R/gibbs_bt_sbm.R`, with its original Rcpp update code under `code/BTSBM/src/`.

The supplied baseline fitting uses 10,000 Gibbs iterations, 2,000 burn-in, GN prior, gamma=.71, a=2 and b=exp(digamma(a)), with authors' posterior relabeling, minVI from the posterior similarity matrix, and strength summary. `train.sh` compiles candidate BTSBM from source into its output directory every run; the frozen runtime deliberately excludes installed BTSBM. `predict.R` reads W and fits each network independently. Generation is offline. BENCH_SEED controls inference randomness independently of data generation. Unscored comparison models are not run.

Across three reference repetitions, measured training times were 6.07, 6.27, 6.12 seconds, and validation prediction times were 467.02, 466.17, 469.32 seconds. Only a separate 12-player, one-network, 400-iteration/100-burn-in fixture is used by the explicit smoke command. Default fitting is unaffected.

## 5. Interface (frozen)
`bash code/train.sh --out DIR` builds the editable R package into DIR. `bash code/predict.sh --model DIR --input PATH --output FILE` writes JSON in `bench/io_spec.md`. `BENCH_SEED` controls inference randomness; `BENCH_CPUS` bounds dataset workers; each worker uses one BLAS thread. `BENCH_DATA_DIR` and `BENCH_BENCH_DIR` identify read-only data and scoring code.

Use `bench status`, `bench eval-dev`, `bench validate-output`, and `bench submit`. The supplied read-only runtime provides R and compilers. `code/env.lock` and dependencies are immutable. No network access or package downloads are needed.

## 6. Rules
Posterior inference and numerical implementation under `code/` are editable within the fixed BT-SBM target and summary rules above, subject to the interface, observed-data provenance and timing honesty. `data/`, `bench/`, `README.md`, script signatures and prediction schema are frozen. Resources are 12 CPUs and 48 GB memory. Do not hard-code predictions, guess held-out truth, embed cached outputs, or read unprovided data.

## 7. Known pitfalls
The archived fossil metric differs from standard signed ARI; the frozen scorer preserves its original implementation. A partition must contain player group assignments, not averaged labels. The selected outputs are the minVI partition and normalized posterior strengths defined in `fit_helpers.R`; candidates may improve their posterior approximation and numerical computation. Keep output network/player order identical to input. Fit time records inference elapsed time, excluding compilation. Constant strengths give undefined Spearman and score 0. Small smoke ARI or guard failure says nothing about full performance. Source `.o` and `.so` files are discarded before compilation; an installed frozen BTSBM is never an estimator fallback.

## Research contract

**Objective:** Improve inference and posterior minimum-VI partition recovery for the specified Bayesian Bradley-Terry stochastic block model while retaining its posterior strength-ranking output.

**Fixed research objects:** The BT binomial pairwise-comparison likelihood, Gnedin partition prior with gamma=.71, and Gamma block-strength prior a=2, b=exp(digamma(2)) are fixed. The minVI Bayes decision loss and normalized posterior-strength summary in fit_helpers.R define the outputs; data/exposure designs, literal fossil ARI and Spearman threshold .8211555622531486 remain fixed. VI is the sum of the two partition entropies minus twice their mutual information; minVI minimizes its posterior expectation. The strength summary averages retained posterior player strengths, then divides by their geometric mean. These fixed objects keep comparisons focused on inference quality.

**Editable methods:** Posterior sampling or approximation, parameterization, relabeling implementations, optimization and numerical search for the same minVI decision may change. Gibbs iteration counts, burn-in and scheduling are baseline computation choices; unscored comparison models need not run.

**Forbidden substitutions:** No replacement with a different likelihood, partition/strength prior, arbitrary community algorithm, alternative partition decision loss or unrelated strength ranking. Do not use hidden design/K/strength labels, pool independent networks or alter metric/coverage. When retaining the supplied Gibbs sampler, changing only T_iter=10000 or T_burn=2000 does not count as a method improvement. Partition labels must be the consecutive integers from 1 through the number of predicted groups. Do not move labels outside this range to exclude groups from the metric.

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
