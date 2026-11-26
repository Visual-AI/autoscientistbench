# 041_ClusterSMART: longitudinal inference in clustered SMARTs

## Goal and baseline
Improve mean power of 95% confidence intervals for the end-of-study (1,1) versus (-1,-1) contrast while preserving coverage. A clustered SMART is a sequential, multiple assignment randomized trial. Treatment decisions apply to clusters; nonresponding clusters receive the later treatment decision. Each pair specifies the initial treatment and the later treatment for nonresponders. The target is the difference in marginal mean final outcomes under these adaptive interventions, not a contrast conditional on response.

The baseline is the authors' longitudinal estimator from three_lvl_cSMART_sim_study, commit c25df1882deb624215e217dc6f31b7fda0a9f68b. It uses repeated outcomes within people and clusters to improve end-of-study inference compared with using only final outcomes. Working mean/covariance models, fitting losses and interval methods may change while preserving the same end-of-study clustered adaptive intervention (cAI) estimand, SMART design and longitudinal observed-data boundary.

## Data
Each input is an independent observed SMART study. RDS batches contain observations, study structure and original fitting settings. The four fixed high-correlation environments compare inference across different effect settings. Public validation contains 600 studies per environment. There is no training split. Validation is for evaluation and method or hyperparameter selection only. Per-study fitting uses that study's observations only; do not pool validation cases to fit reusable models or preprocessing. True effects are evaluation targets, not method inputs.

## Metric and constraints
For each environment, power is the fraction of intervals strictly excluding zero; coverage is the fraction strictly containing the true contrast. The fixed coverage floors are g1: 0.9337, g2: 0.9247, g3: 0.9167 and g4: 0.899. These are the existing coverage targets after the 0.013 allowance. Score is the equally weighted mean power if every environment meets its floor; otherwise 0. Coverage is scorer-enforced. Complete prediction IDs, finiteness and interval order are validator-enforced. Using supplied observations, avoiding hardcoded answers and respecting compute require patch review. Measured validation power is 90.79166666666666%, with standard deviation 0.0 percentage points across the reference repetitions. PASS requires a relative improvement of at least 0.5% (delta=0.005), or the noise threshold if larger.

## Interfaces
`bash code/train.sh --out DIR` prepares reusable candidate artifacts (the baseline writes a training-free marker). `bash code/predict.sh --model DIR --input INPUT_DIR --output PRED.csv` estimates the specified contrast independently for each observed case. Output columns: id,estimate,lower,upper. See bench/io_spec.md.

## Runtime and compute
32 CPU threads, 144 GB RAM. The frozen R/Conda environment is supplied under BENCH_RUNTIME_DIR; no installation or network is needed. BENCH_CPUS controls case parallelism; BLAS uses one thread/worker. BENCH_SEED controls estimator randomness. Python validator/scorer need only the standard library. Across reference repetitions, the training-free marker took 0.06, 0.06 and 0.06 seconds. Validation prediction took 107.43, 87.14 and 77.92 seconds.

## Rules
Code may be changed; datasets, runtime and trusted bench files are frozen. Do not read hidden-test labels, access original reproduction results, hardcode sample-specific answers, or change evaluation cases. Validation may guide evaluation and method, hyperparameter or checkpoint selection only; every submitted method must return intervals for all cases.

## Verification and submission
Submit code changes with the normal benchmark patch workflow. Predictions must be produced by the submitted code from the supplied observations.

## Research contract

**Objective:** Improve power of valid intervals for the fixed end-of-study (1,1) versus (-1,-1) adaptive-intervention contrast in clustered SMART studies.

**Fixed research objects:** The intervention regimes, contrast, outcome time, SMART randomization/cluster structure and four supplied high-correlation environments define the research question. All supplied study observations, per-environment coverage floors, strict zero exclusion/inclusion and equal mean power remain fixed.

**Editable methods:** Working outcome/covariance models, weighting, nuisance fitting losses, estimation algorithms and interval construction may change for the same contrast. Three-level fitting iterations and covariance parameterizations are baseline methods, subject to the limit under Forbidden substitutions; the task does not ask to approximate a fixed Bayesian posterior.

**Forbidden substitutions:** Do not switch to another treatment/time contrast or predictive accuracy, change randomization structure or intervention definitions, or use oracle effects/hidden outcomes. When retaining the supplied multilayer fitting and inference routines, changing only max_iter=100, the convergence threshold 1e-5 or alpha=0.05 does not count as a method improvement. Switching only the existing `var_homo_over_time` or `correlation_structure` entries of `fit_settings`, or multiplying interval widths by a constant, also does not count. No case replacement, cross-study outcome pooling or changed coverage/power arithmetic. Interval endpoints must come from the stated inference method and the current study's estimated uncertainty. Do not clip intervals at zero, use the known sign of the effect, or move point estimates to force rejection of zero. Patch audit enforces this rule; alternative confidence-interval methods remain permitted.

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
