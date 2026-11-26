# Parameterizing the effect of a continuous treatment using average derivative effects

## 1. Goal
Estimate the average least squares effect (ALSE, Psi) from independent observed datasets produced by the fixed structural equation model in `bench/generator.R`. Improve the guarded scaled mean squared error (lower is better). Each dataset receives an ALSE point estimate and standard error under each of two nuisance-regression conditions: a generalized additive model (`gam`) and a random forest (`ranger`). These learner families are fixed experiment conditions, not interchangeable output labels. Within each family, fitting algorithms and valid ALSE inference procedures may change, with tuning as part of such a change; the same replacement learner cannot be relabeled into both conditions.

## 2. Data
There is no population-level training dataset. Validation is only for evaluation and method or hyperparameter selection. Per-instance nuisance fitting uses only that dataset's observations; do not carry fitted models or learned preprocessing across independent cases. `data/train/README.txt` records this convention. `data/val/input/` contains 320 independently generated datasets: 40 at each n=500,1000,...,4000. Each RDS dataset contains covariates X1,X2,X3, continuous treatment A and outcome Y. The JSON manifest identifies cases and their sizes. `data/val/labels.json` contains separate evaluation truths, never fitting targets. The validation generation seed is 181001; `bench/generator.R` is the frozen SEM generator. Inputs contain neither truth nor generation seeds. Do not re-split or substitute data.

## 3. Metric and scoring
`scaled_mse` is the equal mean of the 16 (n,learner) cell MSEs after multiplication by n. The fixed Wald interval is estimate ±1.959964*standard error. The score is +Infinity unless pooled coverage is at least 0.92 and coverage in every cell is at least 0.85. Scorer-enforced checks require every expected case and both learners exactly once, finite estimates and nonnegative finite standard errors, and the complete default sample-size/replication grid. The scorer recomputes all statistics from predictions and separate truth labels. Reading observations, performing the declared estimator, retaining the two nuisance families and avoiding hard-coded answers are patch-audit requirements. Output labels alone do not verify learner-family compliance. The measured validation scaled MSE is 3.454153, with a standard deviation of 0.006009 across reference repetitions. The measured validation runs satisfy both coverage guards. The improvement requirement remains delta=0.005.

## 4. Baseline
Official source: https://github.com/ohines/alse at commit `12376e5ab5c10d10950bcbc2fadd1d2df7d3a67b`. The GAM uses smooth main effects and pairwise interactions. Ranger uses 1000 trees. For each learner, `code/method.R` calls `alse_analysis` and computes both no-sample-splitting (noSS) and five-fold sample-splitting estimates. It emits `psi_0_1` and `std_err_0_1` from noSS. In this branch, the baseline centers A and Y using their fitted conditional means, divides the sum of residual products by the sum of squared treatment residuals, and computes an influence-function standard error.

Case RNGs are derived from BENCH_SEED. Workers are limited by BENCH_CPUS, and each ranger fit and prediction uses a single thread. Protocol recording took 0.06 seconds in each reference repetition. Validation prediction took 890.36, 896.83 and 891.92 seconds. The default CPU budget is 36, memory cap 144 GB.

## 5. Interface (frozen)
Run entry points from `code/`. `bash train.sh --out DIR` records the protocol; `bash predict.sh --model DIR --input PATH --output FILE` estimates ALSE from each observed dataset and writes CSV. See `bench/io_spec.md` for exact fields. BENCH_SEED controls estimator randomness; BENCH_CPUS limits workers; BENCH_DATA_DIR and BENCH_BENCH_DIR locate read-only data and protocol files. `bench status`, `bench eval-dev`, `bench validate-output`, and `bench submit` use these interfaces. The supplied runtime provides R 4.2.2, mgcv 1.9-1, ranger 0.16.0 and their dependencies. The Python scorer uses only the standard library. `env.lock`, dependencies and the runtime are fixed.

## 6. Rules
The candidate surface is `code/`; methods may be improved, with hyperparameters retuned as part of a method change, while preserving script arguments and output format. Data, bench files, README and prediction schema are frozen. Memory: 144 GB. There is no network. Reading hidden labels, embedding cached source results, guessing test cases or hard-coding the known estimand as a prediction is disqualifying.

## 7. Known pitfalls
The full protocol uses 40 datasets per size. Both learners share each observed dataset; random streams are fixed per case and independent of worker scheduling. Estimates and standard errors must target ALSE, not a different weighted derivative estimand. The baseline selects the noSS branch; sample splitting, cross-fitting and other valid ALSE procedures are allowed.

## Research contract

**Objective.** Improve ALSE (Psi) estimation and uncertainty quantification under the two specified nuisance-learning conditions, GAM and random forest.

**Fixed research objects:** The target is Psi = E[Cov(A,Y|X)] / E[Var(A|X)], a variance-weighted average of conditional least-squares slopes. The unweighted target psi = E[Cov(A,Y|X) / Var(A|X)] is distinct and is not scored. Both nuisance-learning conditions (GAM and random forest), the eight sample sizes, each dataset, fixed Wald multiplier 1.959964, equal 16-cell n*MSE and coverage floors remain fixed.

**Editable methods:** Nuisance fitting implementations, bases and smoothing within GAM, forest construction within the forest family with tuning as part of such a change, numerical optimization and variance estimation may change. Valid ALSE estimation, sample splitting and cross-fitting may change while retaining both nuisance families; auxiliary fitting losses cannot redefine ALSE.

**Forbidden substitutions:** Do not replace the GAM/forest experimental conditions with arbitrary neural predictors or one duplicate learner; their names are not cosmetic slots. Do not substitute psi/another weighted derivative target, use known true Psi as a prediction, pool independent datasets or change the Wald/scoring rule. When retaining the supplied ranger nuisance-fitting routine, changing only num.trees=1000 does not count as a method improvement. Only switching to the five-fold sample-splitting results the baseline already computes (`psi_0_2`, `std_err_0_2`) also does not count.

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
