# Uncovering Bias Mechanisms in Observational Studies

## 1. Goal

**Research question.** Can the relationship between the estimated bias magnitude |b1(X)| and the prediction error of nuisance models fitted on an observational study (OS) reveal which mechanism causes the bias? The paper's Table 1 predicts a different sign pattern for each mechanism. In particular, the correlation with treatment-model error should be positive under confounding and zero under no bias, transportability and selection type 1.

**Experiment.** Paper Section 5 and Figure 3 run 200 synthetic RCT/OS experiments per mechanism. Each experiment fits nuisance models on fitting cohorts, then computes Pearson's r and its p-value on separate OS patients. A run counts when r has the predicted sign at p < 0.01. This task uses one cell as its metric, the treatment signal under confounding, and four cells where no signal is expected as false-positive limits. The signal must fire under confounding and stay quiet elsewhere.

**Your job.** Improve `rho_b1_A_detection` (higher is better): detect confounding by a positive Pearson correlation between estimated bias magnitude and treatment-model squared prediction error, at two-sided p < 0.01. Fit nuisance functions separately for each supplied RCT/observational experiment using the paper's synthetic binary-covariate setting. You may improve the estimator under the rules below.

## 2. Data

- `data/train/` explains the absence of cross-experiment training. Each experiment carries its own fitting cohorts.

- `data/val/input.npz` and `data/val/truth.npz`: 800 public experiments, 200 per mechanism, generated with public seed 99. Mechanisms are no bias, transportability, confounding, and selection type 1. The paper's selection type 2 is not included. Mechanism identity is present only in labels. `truth.npz` also holds latent probability tables and the baseline's per-experiment results (see `bench/io_spec.md`).

- `bench/generator.py` is the frozen distribution. Prediction inputs contain observations only; seeds, latent parameters and mechanism labels are unavailable.

- Every experiment has d=6 with five measured binary covariates. RCT and OS fitting cohorts have 50000 patients each; independent evaluation candidates have 40000 per cohort. Each statistic uses the first 2000 eligible OS candidates. See `bench/io_spec.md` for the exact masks and row layout.

- Only factual selected outcomes are visible; unselected A/Y values use -1. IDs and experiment order have no predictive meaning. Do not re-split, relabel, generate extra experiments or import outside data.

## 3. Metric and scoring

`bench/scorer.py` recomputes Pearson statistics from the submitted probability table and frozen observations. Bias magnitude is `abs((mu1_rct-mu0_rct)-(mu1_obs-mu0_obs))` at each patient's stratum. The treatment error is the squared difference between A and the OS treatment probability. The raw metric is the share of confounding experiments with r_A > 0 and p_A < 0.01. The score is that share if all four fixed FP constraints pass, otherwise **0**. Invalid probability certificates or forged statistics produce **Infinity**, which the harness rejects.

| Guard | Requirement | Enforcement |
|---|---|---|
| Treatment false positives, no bias | FP_A ≤ 0.04 | scorer |
| Treatment false positives, selection type 1 | FP_A ≤ 0.087 | scorer |
| Treatment false positives, transportability | FP_A ≤ 0.160 | scorer |
| Selection false positives, confounding | FP_S ≤ 0.142 | scorer |
| Statistics certificate | r/p recomputed from submitted models; tolerance 2e-7; identical significance decisions | scorer |
| Independent experiments and opaque IDs | Subset, order and ID changes preserve predictions | judge probe |
| Fit only on fitting cohorts | Altering evaluation outcomes cannot change fitted model probabilities | judge probe and patch audit |
| No oracle / extra simulation / mechanism lookup | No seed inversion, truth-table predictions, label lookup or cross-experiment pairing | patch audit |

Validation baseline: **0.685**, algorithm-seed standard deviation **0.0**. Its four false-positive rates are 0.015 / 0.055 / 0.115 / 0.100 in the table order above. The judge compares against its archived reference with relative improvement threshold `max(delta, 2*reference_std/abs(reference_mean))`; delta is 0.05. The unchanged method is deterministic given observations, with one algorithm seed and parity relative tolerance 1e-6. This does not remove Monte Carlo uncertainty in the generated experiments: each validation rate rests on 200 experiments, with a binomial standard error of about 0.009 at a rate of 0.015 and about 0.033 at a rate of 0.685. A validation rate below a cap does not guarantee the hidden rate is below it. Y0/Y1 detection rates are diagnostics, not extra score constraints.

## 4. Baseline

Official code: https://github.com/demireal/benchmarking-os at `a1fa292b2ad1e3215bfde9e4696a2201e129b528`. The provided source snapshot contains the original `synthetic/utils_estimate.py` and `synthetic/utils_data_gen.py`. LogisticRegression defaults, one-hot features, model fitting, b1 arithmetic and Pearson filtering are preserved.

Per-experiment fitting uses a `BENCH_CPUS` process pool with one BLAS thread per worker. `train.sh` writes configuration; `predict.sh` fits each experiment independently. There is no reusable baseline model between experiments. At 8 CPUs, reference train takes 3.03 s; the public validation prediction takes 62.29 s (from the reference run). The declared memory budget is 16 GB; this is a budget, not a measured peak.

## 5. Interface (frozen)

- `bash code/train.sh --out DIR` writes the model/configuration directory.

- `bash code/predict.sh --model DIR --input PATH --output FILE` writes CSV: one row per opaque experiment ID, 256 stratum probabilities and eight r/p statistics. Exact header and order are in `bench/format.py` and `bench/io_spec.md`.

- Environment: `BENCH_SEED`, `BENCH_CPUS`, `BENCH_DATA_DIR`, `BENCH_BENCH_DIR`, `BENCH_CODE_DIR`. Load frozen helpers by path from `BENCH_BENCH_DIR`.

- Python 3.12, CPU only. Only PyPI `name==version` pins are supported, with no editable, local or git dependencies.

## 6. Rules

The research scope below determines which components may change. Supplied data and splits, trusted tests, information boundaries and the evaluation metric remain fixed.

- Editable surface: nuisance probability estimators, training surrogate losses and fitting methods under `code/`, within the fixed causal/statistical research object and cohort boundaries. The table describes fitted conditional probabilities for every measured covariate pattern. It permits any estimator of those probabilities and does not require logistic regression.

- Frozen: `data/`, `bench/`, this README, script arguments and prediction format.

- Fit each experiment's outcome/treatment/selection probability functions using only its own fitting cohorts. Evaluation outcomes are used solely for the specified statistics, never to fit, select or alter the probability functions. Each fitted prediction is a function of measured covariates and cohort only. Public validation labels are for evaluation and method, hyperparameter or checkpoint selection only and must not be looked up during prediction. (Judge probe and patch audit.)

- Do not read or guess hidden truth, memorize labels or paper results, reverse generator seeds, generate extra simulations, identify mechanisms by matching experiments, or hard-code ID-specific predictions. A correct-looking oracle certificate does not exempt a patch from these rules. (Patch audit.)

- Runs use 8 CPUs and the declared 16 GB memory budget, without network access.

- The baseline uses one deterministic algorithm seed. Candidate randomness must honor `BENCH_SEED` and remain reproducible per experiment; process scheduling or input order must not change predictions. (Judge probe.)

## 7. Known pitfalls

The b1 definition is the fixed difference of estimated treatment effects, preserved from the released code; the paper's Eq. 18 writes b1 from the treated arm only. The synthetic RCT selection probability is 0.95; the paper's text says 1. These scientific definitions remain fixed for all methods. Pearson uses two-sided p-values but only positive r counts. The four FP limits are absolute, not baseline-relative. Constant bias/error vectors make Pearson undefined and are invalid.

`predict` does the expensive per-experiment fitting. Keep BLAS single-threaded when using a process pool, and load only the requested ID's NPZ member. The output path may have no extension. Scripts must resolve absolute arguments before changing directory and explicitly set PYTHONPATH under PYTHONSAFEPATH=1. Structural probes supplement the scorer; absence of a probe in a local development run does not waive patch-audit rules.

## Research scope

**Objective.** Detect the specified confounding mechanism using the original correlation between estimated bias magnitude and treatment-prediction squared error, subject to all fixed false-positive controls.

**Fixed research objects.** Synthetic causal mechanisms, observed/unmeasured covariate distinction, factual-selection masks and independent RCT/OS fitting/evaluation cohorts; b1 as the specified difference of estimated treatment effects, eligible-row ordering, two-sided Pearson tests with positive-significance rule and four FP caps. These define the question: mechanisms can only be told apart if every method is judged by the same statistic on the same cohorts.

**Editable methods.** Nuisance outcome/treatment/selection probability estimators on the measured covariates, fitting surrogates and numerical optimization; per-experiment statistical modeling using fitting cohorts only. The paper's consistency result (Theorem 4.8) assumes consistent nuisance estimators, not a specific model family.

**Forbidden substitutions.** Every table entry must be an estimate of the named conditional probability for that cohort and stratum; do not choose table values to shape the bias or error vector instead of estimating those probabilities. Do not infer the experiment's bias mechanism from the fitting cohorts and then set table values according to that inference; the mechanism may only be revealed by the fixed statistic on the evaluation rows. Changing the causal mechanisms, estimand, bias/error statistic, significance rule or cohort eligibility; fitting nuisance probabilities to evaluation outcomes, exploiting mechanism labels/oracle tables or pairing experiments; training or fitting reusable model state or learned preprocessing on validation features or labels, including final train+val refits or validation-based distillation.

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
