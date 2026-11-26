# Skew-symmetric approximations of posterior distributions

## 1. Goal

Estimate posterior means for high-dimensional Poisson regression with rate exp(X beta) and independent Student-t(3) coefficient priors centered at zero with unit scale. Improve the standardized absolute posterior-mean bias of the skew-symmetric Laplace baseline. The five original settings are n = 200, 400, 600, 800, 1000 and d = 60, 84, 102, 120, 132. You may change candidate code within the interface and resource guards below.

## 2. Data

`data/val/input.json` holds 25 archived observed X/y datasets, five per sample size. `data/val/labels.json` holds their archived long-HMC coordinate means and standard deviations, plus same-case baseline timing and plain-Laplace fallback values. There is no cross-instance training dataset; `data/train/` records this fact. Fit each case only from its own observed X/y. Validation is for evaluation and method or hyperparameter selection, without fitting reusable models or learned preprocessing. This is an archived-case evaluation, not a new independently simulated reference study. Do not re-split, relabel, regenerate reference labels, or import outside data. Prediction receives observed data only; fitting may not read reference labels, including public validation labels.

## 3. Metric and scoring

Lower `standardized_bias` is better. For each case, compute median_j |mean_q[j] - HMC_mean[j]| / HMC_sd[j]; average cases within each n and then average the n groups. The frozen scorer implements this formula.

- Every full fit must use at most 21,500 posterior evaluations, including optimizer, line search, Hessian and skew step. The scorer checks reported counts; truthful instrumented provenance requires patch audit.
- No reference accesses are allowed while fitting. The scorer rejects nonzero reported accesses; sandbox isolation protects private labels. Actual zero access, including public validation labels, requires patch audit.

A violation on any case sets the whole score to the archived same-split plain-Laplace aggregate. Invalid IDs, shapes, nonfinite numbers or missing telemetry are rejected. Passing requires relative improvement of at least max(0.005, 2*reference_std/abs(reference_mean)) against the reference. The measured validation reference has mean standardized_bias 0.0382760722 and standard deviation 0.0001192012 across three runs.

## 4. Baseline

The baseline is the original skew-symmetric Laplace estimator from [SkewAppr_Post](https://github.com/Francesco16p/SkewAppr_Post), upstream commit `06d1e597500757543b7678a3f2298e88d70aad8f`, with the existing metric implementation retained in `code/metric_original/`; its older evaluation cap is not current, and `bench/scorer.py` is authoritative. `code/predict.R` optimizes the unchanged Student-t Poisson model, requests its Hessian, draws 10,000 Gaussian proposals, evaluates the original skew function at each proposal and its reflection, and computes the original weighted mean. No fitted model or cached candidate prediction is included.

Model preparation (`prepare.R`) compiles the candidate model. Per-case fitting (`predict.R`) reads observations and writes JSON estimates and telemetry. The generated Stan counter is rebuilt on every clean train, so edits to `poisson_hd.stan` take effect. For validation seeds 1, 2 and 3, compilation took 52.76, 52.03 and 59.89 seconds; prediction over the full validation set took 131.32, 131.34 and 131.03 seconds, respectively.

## 5. Interface (frozen)

`bash code/train.sh --out DIR` prepares candidate artifacts in DIR; the baseline compiles its editable Stan implementation without fitting. `bash code/predict.sh --model DIR --input PATH --output FILE` fits each supplied case and writes the format in `bench/io_spec.md`. `BENCH_SEED` offsets each archived fit seed; seed 1 preserves the original random stream. `BENCH_CPUS` is the CPU budget. `BENCH_DATA_DIR` and `BENCH_BENCH_DIR` locate public data and frozen evaluation code. `bench eval-dev`, `bench validate-output FILE`, and `bench submit` use the same interface.

The environment provides R 4.5.3, rstan 2.32.7 and StanHeaders 2.39.1 with the compiler and libraries. Python scoring uses only the standard library. Dependencies are fixed by the provided runtime and `env.lock`.

## 6. Rules

Everything under `code/` is editable, including approximation families, auxiliary approximation losses, inference algorithms, optimizers and equivalent Stan implementation of the same target posterior. The scientific target posterior remains the supplied Poisson likelihood with independent Student-t(3) coefficient priors; candidate approximations may change without changing that target. `data/`, `bench/`, this README, script signatures and prediction format are frozen. Judge resources are 8 CPUs and 48 GB memory. The posterior-evaluation and reference-access guards above apply. Every run must honor its supplied `BENCH_SEED`. No network is available during compilation or fitting. Reference reads, memorized outputs, hard-coded predictions, and bypassed telemetry are disqualifying.

## 7. Known pitfalls

The skew step alone costs 20,000 posterior calls; excluding optimizer and Hessian calls understates cost. Absolute errors must be divided by coordinate HMC standard deviation before taking the coordinate median. The fallback is the archived plain-Laplace value, not infinity or an arbitrary penalty. The baseline's zero reference-access field is a claim that patch review must verify, not an independent access monitor. The compiler needs `-DTBB_INTERFACE_NEW` for the installed oneTBB. Rebuilding the model is intentional and must not be replaced by an archived fitted artifact.

## Research contract

**Objective:** Approximate posterior means for the fixed Poisson regression/Student-t(3) posterior under the existing posterior-evaluation budget.

**Fixed research objects:** The Poisson likelihood/link, independent Student-t(3) coefficient priors, observed X/y, coordinates and posterior-mean functional remain fixed. The HMC coordinate reference, standardized absolute bias then coordinate median/cell mean, 21500 actual posterior-evaluation cap and zero-reference-access rule remain fixed.

**Editable methods:** Posterior approximation family, numerical sampling/integration, auxiliary approximation loss, optimizer, reparameterization and implementation may change while targeting that posterior. Skew-symmetric Laplace and its proposal/kernel settings are baseline methods; a proposal-count change alone is covered under Forbidden substitutions.

**Forbidden substitutions:** Do not change likelihood, prior/scale, posterior temperature or coordinates, optimize predictive fit instead of the requested posterior mean, or use reference summaries during fitting. No fabricated posterior-call telemetry, cached outputs, cross-instance fitting or changed fallback/metric. When retaining the supplied skew-symmetric Laplace estimator, changing only the 10,000 Gaussian proposal count does not count as a method improvement.

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
