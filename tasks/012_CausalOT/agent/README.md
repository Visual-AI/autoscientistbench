# Statistical Inference in Causal Partial Identification under Smooth Densities

## 1. Goal

**Research problem.** The two potential outcomes of a unit are never observed together, so E||Y(0)-Y(1)||^2 is only partially identified. Its lower bound given covariates Z is the conditional optimal transport (COT) value: the minimum of E||Y(0)-Y(1)||^2 over joint laws that match the observable laws of (Y(0),Z) and (Y(1),Z). Equivalently, it is the average over Z of the squared 2-Wasserstein distance between the two conditional laws of Y given Z. The paper asks how to carry out valid inference for this value with multivariate outcomes, where quantile methods do not apply and other COT estimators have non-negligible bias or no established limit distribution. Its answer assumes smooth densities: plug wavelet density estimates into the COT functional, prove asymptotic normality at rate sqrt(n), and estimate the variance by bootstrap.

**Experiment.** The paper's inference experiment measures empirical coverage of nominal 95% intervals on synthetic data with a known target across (dY,dZ,n). Table 2 reports coverage only and has no competing method. This task uses one configuration of that table, (dY,dZ,n)=(2,1,3000), and adds interval length as the objective: a candidate is compared with the baseline on interval length under a fixed coverage requirement. Unlike the paper, which repeats one distribution per configuration, every instance here has its own distribution and therefore its own target value.

**Task.** For each instance, estimate the COT value from two groups of observed outcomes and covariates, and report a 95% confidence interval. Improve `ci_length_gmean` (lower is better) on hidden instances drawn from the same family of distributions as validation, while satisfying the coverage requirement. You may improve the estimation method subject to the rules below.

## 2. Data

- There is no training split. `data/train/README.md` describes this training-free protocol. Each instance is fitted independently from its own observed input during prediction. Validation may be used for evaluation and for method, hyperparameter or configuration selection; no model may be fitted on it.
- `data/val/input.npz` and `data/val/labels.npz` contain 100 public validation instances. Instance IDs and ordering carry no information about the observations.
- Hidden evaluation observations are 200 instances drawn independently from the same family and held by the judge. They cannot be read from this workspace.
- Input contains only opaque IDs and Y0, Z0, Y1, Z1 observations; `truth` is a label. Every instance has (dY,dZ,n=m)=(2,1,3000). Covariates are independent Uniform(0,1). Given Z=z, each group's outcome has a smooth conditional density that changes smoothly with z. The distributions differ between instances and their parameters are not disclosed. Outcomes are clipped to [0,1]; the distributions are drawn to stay inside that range, so this touches well under 0.1% of values.
- Labels are for offline diagnostics and configuration selection. Do not fit a predictor of the target from them, infer hidden seeds, generate additional simulated instances, or combine observations across instances during prediction. Do not re-split, relabel, or add outside data. Bootstrap resampling and internal integration draws from distributions fitted to the supplied observations without a parametric assumption are allowed; simulating from a population model you assumed or reconstructed is prohibited.

## 3. Metric and scoring

`ci_length_gmean` is the mean interval length (the name is historical), provided empirical coverage is at least **0.80 on test and 0.75 on val**. Otherwise the score is **Infinity**. Coverage includes interval endpoints. The floors are fixed numbers and sit below the nominal level because the baseline itself falls short of 95%: on validation its coverage is 0.85 to 0.90, and its point estimate is above the target on about 78% of instances, by about 0.75 bootstrap standard errors on average. The floors are a lower limit on the outcome, not the target: intervals must still aim at nominal 95% coverage (section 6 and Research scope).

The released code calls point ± normal.ppf(.975) × std(bootstrap estimates, ddof=1) a "basic" interval. The baseline uses this normal approximation with 100 bootstrap resamples. Alternative estimators and interval constructions may use their own internal resampling strategy on the same observations. The scored target `truth` is the instance's COT value for its distribution before clipping.

| Guard | Enforcement |
|---|---|
| Coverage at least 0.80 on test, 0.75 on val | scorer |
| Exact ID set, finite predictions, ordered interval | scorer and shared validator |
| Each prediction uses only its own observations and a fixed configuration selected on validation | patch audit |
| No truth-table lookup, model-to-truth mapping, label fitting, hidden-seed inference, hard-coded outputs or extra simulation | patch audit |

An interval centred on the true value cannot be told apart from an honest one by coverage alone, so the observations-only rules rest on patch audit. There is no automatic numerical oracle detector.

Measured baseline on validation at algorithm seeds 1, 2 and 3: **0.00378545647698704**, **0.00374911009247535** and **0.00376072570145422**; mean **0.00376509742363887**, seed standard deviation **0.0000151569320315326**. Coverage is 0.90, 0.86 and 0.85. Each instance moves coverage by 0.01, so differences of this size are within noise.

The judge evaluates three algorithm seeds and compares their mean with the reference measured under the same protocol. PASS requires relative improvement of at least max(delta, 2 × reference seed standard deviation / reference mean); delta is **3%**. All seeds must pass the coverage guards.

## 4. Baseline

Official code: https://github.com/siruilin1998/causalOT_inference at commit `309641bc4eb683fef7e5c55911f512b80a602cd1`. The paper is the public preprint arXiv:2602.20681v1. `code/estimator.py` extracts the original Table-2 notebook's estimator and bootstrap function bodies unchanged; the notebook is in `code/causal_OT_for_FIG3TAB2/`.

The baseline fits wavelet density estimators using db4, periodization, J_joint=4 for each group's (Y,Z) density, J_z=6 for the pooled Z density, no threshold, nonnegativity and renormalization. Each COT computation draws Nz=120 covariate values from the pooled Z estimate, draws Ny=120 outcomes per group at each value, solves POT exact empirical transport and averages. There are 100 paired bootstrap resamples within each group. There is no separate model training; `train.sh` saves the configuration and `predict.sh` fits each instance independently.

Compared with the source, only observation loading, configuration saving, CSV output, seeds, paths and independent-instance scheduling differ. Data-generation seeds are separate from algorithm seeds. Algorithm randomness is derived from `BENCH_SEED` and the instance's observation bytes, independent of its ID and position; the original internal random draw order remains unchanged. The source's method, estimator settings and bootstrap loop are retained. Notebook outputs and author-generated plots are removed.

Measured on **8 CPUs**: configuration writing averages **1.65 s**; prediction on validation averages **5.28 min** (range 5.26 to 5.30 min), and on the 200 hidden instances **10.5 min**. Each worker uses one numerical-library thread. Memory budget is **16 GB**, not a measured peak.

## 5. Interface (frozen)

- `bash code/train.sh --out DIR` writes `config.json` into the model directory.
- `bash code/predict.sh --model DIR --input PATH --output FILE` writes CSV with header `id,point,lower,upper`; see `bench/io_spec.md`.
- `BENCH_SEED` is the algorithm seed; `BENCH_CPUS` is the worker budget. `BENCH_DATA_DIR`, `BENCH_BENCH_DIR`, `BENCH_CODE_DIR` locate public data, frozen helpers and code. Preserve these interfaces and accept absolute paths.
- Dependencies are rebuilt from exact `name==version` PyPI pins in `code/env.lock`. Editable, git and local-path packages are unsupported.

## 6. Rules

- Editable surface: observations-only density/COT estimators, valid confidence-interval constructions, numerical surrogates and computation under `code/`, within the fixed estimand and coverage problem below.
- Frozen surface: `data/`, `bench/`, this README and the interface specification.
- The judge uses 8 CPUs and a 16 GB memory budget.
- Three seeds are used. Honor `BENCH_SEED` consistently in train and predict. Data may not be pooled across instances. Public labels may guide configuration selection but must not supply point estimates or interval centers.
- No network is available during train or predict. Do not read or guess hidden data, embed truths from the paper or the labels, simulate extra data, or hard-code predictions. Patch audit checks for such violations even if the coverage score is finite.

## 7. Known pitfalls

- Almost all computation occurs in predict. A full validation call includes 100 bootstrap estimates for every instance and takes about 5 minutes on 8 CPUs. Budget development calls accordingly; use subsets of public inputs for initial debugging.
- The fixed coverage threshold makes the objective discontinuous. Short intervals with inadequate coverage score Infinity. Validation has 100 instances, so its coverage moves in steps of 0.01.
- "Basic" describes the baseline's normal interval. Alternative valid interval constructions are allowed for the same COT target and fixed coverage requirement.
- The baseline uses one BLAS/OpenMP thread per worker. Alternative allocations must keep all workers and numerical threads within BENCH_CPUS.
- Prediction paths may have no extension. Write exactly the requested file. Scripts run with `PYTHONSAFEPATH=1`; preserve the explicit code import path.
- A zero-width oracle could satisfy numerical coverage if supplied the truth; it remains disallowed by the observations-only and no-truth-table rules.

## Research scope

**Objective.** Construct informative 95% confidence intervals for each instance's COT value from its pair of observed groups, minimizing mean interval length subject to coverage.

**Fixed research objects.** The instance family with (dY,dZ,n)=(2,1,3000) and one distribution per instance; clipped observed data and the independent per-instance boundary; the target, each instance's COT value before clipping; endpoint-inclusive coverage above the floors in section 3 and mean interval length. Interval lengths are comparable only for the same target, data and coverage requirement, so these stay fixed.

**Editable methods.** Density estimation, conditional optimal-transport solvers, valid confidence-interval construction and numerical optimization; bootstrap resampling or integration from distributions fitted to the current observations; a fixed configuration selected on validation.

**Forbidden substitutions.** The estimator must stay nonparametric: it may assume smooth densities, but it may not assume a parametric form for the conditional laws, such as a Gaussian or location-scale model, and it may not evaluate the closed-form value of such a model at fitted parameters. Intervals must target nominal 95% coverage: do not lower the nominal level, scale widths by a factor tuned against validation coverage, or return degenerate intervals on a subset of instances. Changing the estimand to the clipped-data population, replacing the distribution or relaxing coverage; extra population simulations, lookup of the truth, fitting to truth labels or sharing observations across instances; training or fitting reusable model state or learned preprocessing on validation features or labels, including final train+val refits or validation-based distillation. The scorer enforces coverage. Patch audit covers the rest.

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
