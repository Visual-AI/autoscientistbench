# 043_Geodesic-DiD: Geodesic difference-in-differences

## 1. Goal and baseline
Improve estimation of the distribution-valued average treatment effect on treated units from control and treated groups observed before and after treatment. The effect is a geodesic from the untreated counterfactual post-treatment distribution to the observed treated post-treatment distribution, under univariate Wasserstein geometry.

The baseline is the authors' `gdd` method, upstream commit `6c7237af41226a2d68071c1f54d6e65339297fcb`. It averages observed quantile curves within each group and period. For control quantiles Q_c_pre and Q_c_post, treated quantiles Q_t_pre and Q_t_post, and the CDF F_c_pre corresponding to Q_c_pre, it computes Q_cf(p) = Q_c_post(F_c_pre(Q_t_pre(p))). The output starts at Q_cf and ends at Q_t_post.
You may replace the estimator in `code/` under the same data and compute budget.


## 2. Data
Each dataset has n=1000 units. The task is inference-only; no training split is supplied. `data/val/input/` contains 500 observed datasets. `data/val/labels/` is only for evaluation and method, hyperparameter or checkpoint selection. Do not fit reusable models or learned preprocessing on validation. Per-instance estimation from that instance's observations remains permitted.

Each outcome distribution is specified by its location, scale and support. These descriptors determine its 101-point quantile observation. See `bench/io_spec.md`.

## 3. Metric
Output the start and end quantile curves of the estimated map; the frozen scorer computes the comparison. For each replication, it interpolates the predicted end curve as a function of the predicted start curve and evaluates it at the fixed reference curve. It then computes root-mean-square discrepancy from the fixed target over the 101 grid points. `mean_geodesic_error` averages these distances; lower is better. The fixed reference and target are population curves: for each probability, the quantile averaged over the random unit location, by numerical integration.

Measured validation mean is 0.09536434825034541, with standard deviation 0.0 across reference repetitions.
The validator requires one row per input id in index order, the prescribed columns, finite and strictly increasing quantile curves, and endpoints -3 and 3 within absolute tolerance 1e-7. Hardcoded population answers are forbidden and require patch audit; the numerical validator does not enforce this rule.

## 4. Interfaces and resources
Run `bash code/train.sh --out MODEL_DIR`, then
`bash code/predict.sh --model MODEL_DIR --input INPUT_DIR --output PRED.csv`.
Train records the inference-only model contract; predict estimates the map on
every supplied dataset without reading labels. Use `bench eval-dev` in the
harness for development evaluation. `BENCH_CPUS` caps workers; the harness
provides `BENCH_SEED`. Task resources are 16 CPUs, 80 GB, default research budget
5 hours.

## 5. Runtime
The supplied read-only R 4.5.3 environment includes truncnorm, purrr and pracma. Python orchestration uses only the standard library, so `env.lock` contains comments only. Do not rebuild the environment.

Measured model-recording times were 0.11, 0.12 and 0.12 seconds. Validation prediction times were 24.38, 23.83 and 25.93 seconds.

## 6. Rules
Change only `code/`. Do not alter frozen `bench/`, labels or evaluation data;
do not read hidden-test data, use old results, hardcode synthetic population
answers or fabricate predictions. Execution has no network. Estimator changes
are allowed while the input and scoring contract remain fixed.

## 7. Submission
Submit code through the benchmark patch mechanism. Submissions are evaluated from a clean checkout for training, prediction, validation and scoring.

## Research contract

**Objective:** Estimate the fixed univariate Wasserstein geodesic average treatment effect on treated distributions (GATT), minimizing the original geodesic error.

**Fixed research objects:** The distribution-valued DiD/GATT estimand, univariate Wasserstein geometry, fixed counterfactual/transport definition and quantile coordinate/support convention remain fixed. All n=1000 observed units, 101-point outcome representation, 500 replications and frozen geodesic-distance metric remain fixed.

**Editable methods:** Distribution regression, nuisance modeling, fitting losses, transport estimation and numerical solvers may change while estimating the same geodesic GATT. The baseline gdd implementation is replaceable; internal representations must map back to the same quantile outputs.

**Forbidden substitutions:** Do not replace distribution-valued GATT with a scalar mean effect or alter Wasserstein/reference transport geometry, quantile support or evaluated units. No hidden potential outcomes, population transport constants, extra data or metric changes.

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
