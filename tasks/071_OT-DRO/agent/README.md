# Loss-aware OT-DRO linear regression

## 1. Goal
For each observed regression instance, choose a lower-triangular transport factor `L`, a positive radius `eps`, and its DRO-optimal decision `w_star`. Minimize `e_oos`, the mean true out-of-sample absolute error.

The experiment asks whether learning transport geometry reduces conservative DRO regression errors. It compares identity and learned geometry; this task compares your method with the supplied learning method. Each instance contains J=10 observations of `y=w_m*x+e`, with `x~Uniform(-10,10)` and Gaussian noise, and 20 fixed bootstrap resamples. The frozen generator defines the ten model distributions. The ambiguity set uses type-1 Wasserstein distance with cost `||L^T(xi_1-xi_2)||`. You may improve the method under the constraints below.

## 2. Data
- `data/train/input.npz` and `data/train/truth.npz`: 50 public instances in the training split; the baseline does not use them. Labels may be used for offline evaluation and hyperparameter selection, not to supply decisions or fit a truth-table predictor.
- `data/val/input.npz` and `data/val/truth.npz`: 20 public validation instances. Public observation seeds are `1000*m+d` for model indices m=0..9 and d in {0,3}; labels contain the full metadata.
- Hidden evaluation instances are unavailable in this workspace.
- `bench/generator.py` is the frozen data-generating process, matching the source draw order exactly. Inputs contain only opaque IDs, observations and bootstrap indices, with shuffled instance order. True parameters and auxiliary coverage seeds are labels.
- Each decision must depend only on that instance's observed data and a fixed development-selected configuration. Do not generate additional instances, use the model table or identify model membership from IDs. Do not re-split or relabel data. See `bench/io_spec.md` for shapes and formats.

## 3. Metric and scoring
`e_oos` (lower is better) is `mean E[|(w_m-w_star)*x+e|]`. The scorer uses the closed-form Gaussian absolute moment and 2001-node Gauss-Legendre quadrature over x, matching the source evaluation. Quadrature has no simulation noise; variation across observed datasets remains. The mean irreducible noise floor is approximately 22.451.

| Guard | Requirement | Enforcement |
|---|---|---|
| DRO optimality | Every decision minimizes `mean(abs(y-w*x))+eps*norm(solve(L,[-w,1]))`; decision error ≤ `1e-3*max(1,abs(w_opt))` or relative objective error ≤ `1e-8` | scorer |
| Bootstrap exceedance | Mean fraction of the 20 given resamples outside the submitted ball ≤ `max(0.10,baseline)+0.05`; validation cap 0.4225 | scorer |
| True-distribution coverage | Coverage against the frozen 5000-point proxy of the true distribution ≥ `baseline-0.05`; validation floor 0.1500 | scorer |
| Observations only | No truth-table lookup, model-index inference, remembered outputs, extra simulation, or use of another instance's data at prediction time | patch audit |
| Fixed instance definition | J=10 and the 20 supplied resamples define the instance; cannot replace or augment them | patch audit; scorer uses the supplied observations and resamples |

Guard baselines are the unchanged method's values on the same instances, frozen in the labels. There is no per-instance exceedance cap. Any numerical guard violation gives `Infinity`. Format errors fail validation. The observations-only rule requires patch review.

Measured validation baseline: **29.7475439266**, seed standard deviation **0.0**; exceedance 0.3725, coverage 0.2000. The judge uses one seed for this deterministic baseline and compares candidates against its archived reference. PASS requires relative improvement of at least **5%** and all guards passing. `bench status` exposes the public baseline and threshold.

## 4. Baseline
Official code: https://github.com/JonasOhn/trainable-ot-dro at upstream commit `6e8d18f023f4f2e920af22dc1ef32b292dbb9631`. `code/src/trainable_ot_dro/bilevel_optimization.py` implements the optimizer, `reformulations.py` the DRO problem, and `conic_problem.py` the Clarabel solve.

`train.sh` writes `config.json`; `predict.sh` learns L independently on each instance. It computes eps at L0=I as the 0.9 bootstrap-distance quantile, then calls the baseline optimizer: Adam learning rate 1e-4, lambda=10, eta=100, relative-improvement stopping tolerance 1e-6, at most 1e6 iterations, storing every 100 iterations. The last stored decision and L are submitted. The released implementation clips the normalized Adam direction to ±1e4 before the learning rate is applied.

Measured at **8 CPUs**: configuration writing 0.52 s; prediction on validation 322.77 s (5.38 min). The pool uses eight workers with one numerical-library thread each. Memory budget is 4 GB; this is a configured limit, not a measured peak. The method has a heavy runtime tail; runtime cannot be estimated from the short configuration-writing stage.

## 5. Interface (frozen)
- `bash code/train.sh --out DIR` writes all prediction configuration into DIR.
- `bash code/predict.sh --model DIR --input PATH --output FILE` writes CSV with header `id,w_star,L11,L21,L22,eps`, one finite row per input ID, in any order.
- `BENCH_SEED` must be honored; `BENCH_CPUS` supplies the CPU budget. `BENCH_DATA_DIR`, `BENCH_BENCH_DIR` and `BENCH_CODE_DIR` give the data, frozen helpers and code locations.
- `bench status`, `bench eval-dev --seed 1`, `bench validate-output FILE`, and `bench submit --final` are the development interface.
- Dependencies are rebuilt from `code/env.lock` with exact `name==version` PyPI pins. Editable, git and local-path dependencies are unsupported. Use `uv pip install` in the development venv; submission freezes its dependencies.

## 6. Rules
- Free surface: everything under `code/`, including ambiguity-set learning, auxiliary optimization losses, numerical solvers and the hyperparameters of a changed method, while preserving the scalar linear decision, absolute-loss type-1 Wasserstein DRO objective, scoring guards and observations-only rule. Keep both script signatures and the output format.
- Frozen surface: `data/`, `bench/`, this README and the interface specification.
- The allocation is 8 CPUs and 4 GB memory. No network is available during train or predict.
- One judge seed is configured; `BENCH_SEED` must still be accepted. Do not read or guess hidden data, hard-code predictions, infer generator seeds, use the generator's model table or model identities, fit from truth labels, or simulate extra data. Public labels are for offline diagnostics and configuration selection only.

## 7. Known pitfalls
- The baseline performs a complete bilevel optimization for each instance. Budget validation calls using the measured 5.38 min above; first evaluate changes on a small subset of the public training instances.
- The last few slow instances can dominate wall time. Use `BENCH_CPUS` independent workers and one BLAS thread each.
- `L11` and `L22` must be positive and `eps>0`. Arbitrary decisions that do not solve the submitted DRO problem fail the optimality guard.
- Collapsing the ambiguity set may lower raw error but violates exceedance and coverage. Inspect the scorer diagnostics when the metric is `Infinity`.
- The output filename may have no extension. Write exactly the requested path. Scripts receive absolute paths and run with `PYTHONSAFEPATH=1`; preserve the package path setup.
- Labels for development training instances have no frozen baseline fields; baseline-relative guard comparisons are defined on the validation and evaluation collections. The scorer's documented absolute fallback is only a diagnostic for labels without those fields.

## Research contract

**Objective:** Learn an admissible type-1 Wasserstein ambiguity geometry and radius for the fixed scalar absolute-loss DRO regression problem, minimizing out-of-sample absolute error.

**Fixed research objects:** The decision model y_hat=w*x without intercept, absolute residual loss and DRO objective mean|y-w*x|+eps*||L^-1[-w,1]|| are fixed. Mahalanobis ground cost ||L^T(xi1-xi2)||, positive lower-triangular L/radius, ten observations/twenty supplied resamples, optimality/exceedance/coverage guards and exact expected-absolute-error metric remain fixed.

**Editable methods:** Methods for learning L/eps, bilevel optimization, auxiliary optimization losses, numerical differentiation and conic/other solvers may change, with tuning of the changed method. Any submitted w must still solve the same fixed DRO objective for its submitted ambiguity set; Adam/initial radius learning and stopping choices are baseline methods.

**Forbidden substitutions:** Do not replace linear regression by a nonlinear model, absolute loss by squared error, type-1 cost by another ambiguity class, or submit an arbitrary low-error decision that is not DRO-optimal. No model-table truth/model-ID lookup, extra observations/resamples, hidden true distribution access or altered metric/guards. When retaining the supplied bilevel optimization pipeline, changing only gamma=0.1, learning_rate=1e-4, lambda=10, eta=100, rel_impr_obj=1e-6 or n_iter_max=1e6 does not count as a method improvement.

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
