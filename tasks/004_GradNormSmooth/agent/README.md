# Gradient-Normalized Smoothness for Optimization with Approximate Hessians

## 1. Goal

**Research question:** how few accepted updates does a method need to reach high accuracy on a fixed LogSumExp problem, when gradient calls and linear-system solves are capped?

Minimize the centered LogSumExp objective of the source paper's Figure 3(a), with a 1000 by 500 random matrix, mu=1 and an all-ones initial point:

`f(x) = logsumexp(C x - b)`, where `C = A.T - g0` (g0 subtracted from every row) and `g0 = A @ softmax(-b)`.

Improve `iterations` (lower is better) on fresh instances held by the judge from the same distribution as the public validation set. The first method slot is scored against that fixed mathematical objective; the other three slots report the paper's comparison curves. Methods and hyperparameters may be changed within §6.

**The experiment.** The paper studies a gradient-regularized Newton method (Algorithm 1), with gradient `g` and `||g||_*^2 = g' B^-1 g`:

`x_next = x - (H(x) + lambda B)^-1 g(x)`, where `lambda = ||g(x)||_* / gamma`.

H is the Hessian or an approximation. The theory ties the step size gamma to the Gradient-Normalized Smoothness, the radius around x in which the gradient field stays well approximated; in practice an adaptive search chooses gamma. Figure 3(a) plots f(x_k)-fstar against iterations on one shared random problem, over a 500-update horizon. The Hessian and the search rule differ per method:

| Slot | Hessian | Search rule |
|---|---|---|
| `exact_func` (scored) | exact | Function search: accept when `f(x) - f(x_next) >= ||g_next||_*^2 / (8 lambda)`. Halve gamma on rejection, double it on acceptance. |
| `inexact_func` | weighted Gauss-Newton `C' Diag(softmax(Cx-b)) C / mu` | Same function search. |
| `exact_grad_alpha1` | exact | Earlier gradient search: accept when `g_next'(x - x_next) >= ||g_next||_*^2 / (4 lambda)`, with `lambda = M ||g||_*`, so gamma = 1/M. Divide M by 4 on acceptance, multiply by 4 on rejection. |
| `exact_grad_alpha0` | exact | Same gradient search with `lambda = M`, so gamma = `||g||_*`/M. |

The paper reports that the function search keeps gamma larger than the gradient search does, and that the inexact Hessian performs similarly to the exact one on this objective.

## 2. Data

- `data/train/input.npz`, `data/train/labels.npz`: one public development instance, seed 42. Baseline training needs no data.
- `data/val/input.npz`, `data/val/labels.npz`: two public instances, notebook seed 3124 and seed 2718.
- The judge holds independently generated instances whose seeds are unavailable during development.
- `bench/generator.py` freezes the original RandomState draw sequence and centering: A (500 by 1000) and b (1000) have independent entries uniform on [-1, 1]. Inputs contain ids and raw A and b arrays; labels additionally contain fstar. Public seed maps are in each public data directory.
- Do not alter the supplied instances or use outside data. Additional public experiments may use the generator, but predictions must solve the supplied problem independently through its oracle.

## 3. Metric and scoring

`bench/scorer.py` computes the first recorded iteration where f(x)-fstar is at most 1e-7; the initial point is iteration 0. If matrix inverses exceed 24 or gradient calls exceed 25 up to that iteration, or no crossing occurs within 500 updates, that instance scores 500. The metric `iterations` is the mean of these scores for `exact_func`. The other method slots have diagnostic threshold crossings without budgets, not contributions to the main metric, but their rows must still pass the checks below.

Only `grad` calls and successful `solve` calls are charged. `func`, `hess` and `approx_hess` calls are logged in `events` and have no budget. The paper's figure has no oracle budget; the limits of 25 and 24 equal what the baseline spends on seed 3124.

| Guard | Class | Enforcement |
|---|---|---|
| Threshold, horizon and oracle budgets | scorer | First crossing from recomputed objectives; budget failure is 500 |
| Objective, summary and event/counter consistency; every slot starts at ones | scorer | Recompute every recorded objective (tolerance 1e-10) and recount events; mismatch is Infinity |
| CSV schema, exactly the four slots per id, 1 to 501 finite records | validator | `bench/validate.py` rejects the file |
| Actual calls agree with reported events and counters | judge probe | Instrumented oracle replay |
| Only public oracle access, no known-solution shortcuts or fabricated transcripts | patch audit | Review submitted code; the scorer cannot authenticate an in-process transcript |

Baseline validation `iterations` = **16.5**, seed standard deviation **0.0**: 17 on seed 3124 (25 gradient calls, 24 inverses) and 16 on seed 2718 (23, 22). The comparison slots cross at validation means 52.5, 19.0 and 21.0 and exceed the budgets, which does not affect the metric. PASS requires at least 1% relative improvement against the baseline mean on the judge's instances, subject to the harness's variance threshold. The baseline is deterministic given inputs.

## 4. Baseline

Official code: https://github.com/epfml/grad-norm-smooth at `9f4ca00029f0b17fb9d48eb393d77bc8af153edc`.
`src/methods.py` implements Algorithm 1 with exact Hessian and function search; comparison slots use the inexact Hessian and the two gradient-search settings. The original 500-update horizon, stopping tolerance 1e-8, adaptive search, explicit metric inverse, defaults and numerical operations are retained. The defaults are gamma_0=1, gamma_min=1e-9, H_0=1 and H_min=1e-5. The baseline metric is B=A@A.T on the stored A, and its stopping test reads f* from `func` at the origin.

`bench_run.py` adds input loading, config save/load, the frozen oracle and trajectory export. The original plots, executed notebooks and unrelated data are not included. The reproduced public instance takes 17 updates; the paper's digitized curve near 359 updates used an unreleased figure-specific instance and is not matched.

Measured on 8 CPUs: train **0.97 s**, prediction of the two validation instances with all four slots **3.77 s**. The program retains four BLAS threads per solve.

## 5. Interface (frozen)

- `bash code/train.sh --out DIR` writes `config.json` and run metadata.
- `bash code/predict.sh --model DIR --input PATH --output FILE` solves each supplied instance and writes the CSV specified in `bench/io_spec.md`.
- Environment: `BENCH_SEED`, `BENCH_CPUS`, `BENCH_DATA_DIR`, `BENCH_BENCH_DIR`, `BENCH_CODE_DIR`.
- Load the frozen oracle by path from `BENCH_BENCH_DIR`; use `func`, `grad`, `hess`, `approx_hess`, `solve`, `record` and `history`. Each method/instance needs a fresh oracle. Record the initial and every accepted iterate.
- Dependencies are NumPy, SciPy and threadpoolctl. Package changes must be reproducible through `name==version` pins in `code/env.lock`; only public PyPI packages are used by the baseline.

## 6. Rules

Everything under `code/` is editable, including algorithms and hyperparameters, while the entry signatures and output format are fixed. `data/`, `bench/`, this README and oracle semantics are frozen.

Checked by patch audit:

- All objective, derivative and linear-system solver calls must go through the frozen API. The fixed metric B=A@A.T and its preprocessing inverse are exempt from the inverse budget, as in the source.
- Do not inspect oracle closures, replace frozen modules, reset counters, fake events, skip accepted iterates, or compute uncounted derivatives/solves elsewhere. Do not store or retrieve precomputed solutions.
- The objective is centered at zero: directly returning the known optimum, or exploiting this knowledge to bypass optimization, is explicitly disallowed.
- Hidden seeds must never be inferred or searched.

Train and predict run without network access, on 8 CPUs and a configured 16 GB budget, which is a limit, not a measured peak. The judge runs one seed for this deterministic baseline. Candidate randomness must honor `BENCH_SEED`.

## 7. Known pitfalls

The 1e-7 scoring threshold differs from the authors' 1e-8 stopping test. Report the first crossing, and sample its counters even if later iterations were executed. Rejected search candidates count toward both budgets. The public 17-update run exactly exhausts both budgets. A fresh instance may legitimately score 500. Never append an extension to the supplied output path. Preserve full floating precision and CSV quoting for nested trajectories.

## Research contract

**Objective:** Reach the specified accuracy for the fixed centered LogSumExp problem within its oracle-query budgets.

**Fixed research objects:**

- The centered LogSumExp(Ax-b) objective with mu=1, supplied 1000 by 500 A and b, and all-ones initialization remain fixed. Methods are only comparable on the same problem.
- The first gap at most 1e-7, horizon 500, at most 25 gradient calls and at most 24 charged inverses remain fixed. All accepted iterates and rejected-query costs must be recorded. Without this, fewer updates could be bought with more charged queries per update.

**Editable methods:** The optimizer, search strategy, Hessian approximation and preconditioner may change through the frozen oracle. Surrogates may guide the solver but cannot replace its evaluated objective.

**Forbidden substitutions:** Any point from which a new search direction is computed is an accepted iterate and must be recorded, so the inner steps of a nested, proximal or multi-step solver cannot be grouped under one recorded iteration. Do not use the origin, f(0) or the iterate x_k itself as a step, search direction or subspace basis; the stopping test is the only permitted use of f(0).

- Do not replace the mathematical objective/initialization, bypass counters, suppress accepted iterates, introduce uncounted derivative/inverse calls, or exploit knowledge of the synthetic optimum instead of the allowed oracle.
- Do not train or fit reusable model state or learned preprocessing on validation features or labels, including final train+val refits or validation-based distillation.

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
