# Input and output contract

`--input` is an NPZ archive with `id` (N strings), `A` (N,500,1000 float64), and `b` (N,1000 float64).
Arrays describe the supplied numerical problems, with smoothing mu=1, the notebook's centering and initial point ones.
No generation seed is included in this input. The frozen generator documents the distribution.
Labels contain those arrays and `fstar` (N float64); development labels are public.

Output is CSV, written to the exact supplied filename, with header:
`id,method,history,iterations,final_objective,grad_calls,matrix_inverses`.
There is exactly one row for each input id and each method slot: `exact_func`, `inexact_func`,
`exact_grad_alpha1`, `exact_grad_alpha0`. The first slot is scored; the others report the comparison curves.
Row order is arbitrary; duplicates, missing slots, nonfinite numbers and malformed fields are rejected.

`history` is a quoted JSON array, with 1–501 records in iteration order starting at iteration zero.
Each record has exactly `x` (500 finite coordinates), `func` (finite objective), `grad_calls`,
`matrix_inverses` (nonnegative integers), `events` (ordered oracle operations since the preceding record).
Allowed events are `func`, `grad`, `hess`, `approx_hess`, `inverse`.
Use the frozen oracle's `record(x)` at every accepted iterate, including the initial point, and `history()` to export.
`iterations` is the first recorded index with objective gap at most 1e-7, or 500 if none.
`final_objective` is the last recorded objective; summary counters are sampled at the first crossing,
or at the final record if no crossing exists. All summary values must be finite.

Load `oracle.py` by absolute path from `BENCH_BENCH_DIR` with importlib. `make_oracle(A,b,id,method)` returns
`func(x)`, `grad(x)`, `hess(x)`, `approx_hess(x)`, `solve(matrix,rhs)`, `record(x)`, `history()`.
All objective, derivative and linear-system solver calls must use this interface. `solve` performs the
original Cholesky factorization and solve; each successful call counts once. Forming the fixed Gram matrix
and its inverse is preprocessing and excluded, exactly as in the original history fields. Never cache or reuse
an oracle across instances or method slots. Inspecting its closure or private objective is forbidden.

Scoring recomputes the objective at every iterate from the labels and checks event/counter consistency.
For each instance the main score is the first crossing index if its inverse count is at most 24 and gradient
count at most 25, otherwise 500. The metric `iterations` is their arithmetic mean, lower is better.
Malformed or falsified objectives/counters produce `Infinity`; a legitimate budget overrun produces 500.
Transcript authenticity is additionally checked with an instrumented replay probe and patch audit.

## Research scope

Reach the specified accuracy for the fixed centered LogSumExp problem within its oracle-query budgets.

Fixed research objects and target: The centered LogSumExp(Ax-b) objective with mu=1, supplied 1000 by 500 A and b, and all-ones initialization remain fixed. The first gap at most 1e-7, horizon 500, at most 25 gradient calls and at most 24 charged inverses remain fixed. All accepted iterates and rejected-query costs must be recorded.

Editable methods: The optimizer, search strategy, Hessian approximation and preconditioner may change through the frozen oracle. Surrogates may guide the solver but cannot replace its evaluated objective.

Forbidden substitutions: Do not replace the mathematical objective/initialization, bypass counters, suppress accepted iterates, introduce uncounted derivative/inverse calls, or exploit knowledge of the synthetic optimum instead of the allowed oracle. Any point from which a new search direction is computed is an accepted iterate and must be recorded, so the inner steps of a nested, proximal or multi-step solver cannot be grouped under one recorded iteration. Do not use the origin, f(0) or the iterate x_k itself as a step, search direction or subspace basis; the stopping test is the only permitted use of f(0).

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
