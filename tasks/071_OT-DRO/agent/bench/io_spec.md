# I/O specification

One *instance* = one DRO linear-regression problem at J = 10: ten observed points and
twenty bootstrap resamples of them. The data-generating process is `bench/generator.py` (frozen).

## Input (`--input PATH`, NumPy `.npz`)
| array | shape | dtype | meaning |
|---|---|---|---|
| `id` | (n,) | str | opaque instance id (e.g. `val_007`); the order carries no information |
| `x` | (n, 10) | float64 | the J = 10 regressors x_j ~ U(-10, 10) |
| `y` | (n, 10) | float64 | the responses y_j = w x_j + e_j |
| `bootstrap_indices` | (n, 20, 10) | int64 | the n_b = 20 bootstrap resamples: row k lists the indices (into the 10 points) of resample k |

Nothing else is given: the true (w, sigma^2) of an instance and which of the ten generator models produced it are
labels. The bootstrap resamples are part of the frozen protocol: guard (B) is evaluated on exactly these resamples.

## Output (`--output FILE`, CSV)
Header exactly `id,w_star,L11,L21,L22,eps`; exactly one row per input `id` (same id set, any order); all values
finite; `eps > 0`; `L11 > 0`, `L22 > 0`.
- `w_star` — the submitted decision (scalar regression weight) for the instance;
- `L11,L21,L22` — the lower-triangular transport-cost factor `L = [[L11, 0], [L21, L22]]` of the ambiguity set
  (ground cost `||L^T (xi_1 - xi_2)||_2`; any full-rank cost matrix has such a factor);
- `eps` — the ambiguity radius.
The submission is the triple (w*, L, eps): the scorer re-solves the DRO problem of the instance at the submitted
(L, eps) and requires w* to be its optimum (guard A), and evaluates the bootstrap-exceedance and coverage guards
at the submitted (L, eps).

## Labels (`truth.npz`, public for train/val, held by the judge for test)
`id, model_id, w_true, sigma2, seed, coverage_seed, x, y, bootstrap_indices`. Validation and evaluation labels additionally contain the unchanged baseline's
per-instance `baseline_exceedance`, `baseline_covered`, `baseline_w_star`, `baseline_e_oos` (used by the
scorer to set the baseline-relative guard thresholds and for diagnostics).

## Scoring
`python bench/scorer.py <pred.csv> <truth.npz>` prints one JSON line; the task metric is the key `e_oos`
(mean over instances, lower is better; `Infinity` when a guard is violated). The other keys are diagnostics
(`e_oos_raw`, `bootstrap_exceedance` and its cap, `coverage_rate` and its floor, `decision_violations`, ...).
`python bench/validate.py <pred.csv> <input.npz>` is the format check (exit 0 = ok).

The task-specific Research contract in README is binding: output validity alone does not permit substituting a different research object or objective.

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
