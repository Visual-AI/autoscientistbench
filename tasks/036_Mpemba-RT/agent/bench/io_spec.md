# Fig. 2 permutation optimization interface

Input is an NPZ file opened with `allow_pickle=False`. Arrays: `id` is a unique
opaque string per instance; `n` is its state-space dimension; `p`, `pi` have shape
(B,W), and `modes` has shape (B,W,2). Only the first `n[i]` entries in each row
are active; all padding is zero. `p` is the initial Gibbs probability, `pi` is
the bath equilibrium probability, and columns of `modes` are the authors'
weighted-normalized right modes r2 and r3. These arrays completely specify the
optimization instance. IDs carry no physical information. No generation seed or
parameter metadata is supplied to prediction. Parameters must not be recovered
from private files or used as a key to a precomputed solution table.

Output is text CSV, even when the supplied output path has no extension. Header
exactly: `id,replica,permutation,elapsed_s,iterations`. Provide exactly ten rows
per input ID, replica indices 0 through 9. Row order is arbitrary. `permutation`
is a CSV-quoted JSON list of n integer source indices, each of 0 through n-1
exactly once. The resulting probability state is `p[permutation]`. This preserves
degenerate probabilities exactly. `elapsed_s` is a finite, nonnegative wall-clock
measurement for that independent optimization run, including preparation and
serialization of its permutation. `iterations` is a nonnegative integer work
counter for diagnostics; 50,000 is the baseline, not a constraint on improvements.
Do not select the best of several runs to replace independent replica results.

The public labels NPZ contains input arrays and a JSON `metadata` string holding
public generation seeds and physical parameters. Scoring uses only frozen arrays,
never any objective claimed by the submission. For each instance it computes
`median(log10(sum(abs((modes/pi[:,None]).T @ p[permutation]))))` over ten replicas;
the scalar metric `median_log10_c` averages these instance medians. It is lower
is better. Per-run reported times are diagnostics subject to honesty audit.

Malformed output or a failed permutation check yields `Infinity` under
`median_log10_c`; `score_rule_value=-0.897` preserves the specified finite worst
value as a diagnostic. An exactly zero float64 cost uses the smallest positive
float64 value before log10 (about -323.306); otherwise the formula is unmodified.

## Research scope

The research objective and objects below are fixed; methods may change within this task scope.

**Objective:** Minimize fixed two-slow-mode overlap by permuting each supplied Gibbs vector, preserving independent-replica median-log aggregation.

**Fixed research objects and controls:**

- Supplied p, equilibrium pi and two normalized slow modes of the same Ising instance; the two-mode overlap cost.
- Feasible outputs are permutations of the original probability multiset, including degeneracies.
- Ten independent runs per instance, median of log10 costs followed by instance mean, original outer-seed and isolation protocol.

**Editable components:**

- Permutation search, proposals, optimizer, annealing schedule, iteration count, numerical surrogate and implementation, subject to the annealer limit under Forbidden substitutions.
- Validation-selected configuration or a reusable policy fitted only on training data, solving each instance independently; simulated annealing is a baseline.

**Forbidden substitutions:**

- Change probability vectors, Hamiltonian-derived modes, overlap target or permutation feasible set.
- Return arbitrary distributions, alter median/log ordering, or share/select best solutions across required independent replicas.
- Use private metadata, precomputed permutations or another evaluation instance.
- When retaining the supplied simulated annealer, changing only iterations=50000, the four-entry proposal size or temperature=max(1e-5,0.02*(1-step/iterations)) does not count as a method improvement.

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
