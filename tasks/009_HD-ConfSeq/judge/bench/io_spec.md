# Confidence-sequence tensor interface

Inputs are NPZ archives (load with `allow_pickle=False`). `id` is a string vector of opaque stream IDs.
Each ID is also a key containing a float64 observation matrix. A/B observations have shape (500,10);
coverage observations have shape (1000,5). The number and order of streams are not part of the contract.
Inputs contain no seeds, scenarios, primary/replication roles, or true means. Each stream is independent.

Output is an NPZ archive written through a binary file handle to the exact requested path, without adding a suffix.
It contains the same unique `id` set. All other keys are `ID__field`; no additional keys are allowed.
All numeric arrays must be finite. Each field below is mandatory for its certificate type.

## A/B certificate

`geometry_type` is an optional string scalar: `box_lp` (the default) or `predicate`.
The common fields are `stop`, `decision`, `lower`, `upper`, `volume`, `volume_seed`, `above`, and `below`.
For `predicate`, omit `center`, `radii`, `box_lower`, `box_upper`, and `power`; instead provide
a Unicode `predicate` array of shape (T,), described below. The original `box_lp` fields are:

- `stop`: integer scalar T in 1..500; `decision`: integer scalar 0 (no decision), 1 (significant), or 2 (unsubstantial).
- `lower`, `upper`, `center`, `radii`, `box_lower`, `box_upper`: shape (T,10). Radii must be positive.
- `power`: shape (T,), integers in 1..64. The geometric set is the intersection of the box,
  the unit cube, and `sum(abs((mu-center)/radii)**power) <= 1`.
- `volume`: shape (T,), the volume at every reported time. At the final time the scorer recomputes it
  with the released volume evaluator (5 repetitions of 25000 samples when no analytic containment applies).
- `volume_seed`: integer scalar in 0..2**32-1; repetition j uses this seed+j. Seed selection must depend only
  on `BENCH_SEED` and stream ID, never on results, parameters, or seed search.
- `above`, `below`: shape (10,), Boolean flags at stopping. The scorer recomputes flags and the first stopping time.
- Bounds must equal the coordinate intersection of the submitted box, ellipsoid projection, and [0,1].
  The released fallback to [0,1] for crossed coordinate projections is preserved.

Row index t (zero based) uses only observations with indices strictly less than t and is reported as time t+1.
Thus stop=1 uses no observations; stop=178 uses the first 177 observations. Report every row through stopping.
Significance has precedence over the volume rule at the same time. Volume radius means volume**(1/10),
not a half-width. Stop when any lower>0.5 or upper<0.5; otherwise when radius<=0.02; otherwise continue to 500.

## Coverage certificate

Three representations are supported, selected by string scalar `coverage_type`:

- `hedged`: `lam` shape (999,5), nonnegative; `c`, `theta`, `weights` shape (5,), with c/theta strictly between
  zero and one, positive weights summing to one. For each candidate mean m and each observed x, multiply
  `1 + min(lam,c/(m+1e-12))*(x-m)` for the positive capital and
  `1 - min(lam,c/(1-m+1e-12))*(x-m)` for the negative capital. Each axis capital is
  max(theta*Kplus,(1-theta)*Kminus). Membership is sum(weights*capital)<20.
- `box_lp`: `lower`, `upper`, `center`, `radii` shape (20,5), and `power` shape (20,), between 1 and 64.
  Membership is inside both the box and the corresponding positive-radius Lp ellipsoid.

- `predicate`: `lower`, `upper` shape (20,5) and Unicode `predicate` shape (20,). Each row defines
  its confidence set by the numeric expression format below.

The scorer evaluates membership at the true mean, separately at times 50,100,...,1000, preserving the
released t-1 observation convention (49,99,...,999 observations). A stream is covered only if all 20 events hold.
The baseline exports `hedged`, exactly matching its inherited `is_member_at`, even though its A/B geometric
approximation uses `box_lp` parameters. Alternative methods must represent their own coverage predicate
faithfully; a method may not use a different validity surrogate just to pass the coverage checks.

## General set predicates

A predicate certificate defines the set `{m: lower <= m <= upper and expression(m)}` within
[0,1]^D. Its bounds are part of the set definition, so they are sound enclosing coordinate bounds;
they need not be the tightest projections. They drive the same significance rule as above.
Each row must represent the actual confidence set obtained from the permitted observation prefix.
This format permits polyhedral, nonconvex and neural membership regions without requiring an Lp family.

Each Unicode string is JSON with exactly `nodes` (a nonempty list) and `output` (a node index).
A node has `op` and either the fields below or `args`, a list of earlier node indices:

- `input`: no other fields; a D-vector of proposed mean coordinates.
- `const`: `value`, a finite numeric scalar/array (Boolean constants are allowed).
- Unary: `neg`, `abs`, `exp`, `log`, `sqrt`, `tanh`, `sin`, `cos`, `not`.
- Binary: `add`, `sub`, `mul`, `div`, `pow`, `minimum`, `maximum`, `lt`, `le`, `gt`, `ge`, `eq`, `and`, `or`.
- `matmul`: two arguments; right argument must be a constant vector or matrix (one or two dimensions).
- `sum`, `mean`, `min`, `max`, `all`, `any`: one argument plus integer `axis` within that point's feature shape.
- `take`: one argument plus integer `axis` and integer scalar/array `indices` within the feature shape.
- `where`: three arguments: condition, true value, false value.

Normal NumPy feature broadcasting applies. Expressions operate independently on each mean; no operation
may mix evaluator sample points. The output must be one Boolean for each point. Nonfinite computations,
unknown operators/fields, invalid shapes, duplicate keys and forward references are invalid. The interpreter
has no Python execution, file access, randomness, or access to labels. There is no fixed network architecture,
parameter count, graph depth, or Lp power requirement for this representation.

For example, this describes a disconnected region according to the first coordinate:

```json
{"nodes":[{"op":"input"},{"op":"take","args":[0],"axis":0,"indices":0},{"op":"const","value":0.25},{"op":"const","value":0.75},{"op":"lt","args":[1,2]},{"op":"gt","args":[1,3]},{"op":"or","args":[4,5]}],"output":6}
```

Terminal volume uses the released bounding-box Monte Carlo estimator: five independent repetitions of
25,000 uniform points, `default_rng(volume_seed + j)`, multiplied by the bounding-box volume.
`bench/predicate.py:volume` exposes that deterministic computation. Sampling is batched for memory use;
the random sequence, sample count, average and stopping criterion are unchanged. Legacy Lp certificates
retain their original analytic-containment and smaller-proposal shortcuts. Choose enclosing bounds that
resolve a small set adequately: at a volume-decided stop at least 100 of the 125,000 Monte Carlo points must fall inside the set, otherwise the prediction is invalid; Monte Carlo precision remains part of this task's original volume protocol.
The scorer interprets the frozen expression itself; it does not expose its private true mean or probes to
candidate code. Coverage and A/B certificates must be faithful to the same method, with the same observation
prefix and stream-independence requirements. Expression validity does not establish those source-level facts.

## Labels and checks

`truth.npz` contains observations plus private scenario/role, true mean, true-effect mask, generator seed,
and baseline per-instance diagnostics. Validation labels are public. Baseline diagnostics do not set thresholds:
the coverage floor 0.919 and false-flag cap 3/20 are absolute.

`python bench/validate.py PRED INPUT` rejects malformed archives, missing/duplicate IDs, extra fields,
nonfinite numbers, bad shapes, invalid bounds, radii, powers, flags, and bet parameters.
`python bench/scorer.py PRED TRUTH` prints one JSON line with `mean_stop_t` and diagnostics.
Statistical gate failure gives 500; certificate/protocol inconsistency gives Infinity.

## Research scope

The certificate represents a confidence set for the fixed original bounded mean vector. Replacing a CS construction or geometry is allowed; changing the estimand, observed stream, scientific coordinates, stopping/coverage objective or volume protocol is not. Internal representations may change only if the reported set is faithfully mapped back to the original mean coordinates. The original paper's Table 3 compares different CS families, so an Lp shape is a baseline choice rather than a fixed research object. The empirical coverage checks and expression parser do not independently prove anytime validity or lawful use of observation prefixes; those rest on patch audit.

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
