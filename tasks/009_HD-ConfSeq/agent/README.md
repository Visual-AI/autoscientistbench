# Higher-dimensional confidence sequences for A/B stopping

## Goal

A confidence sequence (CS) is a sequence of confidence sets that contains a mean at all times simultaneously with
probability at least 1-alpha, so a decision stays valid whenever the experimenter stops. The source paper asks how
tight a tractable CS for a bounded vector mean can be. It lifts one-dimensional betting CSs to vectors through a
portfolio of coordinate wealth processes, which is tight but has no explicit boundary, and derives outer
approximations that keep validity: a bounding box, an lp-ellipsoid and their intersection.

This task is the paper's A/B case study (Table 3). CS constructions run on the same simulated web-metric streams with
the same decision rule and are compared by when they may stop. Stop with effect / no effect, over 10,000 batches in
the paper: Conf Sphere 10,000 / 10,000 (no decision), Banach Sphere 5,018 / 8,052, Sub-psi 5,198 / 6,650,
C_bonf 178 / 399, C_bbox 178 / 398, C_lp-safe 178 / 356, C_bbox ∩ lp-safe 178 / 356.

Improve `mean_stop_t` (lower is better) on these bounded observations while satisfying the coverage and false-flag
constraints below. Evaluation uses independently generated streams with private seeds, held by the judge, with the
same distributions and preprocessing as the validation data.

## Data

There is no training split; `data/train/README.md` only says so. Validation is for evaluation and for method,
hyperparameter or configuration selection; no model may be fitted on it. Computation on a stream's own allowed prefix
is part of the method.

- `data/val/input.npz` and `data/val/truth.npz`: 42 A/B streams and 400 coverage streams.
- A/B streams, 500 rows by 10 coordinates: each row compares one batch of simulated web-shop sessions from variants A
  and B, about 400 per variant (+/-15%). A row is `0.5*((mean_B-mean_A)+1)` in [0,1]^10, so no difference is 0.5.
  Coordinates: time, clicks, pages, scroll, add_to_cart, purchase, non_bounce, return_7d, noise_1, noise_2. With an
  effect, variant B's latent engagement is shifted by 0.06 on some metrics. Each scenario has one primary stream
  (the paper's seeds 1 and 2) and 20 replicates (seeds 101..120 and 201..220).
- Coverage streams, 1000 rows by 5 coordinates: 200 each of `bernoulli_mix` and `circle`, from the
  two original seed-42 sequences.
- `bench/generator.py` freezes the original simulator. Inputs contain only opaque IDs and observed matrices;
  scenario identity, primary-stream role, generator seeds, and true parameters are labels.

## Metric and scoring

The score is the mean stopping time of the primary effect and no-effect streams, with alpha=0.05,
coordinate significance threshold 0.5, geometric volume radius tolerance 0.02, and horizon 500. A stream stops as
significant once a coordinate bound excludes 0.5, and as unsubstantial once its volume radius reaches the tolerance.
At the same time, significance takes precedence over the volume rule. The scorer derives the first stop from
all submitted rows and recomputes certificate membership and terminal volume. Replicates and coverage streams only
feed the guards.

| Guard | Enforcement |
|---|---|
| Primary effect decision is significant; all flags belong to its true-effect set | scorer |
| Primary no-effect decision is unsubstantial | scorer |
| Coverage at all 20 saved checkpoints jointly is at least 0.919 for each distribution | scorer |
| At most 3 of 20 replicate streams have any false flag, separately per scenario | scorer |
| Reported decisions, flags, bounds and terminal volume agree with submitted certificates | scorer |
| Format, shapes, finite numbers and legal parameters | validator |
| Each stream is independent; outputs use only its allowed observation prefix | judge probe + patch audit |
| No truth tables, result memorization, extra simulation, seed search, or misleading coverage surrogate | patch audit |
| The set is a valid CS at alpha=0.05; the empirical coverage and false-flag checks do not prove this | patch audit |

A statistical guard failure gives the worst score 500. Malformed predictions or inconsistent certificates
are invalid (Infinity). A consistent certificate does not by itself prove that the code obeyed the information rules.

Measured validation baseline: **267 batches** (primary stops 178 and 356), seed standard deviation **0**, over seeds
1,2,3 on 8 CPUs, with coverage 0.95 (`bernoulli_mix`) and 0.98 (`circle`) and 0 of 20 false flags per scenario.
PASS requires relative improvement at least max(2%, twice the reference seed standard deviation divided by its mean).

## Baseline

Method: `C_bbox ∩ lp-safe` (`HEDGE_nd_ELLIP_BBX_SAFE`), official repository
https://github.com/fdenoodt/higher-dimensional-confidence-sequences at
`ccf280f4360c8e041ba238a2b458794b8149964f`. Core code is in
`code/adaptive_sample/conf_sequences/hedged_capital/`. Each coordinate runs a hedged betting wealth process
(truncation c=3/4, theta=1/2). The portfolio keeps the uniform average wealth below 1/alpha=20; the A/B set is its
outer approximation by a box and an lp-ellipsoid. The snapshot keeps the reproduction's CPU caching and exact batching
patches, and `code/bench_run.py` adds the entry points, certificate export and stream scheduling. No method
hyperparameters were retuned.

Training is configuration-only. Prediction runs up to BENCH_CPUS independent single-threaded workers and stops
computing a stream once its first decision occurs. The original 5×25000 Monte Carlo volume samples, adaptive
powers 6..16 and grid size 2001 remain unchanged. On 8 CPUs, mean training time is 3.49 s (0.058 min); mean validation
prediction time is 49.24 min. One worker spends about 5.5 min on a significant A/B stream and 11 min on an unsubstantial
one. The memory allocation is 16 GB; peak RSS was not measured.

## Interface (frozen)

- `bash code/train.sh --out DIR`: write `config.json` and training metadata.
- `bash code/predict.sh --model DIR --input PATH --output FILE`: write the NPZ prediction contract in `bench/io_spec.md`.
- Environment: `BENCH_SEED`, `BENCH_CPUS`, `BENCH_DATA_DIR`, `BENCH_BENCH_DIR`, `BENCH_CODE_DIR`. Supplied paths are absolute.
- Dependencies are rebuilt from `code/env.lock`. Only pinned PyPI packages are supported; no editable, git or local installs.

## Research contract

**Objective.** Minimize the first valid A/B stopping time for the supplied bounded multivariate mean problem, subject
to the original significance, uncertainty-volume, coverage and false-flag requirements.

**Fixed research objects.** Changing the input distribution, estimand or decision rule changes the task.

- The supplied ten-coordinate A/B observations and their original population-mean coordinates; the five-dimensional
  coverage problems and their observation models.
- The observation streams and splits, per-stream independence, and permitted observation prefixes.
- Alpha 0.05, significance threshold 0.5, volume-radius tolerance 0.02, horizon 500, and the order of stopping decisions.
- The 20-checkpoint joint-coverage and replicate false-flag checks, mean-stop metric and Monte Carlo volume protocol.

**Editable methods.** Every method must preserve statistical validity, the permitted information prefix and the
original coordinate meanings.

- The CS construction, predictable betting strategy where used, within the fixed c, theta and weights, and geometric approximation. An lp shape is a baseline
  choice, and no particular neural architecture or loss is required. General numeric set predicates are supported
  alongside the original geometric and betting certificates.
- Numerical algorithms, internal representations and implementation. Internal transformations must map the resulting
  confidence set and decisions back to the original mean coordinates.
- Configuration selection on validation, subject to the no-memorization and per-stream information rules.

**Forbidden substitutions.** The truncation constant c=3/4, hedge theta=1/2 and uniform portfolio weights are fixed, because the experiment compares constructions at the same stake settings. The set must be valid at alpha=0.05 for every stream bounded in [0,1]^D: do not assume a narrower support or a known variance taken from the simulator or validation, and do not lower the evidence threshold to use the slack in the coverage floor or false-flag cap. At a volume-decided stop at least 100 of the 125,000 Monte Carlo points must fall inside the set, otherwise the prediction is invalid (scorer).

- Replacing the mean coordinates with a different embedding or an easier estimand, or scoring a transformed target
  without mapping back to the original coordinates.
- Changing the supplied observations, stream distribution, stopping thresholds, uncertainty measure, coverage
  aggregation or volume sampling rule.
- A separate optimistic coverage construction unrelated to the A/B method, future observations, private scenario
  identity or memorized distribution truths.
- Fitting reusable model state or learned preprocessing on validation, including refits and distillation.

## Rules

`bench/`, `data/`, this README, the certificate format, and entry-point arguments are frozen.
The judge runs seeds 1,2,3 with 8 CPUs and 16 GB allocated.

Predictions for a stream may depend only on that stream's allowed prefix and a
configuration saved by training. Do not use future observations, other instances, ID ordering, latent scenario
identity, extra generated streams, outside data, memorized answers, truth tables including the finite table of
distribution truths, or a stream identity reverse-engineered from a generator seed. A/B row t uses observations
strictly before t; coverage checkpoint k uses the corresponding prefix given in the specification. Derive the volume
RNG seed only from BENCH_SEED and ID. No searching seeds for favorable Monte Carlo results. Coverage certificates must
faithfully represent the method being evaluated; retaining unrelated optimistic coverage claims violates the protocol.

## Known pitfalls

The authors label a zero-observation row as time 1. Preserve this convention: stop 178 uses 177 observations.
The volume radius is volume**(1/10), not half an interval width. In the baseline, coverage membership uses the
exact hedged wealth portfolio; the A/B bounds/volume use its geometric approximation. Coverage is joint across saved
checkpoints, not coverage at every integer time.

The simulator's purchase coordinate has an indirect treatment effect. Truth labels account for it.
The wrapper disables selected NumPy SIMD kernels to match the reproduction's endpoint arithmetic.
Use a file handle with `np.savez_compressed`; otherwise NumPy appends `.npz` to the requested output path.
Each worker is single-threaded; avoid nested joblib or BLAS pools.

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
