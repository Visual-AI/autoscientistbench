# A resource theoretical unification of Mpemba effects: classical and quantum

## 1. Goal
Find a permutation of an initial Gibbs probability vector that minimizes its
combined overlap with the two slowest nonstationary modes of a fixed-boundary
classical Ising chain. Reducing these overlaps suppresses slow relaxation while
preserving the initial probability multiset. Improve `median_log10_c` (lower is
better) on unseen physical instances. You may change the optimization method
under the rules below.

## 2. Data
- `data/train/input.npz` and `data/train/labels.npz`: six training
  instances, generated with seeds 61001–61006. The baseline needs no training data.
- `data/val/input.npz` and `data/val/labels.npz`: the original seven-spin paper
  instance plus five public instances, seeds 62001–62005. Public labels contain
  generation metadata and the frozen objective arrays.
- `bench/generator.py` defines the family: Ns uniform on {6,7,8}, J uniform on
  [-0.5,-0.3], h uniform on [0.1,0.3], initial inverse temperature uniform on
  [0.4,0.6], bath inverse temperature 1. The paper instance is
  Ns=7, J=-0.4, h=0.2, initial inverse temperature 0.5.
- Prediction receives opaque IDs and the initial probability, equilibrium
  probability and two normalized right modes. Only supplied arrays define the
  objective; generation seeds and parameter metadata are withheld at prediction.
  The dimension is observable. See `bench/io_spec.md` for padding and shapes.
- Fit reusable policies and learned preprocessing only on the designated training
  data. Validation is only for evaluation and method or configuration selection.
  Per-instance optimization may use that instance's supplied objective arrays.
  Do not load labels during prediction, replace the supplied instances, identify
  hidden generation seeds, or use external data or stored solution tables.

## 3. Metric and scoring
For each independent run, the frozen scorer computes
`C = sum(abs((modes/pi[:,None]).T @ p[permutation]))` with K=2 modes.
It takes the median of **log10 C** over ten independent runs for each instance,
then averages the instance medians. This additional instance average extends the
paper's fixed problem to the synthetic family. Log10 of the median is different.

| Guard | Enforcement |
|---|---|
| Complete finite CSV, all IDs and ten replica indices | scorer and validator |
| Every state is an exact permutation, including degeneracies | scorer |
| Decisions are independent of other inputs and input order | patch audit |
| Honest timers; ten independent runs; no cached solutions or private metadata | patch audit |

Invalid predictions receive `Infinity`. The diagnostic `score_rule_value` retains
the prescribed finite worst value -0.897. Exact computed zero cost is represented
using the smallest positive float64 value before log10; other costs are unchanged.

Measured validation baseline: **-4.180171613661**, outer-seed standard
deviation **0.059521201515**.
The judge repeats the complete ten-replica experiment with three outer seeds and
compares the mean with its archived reference. PASS requires all guards and
relative improvement at least `max(0.06, 2*reference_std/abs(reference_mean))`.

## 4. Baseline
Official code: https://github.com/alessum/resource_theory_mpemba at commit
`f5847060c11d371e7c10e241f2247695d6ff507c`. The unmodified
`code/tools/notebook_utils.py::optimize_probability_permutation` implements the
simulated annealer. It makes 50,000 proposals, permutes four sampled entries,
uses temperature `max(1e-5,0.02*(1-step/iterations))`, and returns its best state.

Training writes configuration without fitting. Prediction reads the supplied
NPZ arrays and optimizes each instance and replica independently, then serializes
the resulting permutations as CSV. Only NumPy and SciPy are required for these
entry points.

Measured at 32 CPUs, configuration writing took 0.92, 0.92 and 0.97 s across
reference repetitions. Validation prediction took 4.82, 4.83 and 4.83 s,
respectively.
The process pool uses up to 32 workers, each with one numerical-library thread.
Memory budget is 16 GB, not a measured peak. The method has no cross-instance
training stage.

## 5. Interface (frozen)
- `bash code/train.sh --out DIR` writes `config.json` into the model directory.
- `bash code/predict.sh --model DIR --input PATH --output FILE` writes text CSV
  with header `id,replica,permutation,elapsed_s,iterations`, ten rows per input ID.
- `BENCH_SEED` selects the outer random seed; the baseline maps it to the ten
  consecutive optimizer seeds `10*(BENCH_SEED-1)+replica`. `BENCH_CPUS` limits
  workers. `BENCH_DATA_DIR`, `BENCH_BENCH_DIR`, `BENCH_CODE_DIR` provide paths.
- Use `bench status`, `bench eval-dev --seed 1`, `bench validate-output FILE`,
  and `bench submit --final` for development and submission.
- Dependencies are exact PyPI `name==version` pins in `code/env.lock`.
  Install using `uv pip install`; submission freezes the environment.
  Editable, git and local-path installations cannot be rebuilt.

## 6. Research scope

The research objective and objects below are fixed; methods may change within this task scope.

**Objective:** Minimize fixed two-slow-mode overlap by permuting each supplied Gibbs vector, preserving independent-replica median-log aggregation.

**Fixed research objects:**

- Supplied p, equilibrium pi and two normalized slow modes of the same Ising instance; the two-mode overlap cost.
- Feasible outputs are permutations of the original probability multiset, including degeneracies.
- Ten independent runs per instance, median of log10 costs followed by instance mean, original outer-seed and isolation protocol.

**Editable methods:**

- Permutation search, proposals, optimizer, annealing schedule, iteration count, numerical surrogate and implementation, subject to the annealer limit under Forbidden substitutions.
- Validation-selected configuration or a reusable policy fitted only on training data, solving each instance independently; simulated annealing is a baseline.

**Forbidden substitutions:**

- Change probability vectors, Hamiltonian-derived modes, overlap target or permutation feasible set.
- Return arbitrary distributions, alter median/log ordering, or share/select best solutions across required independent replicas.
- Use private metadata, precomputed permutations or another evaluation instance.
- When retaining the supplied simulated annealer, changing only iterations=50000, the four-entry proposal size or temperature=max(1e-5,0.02*(1-step/iterations)) does not count as a method improvement.

- Free surface: `code/`, including the method, iteration count, solver and
  hyperparameters, subject to the annealer limit under Forbidden substitutions; preserve
  the interfaces, ten independent runs and output format.
- Frozen surface: `data/`, `bench/`, `README.md`, problem definitions and scoring.
- The allocation is 32 CPUs and 16 GB memory.
- Honor all three outer seeds. Each replica must run independently; no copying another replica's optimized answer or selecting a best restart
  across replicas. Every decision depends only on its own supplied instance,
  its algorithmic seed and a fixed development-selected configuration.
- No network during train or predict. No hard-coded predictions, paper solutions,
  hidden-seed reconstruction, reading private instance metadata, precomputed
  permutations, dishonest counters or access outside the supplied interfaces.
- Timer honesty and semantic rules are checked by patch review. The scorer
  recomputes numerical objectives and checks feasible permutations.

## 7. Known pitfalls
Ignore padded entries: padded equilibrium values are zero. Preserve the authors'
mode normalization. Degenerate Gibbs probabilities are legitimate: return each
source index once. Compute ten log costs before taking their median. The scorer
recomputes the objective, so reporting an invented cost has no effect. Output paths
may lack a suffix; CSV output must use exactly the provided path. All runtime
paths are absolute; explicitly set Python import paths under `PYTHONSAFEPATH=1`.
Avoid oversubscribing BLAS when using a process pool. Optional upstream notebooks
need additional dependencies and data; they are not the benchmark entry points.

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
