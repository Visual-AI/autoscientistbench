# Spatial isoform variability: simulation power under null-error constraints

Everything below is binding. The judge retrains your code from scratch with its own seeds,
scores held-out simulations, and compares against the archived baseline.

## 1. Goal

Improve `macro_recall` (higher is better) for detecting spatially variable isoform expression
from cell coordinates and binary relative expression values. Use the supplied Sample 1 Stereo-seq coordinate design to compare per-instance spatial
tests against the Moran's I baseline. Detect frontier, square, square with random
surround, and horizontal/vertical line patterns while controlling rejection on random nulls.
You may change the method under the rules in §6.

## 2. Data

- `data/train/coordinates.npz`: the common Sample 1 coordinate design, array `coords`,
  60,437 rows by two columns, float64. Coordinates are in DNB units, 0.5 micrometers per unit.
  No cross-instance training examples are supplied; the baseline fits no reusable weights.
  Validation is only for evaluation and method, hyperparameter or checkpoint selection.
  Do not fit reusable models or learned preprocessing on validation. Per-instance tests
  may use only that instance's observations.
- `data/val/input.npz` and `data/val/labels.npz`: 4,500 public validation instances. There
  are 20 repetitions per pattern/cell-count cell and 100 per null/cell-count cell.
  The public master seed is 20261030; individual simulation/subsample seeds are in labels.
- `bench/generator.py` is frozen. It generates patterns on the full tissue before uniform
  subsampling to 250, 500, 1,000, 5,000 or 10,000 cells. Cell samples are nested within a
  replicate. The split is fixed.
- Patterns use square widths 200/500/1000/2000 micrometers and horizontal/vertical line
  widths 100/250/500 micrometers. Random surround is Bernoulli(0.5). Nulls randomly assign
  fixed full-tissue fractions 0.50, 0.40, 0.30, 0.20, 0.10 or 0.05 to value one.
- Each input contains observations only: opaque `id`, ragged `offsets`, `coords`,
  and `values`.
  Scenario identity, geometry, replicate grouping and generating seeds belong to labels.
- Labels are for offline evaluation and diagnostics only. Do not use them to fit the
  procedure, reconstruct an instance, or choose a prediction. Do not re-split, relabel,
  generate additional simulations, or obtain outside data. Each call must use only its
  own observed instance; shared/nested observations do not permit cross-instance inference.

## 3. Metric and scoring

`macro_recall` is the equal-weight mean of recall over pattern scenario × cell-count cells.
Every provided instance stays in the denominator; a filtered instance is a noncall. The
baseline calls when its raw permutation p <= 0.05 and Moran's I > 0.01, without BY correction.
This simulation metric differs from the real-data SVI count in the paper's main experiment.
The theoretical maximum of the benchmark metric is 1.

| Guard | Enforcement | Violation |
|---|---|---|
| Pooled null rejection <= 0.058 | scorer | score 0 |
| Every null scenario/cell-count rejection rate <= 0.094 | scorer | score 0 |
| Same pattern-agnostic procedure, only own observations; no truth, scenario or ID lookup | patch audit | disqualification |
| Same calls after subsetting, instance reordering and replacing opaque IDs | patch audit | disqualification |

The scorer returns diagnostic rates and a Monte Carlo SE that clusters nested cell sizes by
scenario/replicate. It cannot prove pattern agnosticism from binary calls alone. Invalid
prediction files fail validation; scorer fallback for malformed files is Infinity. The
binding statistical guard rule uses 0, not that malformed-file sentinel.

Baseline validation: `0.4155555556` mean, `0.0003142697` seed SD across three runs.
Validation prediction times: 210.43, 210.79 and 209.39 seconds on 32 CPUs.
The judge uses three algorithm seeds and compares means. PASS requires relative improvement
at least max(3%, twice the reference seed SD divided by its mean); every guard must hold.
`bench status` also reports the baseline validation score and delta.

## 4. Baseline

Official code: https://github.com/tilgnerlab/Spl-IsoFind at
`a7110c4e4e8a4f6fc0ce71645851615657262675`. The statistical method is
`code/src/SplIsoFind/spatially_variable.py::_compute_single_isoform`: binary k-nearest-neighbor
weights, Moran's I, and a permutation p-value. The reproduced simulation settings are
nperm=10000, k=50, mincells=250, mincellspergroup=20 and minority fraction >= 0.05.
The authors' early exclusions and the reproduction's existing cell-type indexing fix remain.

`train.sh` writes `config.json`; `predict.sh` runs one test per instance in
`BENCH_CPUS` single-threaded processes. Algorithm seeds derive from `BENCH_SEED` and the
instance's observations, independently of generating seeds, instance order and opaque IDs.
The current spatial implementation orders neighbor ties by distance and cell position.
Loading the spatial module directly avoids unused plotting/preprocessing dependencies.

Measured on 32 CPUs: train 0.36, 0.41 and 0.36 seconds (configuration only).
Validation prediction times are listed above. The configured memory allowance is 32 GB.
Python is pinned to 3.12.16 with exact PyPI dependency pins.

## 5. Interface (frozen)

- `bash code/train.sh --out DIR` writes configuration/model artifacts to DIR.
- `bash code/predict.sh --model DIR --input PATH --output FILE` writes CSV with exact
  header `id,called`, one finite binary 0/1 call for every input ID; see `bench/io_spec.md`.
- Environment: `BENCH_SEED`, `BENCH_CPUS`, `BENCH_DATA_DIR` and `BENCH_BENCH_DIR`.
  Paths passed by the harness are absolute. Train and predict receive the same seed.
- `bench status`, `bench eval-dev --seed S`, `bench validate-output FILE`,
  `bench submit --final -m MESSAGE`. Development evaluation runs train, predict,
  validation and scoring through the same interfaces.
- Dependencies may be installed into the run's venv with `uv pip install`. Submission
  freezes them into `code/env.lock`; only rebuildable PyPI `name==version` pins are allowed.

## 6. Rules

The research scope below determines which components may change. Supplied data and splits, trusted tests, information boundaries and the evaluation metric remain fixed.


- Free surface: `code/`, including the statistical procedure, features, hyperparameters of a changed
  procedure, implementation and dependencies, while retaining the interfaces and all guards.
- Frozen surface: `data/`, `bench/`, this README, prediction format and script signatures.
- Decisions must be a single pattern-agnostic function of each instance's observed
  coordinates/values and the algorithm seed. No true-scenario-specific parameters,
  truth tables, memorized answers, ID-dependent calls, batch-dependent calls, label
  access during train/predict, or extra simulation from the public generator.
  Spatial features inferred from the current observed instance are legitimate.
- Three judge seeds are used; honor `BENCH_SEED` and the CPU limit. Seeds used for
  simulation generation must never be guessed or reconstructed.
- Resource allocation: up to 32 CPUs and 32 GB memory.
  There is no network during train/predict; all requirements must be available from
  submitted code, public input or the pinned environment.

## 7. Known pitfalls

- Permutation tests are stochastic despite the absence of model training. Keep randomness
  local to an instance; the judge may replace IDs or run only a subset in a different order.
- Null per-cell constraints are stricter than a pooled average. Strong power with one
  invalid null cell still scores 0. Read the scorer's cell diagnostics.
- The supplied baseline uses uncorrected calls. Alternative testing or within-instance multiple-testing procedures are allowed; keep all instances in the denominator and satisfy the same null-rejection guards.
- The baseline uses a process pool and one BLAS/OpenMP thread per process. Other allocations must remain within the total CPU budget.
  The baseline processes the largest instances first to avoid a long tail.
- NPZ input uses ragged offsets and forbids pickle. Output is CSV even without a filename
  extension. Explicit IDs allow row reordering; missing, extra or duplicate IDs fail.
- `PYTHONSAFEPATH=1` means scripts must explicitly set PYTHONPATH when importing local code.
  Frozen scorer helpers load by their own location, independently of submitted modules.

## Research scope

**Objective.** Detect spatially variable expression from each observed coordinate/value instance, maximizing macro recall subject to the original null-rejection controls.

**Fixed research objects.** supplied coordinate design, observation instances, pattern/null distribution and independent-seed split; single pattern-agnostic per-instance decision rule, macro recall and pooled/per-cell null limits.

**Editable methods.** spatial statistics, single-instance estimators, features and numerical testing procedures; algorithmic implementation and permitted configuration using observations only.

**Forbidden substitutions.** replacing the statistical alternative/null definitions, scoring population or null thresholds; fitting to diagnostic truth labels, using scenario-specific answer rules, crossing instance boundaries or generating extra simulations. Positive calls must rest on spatial association between coordinates and expression. A shift in the marginal fraction of ones alone does not count as spatial variability. Expression counts may still set eligibility or conditional calibration. Patch audit enforces this rule. When retaining the supplied Moran pipeline, changing only test='moransI', nperm=10000, k=50, mincellspergroup=20, imb=0.05, alpha=0.05 or I_threshold=0.01 does not count as a method improvement.

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
