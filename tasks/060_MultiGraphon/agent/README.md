# Low-Complexity and Consistent Graphon Estimation from Multiple Networks

## 1. Goal

Estimate a graphon from each collection of 200 observed undirected networks, with sizes sampled uniformly from integers 10 through 100. Improve `mise` (lower is better). The method may be changed within the rules below.

## 2. Data

- `data/train/` is empty. Fit each collection using only its own observed networks. Validation may select methods and settings, but cannot fit reusable models or learned preprocessing.
- `data/val/input.npz` and `data/val/truth.npz` contain 36 validation collections, three from each of the twelve selected graphons (IDs 1–11 and 13). Validation master seed is 20261030 and shuffle seed is 20261031.
- Final evaluation data are not available in this workspace.
- `bench/generator.py` freezes the official generation process. Input contains observations only: opaque collection IDs, network sizes and packed adjacency matrices. Latent positions, graphon identity, true graphon and generation seed are not inputs. Formats are in `bench/io_spec.md`.
- Each prediction must depend only on its own observed collection and `BENCH_SEED`. Validation labels are for offline assessment and choosing general method settings. Do not re-split, relabel, add outside data or generate extra simulated collections.

## 3. Metric and scoring

`bench/scorer.py` computes the exact step-function error used by the reproduction: separately degree-sort the estimated block matrix and the zero-diagonal 1000 by 1000 reference, expand the estimate as equal-width constant blocks, and average squared error off the reference-grid diagonal. Multiply by 1000, average trials within each graphon, then average the twelve graphons equally. There is no bilinear interpolation. The theoretical lower bound is zero.

| Guard | Enforcement | Consequence |
|---|---|---|
| Exact IDs; finite symmetric probability matrices in [0,1]; positive finite times; specified shapes | scorer and validator | Invalid output: Infinity / format rejection |
| Independence from opaque IDs and batch ordering | protocol and patch audit | Predictions must not change when collection IDs or batch order change |
| No truth-table lookup, model-identity inference for choosing a known truth, hard-coded outputs, extra simulation, or unmeasured method computation | patch audit | Protocol violation disqualifies the submission |

Reported time is diagnostic and is not a trusted timing certificate. Timing scope, collection independence and truth-table avoidance require patch audit.

Measured validation baseline: **mise = 14.91276117**, algorithm-seed standard deviation 0.0. Validation Monte Carlo standard error across the trial samples is 0.10008; it is distinct from algorithm-seed variability. The baseline is deterministic, evaluated with one algorithm seed. PASS requires a relative reduction of at least delta = 3%; the general threshold is max(delta, twice reference seed standard deviation divided by its mean).

## 4. Baseline

The official [JGS repository](https://github.com/RolandBonifaceSogan/JGS-graphon), commit `b6af1306c78868b5bf81de4d7b845c1e7d91b450`, is under `code/`. `src/jgs_estimator.py` jointly ranks normalized degrees (ordinal ties), then estimates block edge frequencies. It uses the automatic block rule `floor(min(S**0.25, N/(3*(M+log(N)))))`, with smoothing and interpolation disabled. Here M is the number of networks, N is the total number of nodes, and S is the sum of squared network sizes.

The adapter uses independent single-threaded process workers, driven by `BENCH_CPUS`. Packed uint8 observations are restored to integer adjacency matrices before estimation. There is no fitting across collections; train writes `config.json` and predict performs estimation.

The task uses the step-function metric. Graphon 12 is excluded by the task definition. Measured on 8 CPUs: train 1.62 s, validation predict 4.77 s. Validation mean per-collection estimation time is 0.2440 s. The memory budget is 8 GB; peak process-tree memory was not measured. Prediction times estimation only; the scorer evaluates error separately.

## 5. Interface (frozen)

- `bash code/train.sh --out DIR` writes all configuration needed by predict.
- `bash code/predict.sh --model DIR --input PATH --output FILE` reads observations and writes NPZ to the exact output filename.
- `BENCH_SEED` is the algorithm seed; `BENCH_CPUS` is the total worker/thread budget; `BENCH_DATA_DIR`, `BENCH_BENCH_DIR` and `BENCH_CODE_DIR` identify the mounted directories.
- `bench status`, `bench eval-dev [--seed S]`, `bench validate-output FILE`, and `bench submit [--final] [-m MSG]` provide the development/submission workflow.
- Install dependencies into the active venv using `uv pip install`; submission records `name==version` pins in `code/env.lock`. Only PyPI pins can be rebuilt. Python is 3.11.

## 6. Rules

Everything under `code/` may be improved, including graphon estimators, auxiliary working models and fitting losses, optimization and numerical settings for the same graphon target, while preserving script signatures, output format and the per-collection observation-only protocol. `bench/`, `data/`, `README.md`, and the prediction format are frozen.

The CPU budget is 8 and the declared memory budget is 8 GB. Train and predict have no network access. All dependencies must be rebuildable from the lock; scripts must honor `BENCH_SEED` even though the baseline has no algorithmic randomness.

Do not access or infer private split membership, use seed searches to reconstruct observations, memorize validation IDs, embed paper results or graphon truth functions in the estimator, identify one of the table graphons and return its known reference, or simulate extra data. The public generator's finite truth table is not an allowed prediction dictionary. General estimators and settings chosen using validation performance are allowed. All computational preprocessing that depends on a collection and contributes to its estimate belongs in that collection's measured time.

## 7. Known pitfalls

The baseline outputs a block matrix, not a resized image. Sorting, zeroing reference diagonals before sorting, and ordinal tie handling affect the metric. Within-block diagonal values of the estimate should not be zeroed.

Each process uses one numeric thread; use at most `BENCH_CPUS` such workers. Prediction output has no guaranteed extension: `np.savez(path)` appends one, so write through an open binary file handle. NPZ object arrays/pickles are disallowed. Script imports must work with `PYTHONSAFEPATH=1`; the adapters explicitly set `PYTHONPATH` and load frozen helpers by absolute path.

A fixed set of graphon definitions is publicly known, which makes truth-table lookup a patch-audit issue. A numerically excellent estimate does not prove compliance. Positive reported times remain subject to patch audit.

## Research contract

**Objective:** Estimate the graphon of each fixed collection of observed networks, minimizing its original degree-sorted step-function off-diagonal L2 error.

**Fixed research objects:** The latent graphon target, observed collection of 200 undirected networks, twelve selected designs and per-collection inference boundary remain fixed. Symmetric probability step-matrix output, degree-sort convention, equal-width expansion to the 1000-grid, off-diagonal L2 times 1000 and equal graphon aggregation remain fixed.

**Editable methods:** Graphon estimation models, auxiliary losses, degree/order estimators, block count/smoothing and numerical methods may change, yielding the required step-function representation. The JGS automatic block rule and estimator are baseline methods; the score does not compare a fixed Bayesian posterior.

**Forbidden substitutions:** Do not redefine the target as a resized image or alternative graph metric, replace the fixed degree-sorted loss, or classify graphon identity then return a memorized reference function. No true latent positions, external/simulated collections, cross-collection fitting or hidden labels/seed lookup. When retaining the supplied JGS estimator, changing only smoothing=False or k=None does not count as a method improvement.

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
