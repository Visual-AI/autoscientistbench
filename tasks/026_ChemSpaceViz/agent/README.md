# Efficient visualization of chemical space

## 1. Goal

Improve `auc_mean` (higher is better) for project-after-clustering chemical-space visualization. Given binary ECFP4 fingerprints and a fixed BitBIRCH partition, produce a two-dimensional layout for every molecule. Clustering first identifies dense chemical neighborhoods. Holding these groups fixed makes it possible to compare how well candidate layouts and independent PCA preserve each group's neighborhood relations. This is a **fixed-instance optimization task**: evaluation measures neighborhood preservation on the supplied instance, with a compute cap. It does not measure generalization to unseen molecules.

## 2. Data

- `data/train/input.npz` provides packed ECFP4 fingerprints, molecule IDs and fixed cluster boundaries. Fingerprints have 1024 bits (Morgan radius 2); the partition was formed using BitBIRCH diameter clustering at threshold 0.65, branching factor 1024 and BF refinement with `n_largest=1`, retaining clusters with more than 20 molecules.
- `data/val/input.npz` is the validation instance; `data/val/labels.npz` supplies its input-derived information to the scorer. These contain the same fixed observations as the designated training input, with no supervised labels. Fit layouts and learned preprocessing only from `data/train/input.npz`. Use validation files only for evaluation and method, hyperparameter or checkpoint selection.
- The judge holds its authoritative evaluation inputs and evaluator. The fixed-instance protocol has no private target labels and makes no statistical hold-out claim.
- Use only the supplied instance. Keep the fingerprints and cluster membership fixed; do not add external data. The precise array and coordinate formats are in `bench/io_spec.md`.

## 3. Metric and scoring

`bench/scorer.py` measures neighborhood preservation separately within each fixed cluster. It ranks neighbors using Tanimoto distance on the original binary fingerprints and squared Euclidean distance in the cluster's 2D layout. For cluster size n, QNN(k) counts ordered molecule pairs whose ranks in both spaces lie from 1 through k, then divides by n*k. Rank zero is excluded. A cluster's AUC is the arithmetic mean of QNN(k) over k=1,...,n-1. `auc_mean` is the unweighted mean of cluster AUCs; `auc_size_weighted` weights them by cluster size for exact ties. The theoretical upper bound is 1. Stable ties follow canonical input order.

| Guard | Enforcement class |
|---|---|
| Exactly the fixed molecule ID set; no duplicates or omissions | scorer, also validator |
| Exactly two finite coordinates for every molecule | scorer, also validator |
| Original fingerprints and partition define evaluation | scorer, using frozen labels |
| Recompute layouts during the run; no hard-coded coordinates or memorized results | patch audit |
| No external data, network, or evaluator modifications | patch audit; runtime isolation |

Invalid outputs have raw score 0 and `auc_mean=Infinity` as the harness rejection sentinel. A zero or tied layout with valid format is scored normally. The judge archives the primary mean and secondary diagnostic. Its current automatic PASS decision uses the primary metric; exact-tie ranking uses the secondary diagnostic.

Baseline on the validation instance: **0.8056602755246448**, σ=0 over the single deterministic reference run. Size-weighted AUC: **0.8081922684411398**. `bench status` also reports the baseline. PASS requires relative primary improvement of at least `delta=0.0001` (0.01%); seed noise can raise that threshold.

## 4. Baseline

The official implementation is [mqcomplab/NAMI](https://github.com/mqcomplab/NAMI), upstream commit `d0cfec976274073ed5162f69e8d999d908a1303e`. `code/NAMI/` preserves its source; `code/main.py` is the headless adapter.

The baseline fits an independent `PCA(n_components=2, random_state=42)` to each cluster's unpacked float32 fingerprints, preserving the reproduction's float32 coordinate output. The method's internal PCA seed remains the original fixed parameter. `BENCH_SEED` initializes process RNGs and is recorded with the model; the baseline is repeatable across harness seeds. Clustering and fingerprint generation are frozen data preparation. There is no epoch selection or supervised training.

Measured on 8 CPUs: training **3.57 s** and validation prediction **2.87 s**. Memory allocation is 16 GB.

The adapter adds data loading, fresh fitting, model saving, prediction by ID, and environment-based thread limits. Cached published layouts are not supplied. NumPy/scikit-learn versions match the reproduction. The GUI and its optional dependencies are not part of the executable benchmark path.

## 5. Interface (frozen)

- `bash code/train.sh --out DIR` fits the designated training instance from `data/train/input.npz` and writes the model directory.
- `bash code/predict.sh --model DIR --input PATH --output FILE` writes CSV `id,x,y`, one row per input ID, in any order.
- Environment: `BENCH_SEED`, `BENCH_CPUS`, `BENCH_DATA_DIR`, `BENCH_BENCH_DIR`; paths passed by the harness are absolute.
- Commands: `bench status`, `bench eval-dev --seed 1`, `bench validate-output FILE`, `bench submit --final`.
- Dependencies are pinned in `code/env.lock`. Only reproducible `name==version` PyPI pins are supported; no editable, Git or local installs. Python is 3.11.

## 6. Rules

The research scope below determines which components may change. Supplied data and splits, trusted tests, information boundaries and the evaluation metric remain fixed.


The two-dimensional layout algorithm under `code/` is editable within the research scope below, including replacement of PCA. Keep `data/`, `bench/`, this README, input fingerprints, cluster membership and output format unchanged. Frozen preprocessing specifies the objective; it is not a tunable method parameter.

Training and prediction run without network with up to 8 CPUs and 16 GB memory. The judge rebuilds code in a clean environment and reruns it from scratch. The baseline uses one harness seed; stochastic replacements must honor `BENCH_SEED`. Predictions must be computed during the allowed stages; hard-coded coordinates, memorized paper results, or attempts to access private judge artifacts are disqualifying. Artifacts computed by `train.sh` during the run may be saved in its output directory.

## 7. Known pitfalls

Respect the cluster boundaries and retain all input IDs. Distances and neighborhood ranks are computed separately inside clusters; global arrangement of cluster centers does not affect the metric. Tied fingerprints and coordinates use stable canonical row order, including the original rank-zero convention. Do not substitute trapezoidal integration for the arithmetic mean of QNN. The scorer handles extreme finite coordinate magnitudes by uniform within-cluster scaling.

Prediction paths may have no extension; write the exact path. All files required at runtime must be committed under `code/` or provided as public data. `PYTHONSAFEPATH=1` requires the entry scripts' explicit module path. Set BLAS/OpenMP threads from `BENCH_CPUS` to avoid oversubscription. The original GUI requires additional packages and is not the benchmark entry point.

## Research scope

**Objective.** Optimize a two-dimensional layout of the existing fixed molecular fingerprint clusters for the original co-ranking QNN neighborhood-preservation AUC.

**Fixed research objects.** The supplied 1024-bit ECFP4 observations, BitBIRCH partition, molecule identities and per-cluster population; two-dimensional coordinate output, Tanimoto reference geometry, stable rank/tie conventions and unweighted QNN-AUC objective.

**Editable methods.** Projection/embedding and numerical layout optimization algorithms, including replacing PCA; internal computational representations and training objectives that optimize the same fixed-instance layout problem.

**Forbidden substitutions.** Changing fingerprints, clustering or membership to create an easier geometry; adding output dimensions, changing high-dimensional distances, supplying cached layouts or claiming held-out generalization. When retaining the supplied PCA pipeline, changing only random_state=42 does not count as a method improvement.

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
