# APGC: transductive multi-view graph clustering on Wisconsin

## 1. Goal
Improve `acc` (higher is better) for graph clustering with Attribute-aware
Bidirectional Structural Refinement and Pseudo-label Guided Multi-level Fusion
(APGC). The input is the Wisconsin WebKB graph. The released loader duplicates
its graph as two views. This benchmark fits the full graph without truth labels,
uses unsupervised checkpoint selection, and evaluates assignments on labels held
by the judge. You may improve the method under the rules in §6.

## 2. Data
- `data/train/graph.npz`: the full graph, 251 nodes and 1703 features; keys
  `ids`, `features`, `adjacency`. Features and adjacency are float32; adjacency
  is symmetric with self loops. Opaque IDs follow graph row order.
- `data/val/input.csv`: full graph IDs. `data/val/labels.csv`: 50 public node
  labels, uniformly sampled once, with header `id,y`. Class IDs are 0 through 4.
- Final evaluation labels are held by the judge. Graph inputs are shared in
  this transductive setting; hidden truth is never available to your code.
- Inputs request the full graph. The scorer applies its own labeled-node mask.
  Hungarian matching is computed separately for each scoring mask.
- Do not re-split, relabel, or obtain data from outside sources. Public labels
  support development evaluation; training and checkpoint selection must remain
  unsupervised.

## 3. Metric and scoring
`bench/scorer.py` computes Hungarian-matched ACC and macro-F1 separately for
five independent clustering runs, then takes their means. `acc` is mean ACC if
both scorer guards pass. Guard violations return Infinity, which the harness
rejects as a failed submission. Diagnostics include ungated ACC and macro-F1.

| Guard | Class | Enforcement |
| --- | --- | --- |
| Mean macro-F1 ≥ 0.30 | scorer | Mean across the five prediction columns |
| Exactly five clusters per full graph run | scorer | Each prediction column must use IDs 0..4 |
| Five independent runs honoring the seed | patch audit | No copying one fitted run into multiple slots |
| Unsupervised fitting and checkpoint selection | patch audit | Do not read public or hidden truth in training |
| No external labels or precomputed answers | patch audit | Use only the supplied graph and current run's fitted state |

The score uses Munkres matching and sklearn ACC/macro-F1, as in the released
`evaluation.py`. For a scoring mask that lacks a class, the evaluator retains
the fixed five-class assignment space instead of inserting artificial cluster
assignments. The baseline's method-internal pseudo-label matching retains the released code; alternatives may replace it.

The measured validation baseline passes both scorer guards. Mean ACC is
0.556000 (population standard deviation 0.031496), and mean macro-F1 is 0.368490.
The judge uses three outer seeds and requires relative improvement at least
`max(0.02, 2 × reference_std / reference_mean)`.

## 4. Baseline
Official code: https://github.com/TianxiangZhao0474/APGC at
`542f5949d21f767c51b4551599e7a1ee83dc85b8`. `models.py` contains the model,
`train.py` the loss and training loop, and `settings.py` the released Wisconsin
settings: hidden 256, latent 32, lr 0.0005, weight decay 0.00001, T0 1300,
patience 1000, sigma 0.01, beta 0.4/-0.2.

The baseline selects the first strict maximum of embedding silhouette, using
the released patience and training loop. Fitting and checkpoint selection use
no truth labels. Pseudo-label matching compares fitted cluster assignments.

`entry.py` trains five independent runs. For outer `BENCH_SEED=s`, internal
seeds are `2023 + 5*(s-1) + slot`, slots 0..4. At 32 CPUs, five processes use
three threads each. The CPU allocation is an upper bound. Measured training
times are 147.93 / 158.93 / 144.81 seconds per outer seed, each including five
runs. Validation prediction takes 3.37 / 3.43 / 3.33 seconds. The memory
allowance is 16 GB. The selected checkpoints precede the contrastive-stage
threshold T0, because silhouette peaks early.

## 5. Interface (frozen)
- `bash code/train.sh --out DIR`: writes everything needed for prediction into
  DIR, including each fitted clustering and selected checkpoint.
- `bash code/predict.sh --model DIR --input PATH --output FILE`: emits
  `id,pred_0,pred_1,pred_2,pred_3,pred_4` in exactly the input row order.
- Full format: `bench/io_spec.md`. Entries must be finite integer class IDs.
- Environment: `BENCH_SEED`, `BENCH_CPUS`, `BENCH_DATA_DIR`,
  `BENCH_BENCH_DIR`, `BENCH_CODE_DIR`. Honor seed and CPU limits.
- Development: `bench status`, `bench eval-dev --seed 1`,
  `bench validate-output FILE`, `bench submit --final -m MESSAGE`.
- Dependencies: `uv pip install` into the supplied venv; submissions freeze
  `code/env.lock`. Only `name==version` pins from PyPI/PyTorch CPU are supported.
  The exact Euclidean k-means dependency is vendored in `code/kmeans_pytorch/`.

## 6. Rules

The research scope below determines which components may change. Supplied data and splits, trusted tests, information boundaries and the evaluation metric remain fixed.

You may change the unsupervised clustering estimator, internal graph representations,
surrogate losses and optimization under `code/` within the research scope below. `data/`, `bench/`,
this README, input/output formats and script signatures are frozen.

The judge retrains from scratch with three outer seeds. All five inner runs
must be independently seeded; their predictions are scored separately, not
combined as an ensemble. Public validation scores may guide development, but
truth labels must not enter fitting or checkpoint selection.

The task allocates 32 CPUs and 16 GB memory. There is no network during training or prediction. All fitted
state must be generated in the current run and saved under `--out`.

Reading or reconstructing hidden truth, embedding labels or published answers,
using external data, and hard-coding cluster assignments are disqualifying.
Saving and returning assignments actually learned from this run's graph is
valid for transductive clustering.

## 7. Known pitfalls
Cluster IDs have no semantic class meaning before Hungarian matching. Output
all requested nodes even though public labels cover only some nodes. Avoid
using public labels to align assignments; the scorer performs its own match.

The model includes node-specific fusion weights, so graph row order matters.
Keep original preprocessing and duplicate views when reproducing the baseline.
The baseline prediction reloads its selected fitted assignments; alternative methods may use their own unsupervised inference procedure. A path without an extension is valid for `--output`. Scripts
must export their import path because the harness enables `PYTHONSAFEPATH`.
A clean git checkout is used for judging, so commit every required code file.

## Research scope

**Objective.** Cluster the fixed Wisconsin graph into five groups without fitting to truth labels, maximizing five-run Hungarian-matched accuracy subject to macro-F1 validity.

**Fixed research objects.** supplied full graph, node attributes, node identities and independent label masks; unsupervised fitting/checkpoint selection, five-cluster output per run, five independent runs and original ACC/F1 scoring.

**Editable methods.** unsupervised graph-clustering architectures, internal graph representations, surrogate losses and optimization; unlabeled checkpoint selection and inference, including replacement of the APGC model.

**Forbidden substitutions.** changing observed graph/attributes or task labels, turning the task into supervised node classification or label-guided checkpoint selection; changing the cluster count, evaluation masks, Hungarian matching or duplicating one fit across independent runs. When retaining the supplied APGC pipeline, changing only T0=1300, lr=0.0005, weight_decay=0.00001, patience=1000, sigma=0.01, beta_up0=0.4, beta_low0=-0.2, hidden_dim=256 or latent_dim=32 does not count as a method improvement.

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
