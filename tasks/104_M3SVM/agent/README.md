# Multi-Class Support Vector Machine with Differential Privacy

The judge retrains your code from scratch, evaluates hidden examples, and compares
against the archived baseline. All rules below are binding.

## 1. Goal

Improve `mean_accuracy` (higher is better) for ISOLET spoken-letter classification:
617 real-valued acoustic features, 26 classes, and differential privacy with
epsilon at most 1 and delta 0.00001 per run. The baseline is PMSVM-AGP. Each evaluation trains ten independent
classifiers and averages their individual accuracies. The research object is a raw-feature affine 617-to-26 multiclass SVM with the
selected margin objective. Private optimization, numerical solvers and privacy
mechanisms may be replaced while preserving that model family and objective.
The supplied input rows, label space, ten-run accuracy and per-run privacy
guarantee remain fixed.

## 2. Data

- `data/train/train.npz`: 998 training rows; keys `id,X,y`. `X` contains 617 raw
  float32 features; `y` contains integer classes 0..25. No feature scaling.
- `data/val/input.npz` and `data/val/labels.npz`: 250 validation rows, with keys
  `id,X` and `id,y`. Validation is a stratified 20% carve-out from the available
  training partition, using fixed seed 20261029 before privacy calibration.
- The test split is held by the judge. A single fixed partition is shared across
  all ten fits. No test examples or split-selection seed are available here.
- IDs are original row numbers, sorted in every file. IDs are identifiers, not
  predictive features. Model selection may use validation. Fitted weights use only the supplied training partition;
  validation may guide model selection but may not be merged into training. No outside data, reconstruction of unavailable rows,
  relabeling, or replacement of the supplied split is allowed.

## 3. Metric and scoring

`bench/scorer.py` computes the arithmetic mean of the ten individual accuracies,
not an ensemble's accuracy. Privacy is checked separately for every run.

| Guard | Enforcement |
|---|---|
| Valid IDs, ten class predictions per row, finite data | scorer and validator |
| Verified mechanism-composition epsilon at delta 0.00001 at most 1 per run | scorer |
| Predictions match the frozen raw-feature affine classifier and its privacy ledger | affine certificate check (`bench/check_model.py`) and patch audit |
| Ledger accurately accounts for the actual private training procedure | patch audit |
| Allowed training data only, independent seeded fits, no prediction-time fitting | patch audit |
| CPU and memory budget | judge execution limits; patch audit |

A violated rule has chance utility 1/26. For harness compatibility the scorer
emits `mean_accuracy: Infinity` to force failure, with `rule_score: 1/26` as a
diagnostic. Falsifying an accountant history is a violation even if the numeric
ledger passes. Run `python bench/check_model.py MODEL --input data/val/input.npz --pred PRED`
after development evaluation. A valid submission must pass the affine certificate
check and the patch audit.

Archived baseline validation `mean_accuracy`: **0.43546667**, population standard deviation
**0.01112515** across the three outer seeds. The individual validation values are
0.42760000, 0.42760000, 0.45120000.
The judge uses three outer seeds; each outer seed produces ten independent fits.
PASS requires relative gain at least `max(0.052, 2 * reference_std / reference_mean)`.
Use `bench status` for task metadata.

## 4. Baseline

Official code: https://github.com/JinseongP/private_multiclass_svm at commit
`aa3cfb5ab6a9f923c2dcf24845e1eb408ad20e28`. `code/main-opacus.py`
contains the original multi-margin loss plus inter-class weight regularization
and the original Adam loop. Hyperparameters: 30 epochs, learning rate 0.005
(scale 5.0), lambda 0.001, p=2, nominal Poisson batch 128, clipping norm 1,
weight decay 0, no scaling. The original regularizer normalizer of 1560 and
Opacus 1.2.0 behavior are preserved.

The baseline takes models at the final epoch. Ten fits run in independent single-thread processes, up to
`BENCH_CPUS` simultaneously. Python 3.10.19 and torch 2.0.1+cpu are pinned.

Measured on 8 CPUs: training takes 9.93 to 10.43 seconds
(0.170 minutes on average), and validation prediction takes 3.97 to 4.27
seconds (0.069 minutes on average). The memory allowance is 16 GB;
peak resident memory was not separately measured during reference.

## 5. Interface (frozen)

- `bash code/train.sh --out DIR` writes all model files and `model_card.json`.
- `bash code/predict.sh --model DIR --input PATH --output FILE` writes a CSV
  following `bench/io_spec.md`, regardless of output filename extension.
- Environment: `BENCH_SEED`, `BENCH_CPUS`, `BENCH_DATA_DIR`, `BENCH_BENCH_DIR`.
  Every fit's randomness must depend on `BENCH_SEED`; the baseline uses NumPy
  SeedSequence with the outer seed and run index.
- Commands: `bench status`, `bench eval-dev --seed S`,
  `bench validate-output FILE`, `bench submit --final`.
- Dependencies: install using `uv pip install`; submission freezes the environment
  into `code/env.lock`. Only exact version pins from PyPI / PyTorch CPU are
  rebuildable. Editable, Git, and local-path dependencies are unsupported.

## 6. Rules

Algorithms under `code/`, including private optimization and the numerical
hyperparameters of a changed method, may change while preserving the fixed affine SVM objective,
prediction interface, privacy guarantee and data rules. `data/`, `bench/`, this README, and the I/O contract are frozen.
The judge uses 8 CPUs and 16 GB memory. Training and prediction run without
network.

The judge uses three outer seeds. Honor `BENCH_SEED` and return all ten individual
runs. No reading or guessing unavailable data, hard-coded predictions, memorized
paper results, misleading certificates, or forged privacy ledgers. Never alter
a fitted certificate or fit parameters while processing prediction input. All
models must use `dp-affine/v1` with finite 26-by-617 weights and 26 biases for
each run. `bench/check_model.py` reconstructs predictions directly from raw features using
float32 linear logits and the original smallest-index argmax tie rule. Generic
nonlinear `dp-classifier/v2` artifacts are outside this research contract.

## 7. Known pitfalls

Privacy noise must be recalibrated after changing the population, batch size,
or training duration. The scorer supports the original Poisson-Gaussian RDP
ledger and a mechanism ledger whose Gaussian, Laplace and Poisson-Gaussian
composition bounds it recomputes. It never accepts a self-reported final epsilon
or arbitrary RDP profile. Numeric accounting verifies the declared mechanism;
patch audit must still verify actual data access, sensitivity bounds, clipping,
noise generation and composition. Internal reparameterizations are allowed only when the final raw-feature
predictor remains affine and the same SVM objective is optimized; account for
all private-data use. Nonlinear feature models and cross-entropy classifiers
substitute different research objects. The public ledger is checked numerically, but is not proof
that clipping and noise were really applied. Opacus computes its sample rate
from the loader length, not simply requested batch size divided by population.
Keep the original regularizer behavior in mind when comparing algorithms.

CSV JSON cells require proper quoting. Prediction IDs must match input order;
NaN, missing rows, repeated IDs and classes outside 0..25 are rejected. Keep all
ten runs separate. `PYTHONSAFEPATH=1` requires the scripts' explicit PYTHONPATH.
The input and frozen benchmark directories are read-only. Write artifacts to
`--out` and predictions to `--output`. Validation carve-out and fixed partitions
make benchmark results different from the paper protocol; compare with the
archived benchmark reference.

## Research contract

**Objective:** Improve private optimization of the selected affine multiclass SVM on ISOLET under epsilon<=1 and delta=1e-5, retaining its margin objective and ten independent-run accuracy metric.

**Fixed research objects:** The raw-feature affine map R^617 to R^26, class argmax rule and selected Regularized_loss (multiclass hinge plus .001/1560 times the ordered pairwise squared weight-distance sum) are the research object. Only the supplied training partition fits weights; fixed validation may guide selection. The ten independent fits, original input/label space, per-run DP guarantee, mean accuracy and acceptance rule remain fixed.

**Editable methods:** Private mechanism, privacy accountant representation supported by trusted formulas, solver, optimizer, learning schedule, iterations and equivalent reparameterization may change, subject to the retuning limit under Forbidden substitutions. Auxiliary numerical surrogates must solve the same SVM objective; all private data access, sensitivity, noise and composition require truthful accounting.

**Forbidden substitutions:** Do not replace the affine SVM with nonlinear features, neural classification, an ensemble prediction, cross-entropy target or another regularized objective. No merging validation into the fitted training population, hidden examples/labels, prediction-time fitting, fabricated ledgers or self-reported final epsilon. When retaining the supplied Adam/Opacus pipeline, changing only lr=0.005, epochs=30, batch_size=128 or max_grad_norm=1 does not count as a method improvement.

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
