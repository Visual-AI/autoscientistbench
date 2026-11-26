# Dropout allocation at the edge of chaos

The judge retrains your code from scratch, scores held-out images, and compares it with the archived baseline.

## 1. Goal
Improve `final_loss_reduction_pct` (higher is better) by finding a better layerwise dropout allocation for the specified critical ReLU MLP. The network and matched training experiment are the research objects. This task uses the final-epoch loss endpoint of the MLP overfitting setting from “Dropout Universality: Scaling Laws and Optimal Scheduling at the Edge-of-Chaos” (ICML 2026). The critical ReLU MLP and layerwise dropout allocation in §4 provide the baseline. Hidden evaluation images have the same preprocessing as the supplied validation images.

## 2. Data
- `data/train/train.npz`: 4,000 training images, with `ids`, float32 `x` of shape `(N,3,32,32)`, and integer labels `y` in 0–9.
- `data/val/input.npz` and `data/val/labels.npz`: 1,000 validation images and public labels. This is a stratified 20% holdout from the paper's training subset, materialized with split seed 20261030.
- The test split is held by the judge and is never readable here. Its inputs are supplied only during judge prediction.
- IDs are supplied identifiers; files are sorted by ID. Inputs already have the authors' channel normalization: mean `(0.4914,0.4822,0.4465)`, standard deviation `(0.2470,0.2435,0.2616)`. Use these supplied normalized tensors with the specified row-major flattening; additional augmentation or changed preprocessing would alter this controlled experiment.
- The baseline fits only `data/train`. Validation may be used for model and method selection. Final candidate fits also use only `data/train`. No external data, reconstruction of public dataset holdouts, relabeling, or new data splits are allowed.

## 3. Metric and scoring
The frozen scorer computes `100 * (1 - mean_final_CE / 2.20679)`, where the denominator is the specified fixed Constant-arm normalization. It averages cross-entropy across images and then across 25 independently trained replicas. Each replica contributes the logits of its submitted final model; it does not score an ensemble or select checkpoints using hidden data. `accuracy_pct` is diagnostic only.

| Guard | Class and enforcement |
| --- | --- |
| Finite correctly aligned logits, all 25 replicas and declared seed mapping | scorer; validator checks the tensor format and IDs |
| Fixed-network weights, manifest, seed mapping and reconstructed logits | judge probe `bench/check_model.py`; training provenance still requires patch audit |
| Independent training replicas, lawful data use, honest seed handling and declarations | patch audit |

Violations return `Infinity` and fail, regardless of metric direction. Run `python bench/check_model.py MODEL_DIR --input data/val/input.npz --pred PRED` after development evaluation. The probe reconstructs the specified MLP from tensor-only weights without importing candidate code. Training provenance and replica independence require patch audit.

Baseline validation score: **16.641494** (population standard deviation **0.078501**, three groups), also available through `bench status`. The judge uses three groups of replicas. PASS requires relative improvement of at least `max(delta, 2 * baseline_std / abs(baseline_mean))`; the task-specific delta is **3.0%**. This is a relative gain in the score, not an accuracy-point threshold.

## 4. Baseline
The baseline is Step (early): dropout `[0.2,0.2,0.2,0,0,0]` (up to floating-point representation). Official code: https://github.com/luklacasito/icml at `ff45440106199e5c25dc8625e7fb6c5f9f345786`. Model: `code/benchmarks/original/relu.py`; allocation: `code/utils/schedules.py`; batch training: `code/utils/training.py`; offline entry points: `code/main.py`.

Baseline settings: input 3072, six hidden layers of width 256, ten output logits, critical Gaussian initialization with variance parameters 1.98 and 0.02, float32 CPU computation, cross-entropy, AdamW lr `1e-4` and weight decay `1e-7`, cosine schedule `T_max=75`, `eta_min=1e-7`, batch size 100, 75 epochs, no gradient clipping or AMP. These are the matched controls of the dropout-allocation experiment. The Step (early) allocation itself is replaceable.

The baseline uses the supplied training tensors, performs no intermediate evaluation or checkpoint selection, honors environment seeds, and exports final model artifacts and predictions. The fixed normalization means the Constant arm need not be retrained. Each group trains 25 models, using four independent workers with four Torch/BLAS threads each at 16 CPUs. Single-thread arithmetic changes training trajectories; four threads per model describe the measured baseline. Candidates must remain within the total CPU allocation.

Measured on 16 CPUs: training **131.08–142.39 s** per group; validation prediction **8.13–8.33 s**. The memory limit is 16 GB.

## 5. Interface (frozen)
- `bash code/train.sh --out DIR`: train 25 independent models and write their artifacts plus `DIR/model_card.json`.
- `bash code/predict.sh --model DIR --input PATH --output FILE`: load the saved models and write raw logits in the exact NPZ format specified by `bench/io_spec.md`.
- Environment: `BENCH_SEED`, `BENCH_CPUS`, `BENCH_DATA_DIR`, `BENCH_BENCH_DIR`, `BENCH_CODE_DIR`. All paths passed by the judge are absolute. Training seeds are `42 + 25*(BENCH_SEED-1) + r`, `r=0,...,24`; preserve this mapping.
- `bench status`, `bench eval-dev [--seed S]`, `bench validate-output FILE`, and `bench submit [--final] [-m MSG]` provide the development workflow.
- Dependencies are pinned in `code/env.lock`. Only rebuildable `name==version` pins from PyPI / the PyTorch CPU index are supported. The baseline uses Torch 2.1.0+cpu and NumPy 1.24.3.

## 6. Rules

The research scope below determines which components may change. Supplied data and splits, trusted tests, information boundaries and the evaluation metric remain fixed.

Change the dropout allocation or replace its search procedure, subject to the preset limit under Forbidden substitutions. Preserve the fixed network, critical initialization, CE training objective and matched controls in §4, including the 75-epoch final endpoint. All 25 replicas use the same chosen allocation. Equivalent computational implementations are allowed. All fits, including final candidate fits, must use training data only; validation is for evaluation and selection.

`data/`, `bench/`, `README.md`, and the output format are frozen. Model cards must describe the actual saved predictor. Predictions may depend only on supplied images and models fitted before prediction. Reading or guessing evaluation data, embedding labels or memorized results, choosing checkpoints from evaluation input, hard-coding predictions, or reporting a false model card disqualifies the submission.

The judge runs three `BENCH_SEED` values at 16 CPUs, with a 16 GB memory limit. Train/predict have no network.

## 7. Known pitfalls
- `logits` has shape `(25,N,10)`; averaging the 25 models into one prediction changes the endpoint and is invalid.
- Save the 75th-epoch model for each replica; use validation only to evaluate and select allocations. Never use hidden evaluation data for selection.
- Write NPZ through an open binary handle: the output filename may have no `.npz` suffix.
- Keep output IDs unique, sorted, and exactly aligned with input IDs. All logits must be finite.
- Keep `PYTHONPATH` setup in both shell scripts because `PYTHONSAFEPATH=1` is set. Data and frozen helpers are read-only.
- Disable dropout at prediction. The frozen probe verifies fixed-network weights and reconstructed logits; initialization, training controls and actual use of the declared dropout require provenance review.
- Carving out validation changes the effective training data and number of optimizer steps per epoch. Compare against the benchmark validation reference, not published full-training-set scores.

## Research scope

**Objective.** Optimize a fixed-budget layerwise dropout allocation in the specified critical ReLU MLP experiment; preserve its final-epoch normalized cross-entropy objective.

**Fixed research objects.** six hidden layers of width 256, input 3072 and ten logits, ReLU activations and critical initialization variances 1.98/0.02; one six-rate allocation shared by 25 independent replicas, rates in [0,1) with mean 0.1; matched experimental controls: float32, cross-entropy, AdamW lr 1e-4 and weight decay 1e-7, cosine T_max 75 and eta_min 1e-7, batch 100, 75 epochs and final checkpoint; supplied normalized tensors without augmentation, training on the train split with validation for selection only, original seed mapping and CE normalization 2.20679.

**Editable methods.** the dropout allocation and its search or optimization algorithm, using training data for fitting and validation for evaluation and selection only; equivalent implementation, data loading and scheduling within the allocated resources.

**Forbidden substitutions.** replacing the controlled MLP with a different predictor or changing its width, depth, activation or initialization; changing the loss, optimizer, epoch endpoint or preprocessing to win a different experiment; changing the total dropout budget, selecting by hidden data, or replacing member CE by ensemble CE. When retaining the supplied `get_dropout_schedule` presets, changing only sched='reverse_step' or h_max=0.2 does not count as a method improvement.

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
