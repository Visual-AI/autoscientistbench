# SENDAI: sparse NDVI field reconstruction

The judge retrains your code from scratch with independent seeds, scores hidden full-field targets,
and compares your result with the archived baseline.

## 1. Goal

Reconstruct 64 × 64 Tarim Basin NDVI fields from a fixed network of sparse sensor histories,
using a simulated-season full-field prior. Improve `ssim` (higher is better) while satisfying the
RMSE and observation-budget guards. This is an in-sample data-assimilation task: designated sparse
training observations may be used for reusable adaptation. Full-field evaluation targets are withheld.
You may change the method under the rules in §6.

## 2. Data

- `data/train/simulation.npz`: `fields`, 72 × 64 × 64 simulated-season NDVI frames. The baseline
  applies the original bad-frame correction and builds lagged samples, using a chronological
  80/20 internal simulation training/validation split.
- `data/train/observations.npz`: `sensors`, sparse adaptation histories, and `sensor_locs`, the
  fixed 64 row/column coordinates. Only these physical measurements are available for real-period adaptation.
- `data/val/input.npz` and `data/val/labels.npz`: 15 public full-field validation frames,
  opaque IDs, sensor histories, and corresponding NDVI targets. Validation frames were selected
  independently of predictions; use them only for evaluation and method, hyperparameter or checkpoint selection.
- The remaining full-field evaluation targets are held by the judge and are never readable here.
  Input format and preprocessing match validation. IDs do not encode dates or split membership.
- Do not re-split, relabel, or obtain external data. Do not fit to public real-period full-field
  validation targets; use them only for model selection. Simulation full fields and the supplied
  sparse adaptation observations are the fitting data.

## 3. Metric and scoring

`bench/scorer.py` computes each independent model's mean per-frame SSIM with a 7 × 7 window
and skimage defaults. Each frame uses `data_range=max(target.max()-target.min(),1e-8)`.
It then averages the three model scores.
It does **not** score an average of their predicted fields. The theoretical upper bound is 1.

| Guard | Limit | Enforcement class |
|---|---|---|
| Mean of three per-model global pixel RMSEs | ≤ 0.130 | scorer |
| Declared physical observations | ≤ 64 sensors and ≤ 5 lags | scorer |
| Certificate agrees with input and output | same observation budget | certificate check (`bench/check_model.py`) |
| Subset predictions agree with the complete prediction | independent windows | subset replay |
| Actual information use, independent fits, no extra truth or longer histories | follow the supplied interface | patch audit |

A violated guard has rule value -1; the scorer emits `ssim: Infinity` as the harness's failure
sentinel, with `rule_worst: -1` in diagnostics. Certificate and subset consistency are checked
separately from scalar scoring, alongside patch review.

Baseline validation SSIM: **0.49500545 ± 0.00657397** (population SD across three outer repetitions).
Each outer repetition contains three independent fits because the RMSE guard applies to their mean.
The judge uses three outer seeds. PASS requires relative improvement of at least
`max(delta, 2 * reference_std / abs(reference_mean))`; task delta is **2.500%**.

## 4. Baseline

Full SENDAI (DA-SHRED plus two hierarchical INR correction layers), official repository
https://github.com/xswzaqnjimko/SENDAI_framework at upstream commit
`2b663cf6ff4389ff89323256233a060f37ce3281`.
Models and losses/training loops are in `code/model/SENDAI/models.py` and `training.py`;
original hyperparameters are in `code/config.py`.

The baseline uses the fixed measurement network and sparse real-period training observations.
It exports model/scaler state and predictions and honors seed/thread environment variables.
Each scoring unit contains three independent runs. CPU helpers are in `fast.py` and `exact_rng.cpp`;
the C++ helper is built locally during training. The baseline uses 2 × (500 + 200) HF epochs.
It also trains an unused GAN and keeps dropout active in frozen modules during HF training.

Measured reference runtime at **128 CPUs**: train up to **35.17 min** per outer repetition,
validation prediction up to **0.12 min** (including three models).
Memory budget: **32 GB**. Three independent fits run concurrently, with up to 32 HF threads each
and up to four LF threads each. Full-field visibility and the fixed measurement network differ
from the paper's evaluation protocol, so the archived benchmark reference is the comparison target.

## 5. Interface (frozen)

- `bash code/train.sh --out DIR`: train from scratch and write all model/scaler artifacts and `model_card.json`.
- `bash code/predict.sh --model DIR --input PATH --output FILE`: write an NPZ to the exact requested path.
- Prediction `pred` has shape `(3, N, 64, 64)`, one independent model per first-axis entry;
  IDs must exactly match input order. See `bench/io_spec.md` for all required keys.
- Honor `BENCH_SEED`, `BENCH_CPUS`, `BENCH_DATA_DIR`, `BENCH_BENCH_DIR`, and `BENCH_CODE_DIR`.
  Outer seed `s` assigns independent fit seeds `3*s`, `3*s+1`, `3*s+2`.
- `bench status`, `bench eval-dev --seed 1`, `bench validate-output FILE`,
  and `bench submit --final -m MESSAGE` provide the usual development workflow.
- Run `python bench/check_model.py DIR --input data/val/input.npz --pred FILE` after development evaluation.
- Dependencies use Python 3.11.18 and `name==version` pins in `code/env.lock` from PyPI/PyTorch CPU.
  Use `uv pip install` in the provided environment; submission freezes it. Editable, Git and local-path
  dependencies cannot be rebuilt.

## 6. Rules

The research scope below determines which components may change. Supplied data and splits, trusted tests, information boundaries and the evaluation metric remain fixed.


- Editable surface: sparse-field reconstruction models, internal representations, training surrogate losses and adaptation methods under `code/`, within the physical observation problem below.
- Frozen surface: `data/`, `bench/`, this README, script signatures and prediction format.
- The allocation is 128 CPUs and **32 GB** memory.
  There is no network during training or prediction. C++ compilation may use the provided system compiler.
- Each prediction must use only its own supplied sensor window and learned parameters. Do not stitch other
  input windows into a longer observation history, look up validation targets, or reconstruct hidden targets
  from external sources. Model selection may use validation scores.
- Honor the three independently assigned fit seeds. Do not reuse a fitted model across those stochastic runs
  or substitute an ensemble average for their individual predictions.
- Disqualifying: reading or guessing hidden targets/split membership, hard-coded predictions or memorized
  paper results, false observation-budget certificates, and fitting to real-period full-field validation truth.

## 7. Known pitfalls

The lagged target is the final frame of its supplied window. The original converter's endpoint convention
is retained. Sensor coordinates include boundary pixels under the original random placement rule.
Labels retain the original simulation-scaler float32 roundtrip convention. Do not clip fields before scoring.
`np.savez` appends an extension when given a filename; open the requested path in binary mode first.
The judge uses a clean committed code checkout and absolute paths with `PYTHONSAFEPATH=1`; commit every
required source file and set package search paths explicitly. Each evaluation trains three full models,
so development calls are expensive. Building the CPU environment can also take substantial time.

## Research scope

**Objective.** Reconstruct the supplied 64 × 64 NDVI fields from the fixed sparse sensor network and simulation prior. Maximize three-fit mean SSIM under the RMSE and observation limits.

**Fixed research objects.** The supplied simulation fields, real sparse measurements, fixed physical sensor locations and withheld full-field targets define the reconstruction problem. Keep the per-window information boundary, at most 64 sensors and 5 lags, three independent fits and the SSIM/RMSE rules. Zero-observation methods remain allowed.

**Editable methods.** Reconstruction and data-assimilation estimators, architecture, simulation representations, surrogate losses, optimization and sparse-observation adaptation. Fit reusable state and learned preprocessing only on designated training observations.

**Forbidden substitutions.** Easier replacement fields or sensor networks, extra sensors or history, fitting to held-out real full fields, fitting reusable state on validation features or labels, stitching windows, changing target range, SSIM or RMSE definitions, or replacing three scored fits with an ensemble prediction. When retaining the supplied SENDAI pipeline, changing only LR=1e-4, SHRED_EPOCHS=800, SHRED_PATIENCE=30, DASHRED_EPOCHS=1500, DASHRED_PATIENCE=50, BATCH_SIZE=16, MAX_TARGET_FREQ=16, HF_WARMUP=100, lambda_smooth=0.1 or the PEEL_CONFIG values target_k=3, epochs=500, lambda_sparse=0.05, finetune_epochs=200 and finetune_lambda=0.005 does not count as a method improvement.

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
