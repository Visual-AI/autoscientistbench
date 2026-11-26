# ThousandWorlds: climate emulation with GPLFR

The judge retrains your code from scratch, scores data held by the judge, and compares
with an archived baseline. The rules below are binding.

## 1. Goal
**Research question.** One global climate model (GCM) run can cost up to millions of
core-hours. Can a fast emulator predict a planet's climate from eight planetary
parameters and a simulator category, when simulations are scarce, the simulators
disagree systematically and some output fields are missing?

**Experiment.** ThousandWorlds is a benchmark for this question. Its standard Table 4
setting compares ten methods, from the training mean and kNN through trees and deep
learning to Gaussian processes, on the multi-partial, multi-complete and single-complete
subsets. All methods share the data, split and metric. The authors tune hyperparameters
per method and subset by cross-validation on planet groups inside the training pool.
GPLFR, the baseline here, ranks best. It is one method for the question, not the object
of study.

**Your task.** Predict global exoplanet climate grids from the eight planetary parameters
and the simulator category. Improve `relative_rmse` (lower is better), while meeting the
two other-subset accuracy thresholds. You may change the method freely within the
interface and data rules.

## 2. Data
- Targets are time-mean climates of tidally locked waterworlds from five GCMs: UM,
  ExoCAM, ExoCAM-pre-2022, LFRic and ExoPlaSim. Study configuration choices outside the
  eight parameters are not inputs, so they act as structured noise.
- The subsets nest. Single-complete has UM simulations only, 48 fields and nothing
  missing. Multi-complete has all five GCMs and the same 48 fields. Multi-partial is the
  full dataset: all five GCMs, 53 fields and structured missingness.
- `data/train/SUBSET/train.npz` contains raw inputs, climate grids, spherical harmonic
  coefficients and observation masks; `stats.pkl` contains training-only preprocessing
  statistics.
- `data/val/input.npz` and `data/val/labels.npz` are public validation data. Validation
  holds out 20% of unique training planet groups, fixed split seed 20261030, with a
  common group assignment across all subsets. Train/val sizes are 1220/307, 1149/294,
  and 164/39 respectively. Model selection may use this set.
- Evaluation data are held by the judge using the paper's standard split. They are
  never readable in the agent workspace.
- Grids have 32 latitudes and 64 longitudes, with 53 channels in multi-partial and 48
  in complete subsets. Surface temperature, ASR and OLR have one level; temperature,
  specific humidity, cloud fraction and the two wind components have 10 levels, or 9 in
  complete subsets. Fields are symmetrized about the equator. Missing fields have a
  whole-field mask. See `bench/io_spec.md`.
- Baseline and candidate training, including final fits, use train only. Validation is for evaluation and selection.
  No external data, corpus downloads, relabeling or modification of the frozen split.

## 3. Metric and scoring
`relative_rmse` follows the authors' float32, latitude-weighted metric. First compute
RMSE per simulation and field, then average valid simulations and levels within each
of eight variables. Normalize each variable by frozen Train-mean RMSE (the training-mean
predictor scored on the same split, stored in the labels) and compute the geometric
mean. Average the resulting scalar across five independent algorithm seeds, as Table 4
does. The prediction file therefore carries five fitted predictors, not a prediction
ensemble.

| Guard or rule | Enforcement | Outcome |
|---|---|---|
| multi-complete relative geometric RMSE ≤ 0.48 | scorer | otherwise score 1.0 |
| single-complete relative geometric RMSE ≤ 0.55 | scorer | otherwise score 1.0 |
| Exact IDs/order/shape and finite predictions | scorer and validator | Infinity / invalid output |
| Fit only on the provided training split; five independent seeded fits; no memorized evaluation data | patch audit | disqualification |

If both accuracy guards pass, the score is multi-partial relative geometric RMSE. The
guards keep the other two subsets in the comparison, and these fixed thresholds are part
of the score.
Validation baseline: **0.2501556878 ± 0.0010542923** (population SD across three seed groups).
The judge repeats training for three BENCH_SEED values and compares the mean score.
PASS requires relative improvement ≥ max(1%, 2 × baseline score standard deviation / mean).

## 4. Baseline
GPLFR from [astroautomata/ThousandWorlds](https://github.com/astroautomata/ThousandWorlds),
upstream commit `baa1dfb7f72046ee1f4eff6929535e64610e22ad`. This task's base commit is
a sanitized source snapshot. GPLFR decodes each output linearly from a low-dimensional
latent state with a GP prior over the inputs, and optimizes latents and kernel
hyperparameters jointly under a tempered MAP objective. Mathematical kernels are in
`code/thousandworlds/models/gplfr.py`, `_gplfr_core.py`, `_torch_kernels.py` and
`_gplfr_weighting.py`. Released hyperparameters are in `code/configs/`. Model files for
the paper's other methods are also in `code/thousandworlds/models/` but are not wired
into `code/main.py`.

The baseline retains float64, latent dimension 150, the released kernels, optimizer,
loss, and 115/120/30 training steps for multi-partial/multi-complete/single-complete.
`code/main.py` reads materialized grouped data with the authors' fold preprocessing on
the training groups, controls seeds and threads through the environment, saves fitted
models and has a separate point-prediction entry point. The raw-corpus downloader and
evaluation loader are not included.

Measured on 64 CPUs: train 4.83–4.99 min per seed group; validation prediction
0.22–0.24 min. Memory limit is 96 GiB.

Each run fits five independent models per subset. BENCH_SEED=1 uses algorithm seeds
0–4, BENCH_SEED=2 uses 5–9, and BENCH_SEED=3 uses 10–14. There are at most ten concurrent
workers, each using at most four CPU threads, constrained by BENCH_CPUS.

## 5. Interface (frozen)
- `bash code/train.sh --out DIR` saves everything needed for prediction.
- `bash code/predict.sh --model DIR --input PATH --output FILE` writes NPZ in the exact
  format of `bench/io_spec.md`; the output path may have no filename extension.
- Environment: BENCH_SEED, BENCH_CPUS, BENCH_DATA_DIR, BENCH_BENCH_DIR, BENCH_CODE_DIR.
- Only pinned PyPI / PyTorch CPU wheels are supported, with no editable or local installs.

## 6. Rules

The research scope below determines which components may change. Supplied data and splits, trusted tests, information boundaries and the evaluation metric remain fixed.

The climate-emulation estimator, derived representations, training surrogate losses and
optimization under `code/` may change within the research scope below. Preserve script
arguments, tensor output format (validator and scorer) and seed handling (patch audit).
`data/`, `bench/`, `README.md` and the prediction format are frozen.

The task has 64 CPUs and a 96 GiB memory limit. There is no runtime network access.
Honour BENCH_SEED in each independent fit. Do not read, reconstruct or guess evaluation
rows; do not hard-code predictions or embed results memorized from the paper (patch audit).

## 7. Known pitfalls
- Humidity is predicted in physical units and converted to dex only by the scorer.
- Missing truth fields are masked; all predicted entries must be finite.
- The five prediction slots are scored separately, then their scores are averaged.
- Row order must match input IDs exactly. Use an open binary handle with np.savez.
- `PYTHONSAFEPATH=1` requires an explicit PYTHONPATH for local imports, supplied by the
  entry scripts. Avoid nested numerical thread pools exceeding BENCH_CPUS.
- Validation follows the training pool, including its auxiliary simulators and physical
  conditions; its difficulty can differ from standard evaluation. A wide guard margin on
  validation therefore does not show the same margin on evaluation.
- Provided training rows exclude the validation planet groups. Compare to this task's
  reference rather than published results, which trained on the full pool.

## Research scope

**Objective.** Emulate the supplied climate simulator outputs from the fixed planetary parameters and simulator category, minimizing the original constrained relative climate-grid RMSE.

**Fixed research objects.** Existing climate simulations, planetary parameter meanings, simulator categories, planet-group splits and physical field grids/masks; three subset definitions, frozen training-mean normalizers, latitude/variable aggregation, two accuracy guards and five independent fitted predictors. These define the comparison: every method in the benchmark faces the same simulations, split and aggregation. Planet grouping keeps a planet simulated by several GCMs from spanning training and evaluation.

**Editable methods.** Climate emulator architecture, kernels, internal representations, training surrogate losses and optimizers, with their hyperparameters; derived input/output representations preserving the same physical-unit grid predictions.

**Forbidden substitutions.** Row IDs only label output rows; no model or preprocessing may use them as an input (patch audit). The released GPLFR options `variable_weights` and `output_coregionalization` stay at their baseline values in `code/configs/` (patch audit). Each prediction slot comes from one fit; averaging fits that differ only by seed inside a slot counts as averaging predictors (patch audit). Replacing the scientific simulator outputs or field meanings with easier climate targets; changing masks, subsets, normalizers, planet grouping or averaging predictors before the five-fit score; training or fitting reusable model state or learned preprocessing on validation features or labels, including final train+val refits or validation-based distillation. Masks, subsets and normalizers come from the judge's labels (scorer); the rest rests on patch audit.

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
