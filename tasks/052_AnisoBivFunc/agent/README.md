# Structural adaptation and rate accelerated estimation in bivariate functional data

## 1. Goal
Estimate the direction of anisotropy from independent noisy bivariate functional observations. The supplied process combines fractional Brownian motions along orthogonal directions and adds observation noise. Alpha is the angle of the direction with greater regularity, measured from the first coordinate axis. Improve the mean per-configuration median absolute angle error in this fixed coordinate.

## 2. Data
This task has no training split. `data/train/` is intentionally empty. `data/val/input/` contains a manifest and lossless binary observations: 60 configurations × 200 independent learning sets, each with N = 100 or 150 surfaces on a 51 × 51 or 101 × 101 grid. `data/val/labels.json` contains validation angles and aggregation cells for evaluation and method, hyperparameter or checkpoint selection only. Do not fit reusable models or learned preprocessing on validation. Per-learning-set estimation using only that set's observed surfaces remains permitted. Do not carry fitted state across independent learning sets.

`bench/generator.R` and its frozen source helpers define the supplied observation process. Validation uses seed0 = 2026 and original seed formula seed0 + 1000 × configuration + replication. Observation IDs, shard names, and order are randomized independently of truth. Predictor inputs contain only IDs, dimensions, storage locations, and observed float64 surfaces. They do not contain angle, noise level, regularity, generation seeds, or truth-bearing configuration identifiers. Do not re-split or relabel data or obtain external observations.

## 3. Metric and scoring
`mean_median_angle_risk` is lower-is-better: compute |alpha_hat_adj − alpha| for each replication, take the median within each of the 60 cells, then take the unweighted mean of those 60 medians. Every full cell has exactly 200 replications.

The scorer enforces complete identity coverage, unique IDs, finite angles in [0, π], and the full 60 × 200 label design; violations return 1e100. The validator also checks observation dimensions and shard extents. Data isolation and absence of hidden-label access are checked by sandboxing and patch review.

Measured validation mean is 0.027840443014127112, with standard deviation 0.0 across reference repetitions.

## 4. Baseline
The baseline uses the authors' `direg` source at upstream commit `2dec6f98f39b0e4c177a3f2043f5cfa2533c8dcf`, with vectorized kernels in `code/sim/fast_kernels.R`. It estimates directional mean-squared increments and noise, converts their ratio to candidate angles, selects the candidate with greatest estimated regularity, and applies a single correction. The calls are `estimate_angle_f`, `identify_angle_f` and `angle_correct_f`. The baseline uses a 15-point identification spacing grid and delta = 1/sqrt(side).

Training writes only a seed marker. Each prediction independently processes one learning set and writes its adjusted angle. Marker creation took 0.06 seconds in each reference repetition. Validation prediction took 1249.58, 1259.83 and 1214.58 seconds on 12 CPUs. The supplied runtime provides R, pracma and purrr; Python validation and scoring use only the standard library.

## 5. Interface (frozen)
- `bash code/train.sh --out DIR` creates the model directory and seed marker.
- `bash code/predict.sh --model DIR --input PATH --output FILE` fits all learning sets in the input directory and writes `id,alpha_hat_adj` CSV.
- `BENCH_SEED` initializes estimator RNG; the baseline estimator is deterministic for fixed observations. `BENCH_CPUS` controls bounded prediction workers. BLAS threads are one per worker.
- `BENCH_DATA_DIR` and `BENCH_BENCH_DIR` locate frozen data and tools. Full details are in `bench/io_spec.md`.
- `bench status`, `bench eval-dev`, `bench validate-output FILE`, and `bench submit` are available in the candidate workspace.
- `code/env.lock`, dependencies and the supplied runtime are immutable. No installation occurs during prediction.

## 6. Rules
Everything under `code/` is editable except the immutable dependency lock; working surface/noise models, auxiliary fitting losses, anisotropy-direction estimators and numerical methods may be replaced while preserving the original anisotropy target, observed surfaces, script arguments and prediction format. `data/`, `bench/`, `README.md`, the metric, and the protocol are frozen. The resource allocation is 12 CPUs and 16 GB memory. No network is available. Reading hidden labels, memorizing predictions, exploiting identifiers to reconstruct hidden truth, or bypassing the observation interface is disqualifying.

## 7. Known pitfalls
Validation observations occupy about 76.812 GB of float64 data. Read one learning set per worker; never load the whole split. Offsets count bytes and each surface uses R column-major order. A configuration median is not a pooled median or mean absolute error.

## Research contract

**Objective:** Estimate the anisotropy direction of the specified bivariate functional process from noisy surfaces, minimizing the fixed angle risk.

**Fixed research objects:** The anisotropy-direction parameter defined above, functional-process design, raw observed surfaces/grid and independent learning sets remain fixed. Angle coordinate in [0,pi], absolute nonwrapped angle error, median per each of 60 configurations and equal mean of medians remain fixed.

**Editable methods:** Direction estimation, working covariance/noise models, regularity estimation, smoothing, auxiliary losses and numerical optimization may change. The three baseline angle routines, spacing grid and correction computation are estimation methods rather than an alternative scientific target.

**Forbidden substitutions:** Do not replace anisotropy direction with a principal-component angle or circular-loss target unless it estimates the same parameter in the specified coordinate. No hidden angle/noise/regularity labels, configuration lookup, modified surfaces, cross-learning-set pooling or changed aggregation. When retaining the supplied three-stage estimator, changing only delta_learn=1/sqrt(side) or the 15-point identification spacing grid from 1/sqrt(side) to 0.4 does not count as a method improvement. Snapping estimated angles to the known angle set in `bench/scorer.py` or the validation labels counts as a configuration lookup.

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
