# CLEAR: Calibrated Learning for Epistemic and Aleatoric Risk

The judge retrains your code from scratch and compares its interval predictions with the archived baseline.

## 1. Goal

Regression intervals usually reflect one kind of uncertainty: aleatoric (noise in the target given the inputs, estimated by quantile regression) or epistemic (limited data and model choice, estimated by an ensemble). The source paper asks whether calibrating the two together gives narrower intervals while keeping nominal coverage. CLEAR builds `[f - g1*a_lo - g2*e_lo, f + g1*a_hi + g2*e_hi]` around a central estimate `f`. Here `a_lo`, `a_hi` are aleatoric half-widths, `e_lo`, `e_hi` are epistemic half-widths and `g2 = lambda * g1`. Lambda minimizes quantile loss on held-out rows; `g1` comes from conformal calibration at each lambda.

The paper's Figure 3 compares CLEAR variant (a) with three calibrated alternatives at 95% nominal coverage over 17 real datasets: PCS (epistemic only), ALEATORIC (bootstrapped CQR) and ALEATORIC-R (bootstrapped CQR on PCS residuals). All share one central estimate, so differences come from the uncertainty method. Their NCIW is on average 17.5%, 28.3% and 3% above CLEAR (a).

This task compares your method with a CLEAR standard variant (a) baseline on one fixed partition. Predict calibrated regression intervals with target coverage 95% for the 17 datasets. Improve `nciw` (lower is better) while satisfying the coverage guard on every dataset. You may change the uncertainty method under the rules below; the central estimate stays and the CLEAR form is not required.

## 2. Data

`data/train/train.npz` contains `x_DATASET` feature matrices and `y_DATASET` targets. `data/val/input.npz` contains `x_DATASET` and opaque `id_DATASET`; matching public targets are in `data/val/labels.npz`. Arrays retain the released preprocessing (categorical features one-hot encoded, nothing further), feature ordering and target units. NPZ archives contain numeric/string arrays without pickle objects.

| Dataset | Training rows | Validation rows | Features |
|---|---:|---:|---:|
| computer | 4914 | 1639 | 21 |
| ailerons | 8250 | 2750 | 40 |
| airfoil | 901 | 301 | 5 |
| ca_housing | 12384 | 4128 | 8 |
| concrete | 608 | 203 | 8 |
| elevator | 9959 | 3320 | 18 |
| energy_efficiency | 460 | 154 | 10 |
| insurance | 802 | 268 | 8 |
| kin8nm | 4914 | 1639 | 8 |
| miami_housing | 8358 | 2787 | 28 |
| naval_propulsion | 7160 | 2387 | 24 |
| parkinsons | 3525 | 1175 | 18 |
| powerplant | 5732 | 1912 | 4 |
| sulfur | 6048 | 2016 | 5 |
| superconductor | 12737 | 4250 | 79 |
| qsar | 3419 | 1141 | 500 |
| allstate | 3000 | 1000 | 1037 |

The test split is held by the judge. The paper evaluates ten random partitions. This benchmark uses one fixed partition across ten algorithm repetitions. The public data do not overlap the held-out partition. Exact duplicate observations were excluded from the public partitions to prevent target disclosure. Use the supplied partitions; do not reconstruct, re-split or augment them with external datasets.

External validation is for evaluation and method selection only. In the paper's standard configuration the validation split also serves as the calibration set; here that role moves inside training. The baseline reserves 20% of the supplied training rows internally for PCS selection and CLEAR calibration, using RandomState seed 20261102 and preserving row order within both subsets. Model fitting uses the remaining 80%; all fitted state therefore depends only on the original training partition. Supplied train/validation/test files remain unchanged.

## 3. Metric and scoring

`bench/scorer.py` is the exact definition. For one dataset and repetition, each row has the factor that scales its interval side about the submitted `pred` until it reaches the target: `(pred - y) / (pred - lower)` for a target below `pred`, `(y - pred) / (upper - pred)` for a target above. The scoring factor is the smallest of these values that covers 95% of rows (the paper's empirical 95% rescaling). NCIW is that factor times the mean original width `upper - lower`, divided by the target range `max(y) - min(y)`. Rescaling is part of scoring only. For every dataset, the scorer averages ten repetition metrics; the final metric is the equal-weight mean over 17 datasets.

As the paper notes, NCIW measures relative uncertainty and ignores marginal coverage. Here coverage is checked by the guard below. The paper also reports quantile loss; it is not scored here.

| Guard | Enforcement class |
|---|---|
| Mean original-interval PICP across ten repetitions is at least 0.87 on test and 0.90 on val, and at most 0.99, for every dataset | scorer |
| A row whose target lies beyond a zero-width side of its interval cannot be covered by rescaling, so its scaling factor is infinite | scorer |
| Exact id set, one row per id/repetition, finite values and lower <= central prediction <= upper | scorer pipeline through validate.py |
| Use only supplied public data; no hidden targets, memorized answers or external original datasets, including copies from an installed package | patch audit |

Any coverage violation returns `Infinity`. PICP is measured before the NCIW scoring rescaling. The floors sit below the 95% target and only catch clear coverage failure. The ceiling is there because multiplying all half-widths by one constant does not change NCIW.

The judge uses three independent training seeds; each seed trains ten algorithm repetitions. PASS requires relative improvement at least `max(delta, 2 * reference standard deviation / reference mean)`. The configured delta is **0.2%**, also reported through `bench status`.

Baseline on val over three seeds: mean NCIW 0.20758, std 0.00030. Its lowest per-dataset mean PICP on val is 0.9089 and its highest is 0.9834.

## 4. Baseline

Official code: https://github.com/Unco3892/clear at commit `9b3c0ea3de76358906dfc4448646b4a9c432f632`. The local baseline snapshot retains the relevant official source, with original data, fitted models, result artifacts and their Git history excluded. CLEAR is in `code/src/clear/clear.py`; PCS is in `code/PCS_UQ/src/PCS/regression/`; the benchmark execution entry point is `code/bench_main.py`.

The baseline selects one of QRF, quantile XGBoost and ExpectileGAM using R² on the internal training-derived calibration subset. QRF and XGBoost use 100 trees and minimum leaf/child weight 10. GAM uses 10 splines and the original smoothing search with its 300 s limit; where the search exceeds the limit, as on allstate and qsar, ExpectileGAM is not a candidate. Each of ten repetitions has 100 PCS bootstraps and 100 residual aleatoric bootstraps; CLEAR selects lambda from the original 4,011-value grid and calibrates gamma (`g1`) on that same internal training-derived subset. Residual predictions include the PCS central estimate as a feature. The authors' expectile-as-quantile GAM implementation is preserved.

The declared budget is 128 CPUs and 192 GB RAM; RAM is a budget, not a measured peak. Measured at 128 CPUs: training 24.4 to 24.9 minutes per seed, val prediction 21.5 to 23.2 seconds.

## 5. Interface (frozen)

- `bash code/train.sh --out DIR`: train and save all prediction artifacts inside DIR.
- `bash code/predict.sh --model DIR --input PATH --output FILE`: emit the CSV described in `bench/io_spec.md` at exactly FILE.
- Environment: honor `BENCH_SEED`, `BENCH_CPUS`, `BENCH_DATA_DIR`, `BENCH_BENCH_DIR` and `BENCH_CODE_DIR`. Paths may be absolute.
- `bench eval-dev --seed S` chooses the development seed; `bench validate-output FILE` checks a CSV.
- Dependencies use the pinned Python 3.10.22 environment and `code/env.lock`. Only PyPI and the PyTorch CPU index are available for rebuilds.

## 6. Rules

- `code/` is editable within the research contract below, including the uncertainty models, losses, features, hyperparameters and training procedure, provided interfaces and output format remain valid.
- Data, scoring/validation code, this README and the I/O format are frozen.
- The judge runs three seeds and compares the seed mean with the reference. Each run must honor `BENCH_SEED`.
- The CPU budget is 128, memory budget 192 GB. Train/predict have no network access.
- Hidden target access, predicting from memorized results, attempting to recover the original full datasets or partition, and hard-coded output tables are disqualifying (patch audit).
- Public validation labels may guide evaluation and method selection, but must not fit calibration parameters (patch audit). Allocate any calibration data within training.

## 7. Known pitfalls

- Output is CSV even though inputs are NPZ. The first column combines an opaque row id with `_r0` through `_r9`; all ten repetitions are required.
- Preserve the central estimate in addition to bounds: NCIW handles asymmetric intervals about that estimate. The coverage guard checks the unscaled bounds.
- Smaller intervals are not sufficient: every dataset must meet the coverage threshold.
- Parallel bootstrap workers share the CPU budget. The baseline uses one numerical thread per worker, caps nested joblib parallelism, and preserves the reproduction's NumPy CPU dispatch settings.
- Both scripts set PYTHONPATH explicitly because the judge enables PYTHONSAFEPATH. Only `code/` is rebuilt, from the submitted patch; save fitted artifacts into the supplied model directory.
- The baseline saves many estimators to support independent prediction. Use node-local temporary files through TMPDIR for transient buffers.

## Research contract

**Objective.** Produce narrow calibrated regression intervals at fixed 95% target coverage on the 17 supplied datasets.

**Fixed research objects.** Scores compare with the baseline only while these stay the same.
- The datasets, partitions, regression targets/units and ten-repetition evaluation remain fixed.
- The NCIW formula/rescaling and per-dataset PICP at least 0.87 on test and 0.90 on val and at most 0.99 remain fixed (scorer).

**Editable methods.** The interval estimator, calibration method, internal features, training surrogate and optimizer of the uncertainty method may change using allowed training/calibration inputs; the regression model behind `pred` may not.

**Forbidden substitutions.** The central estimate `pred` must remain the baseline's PCS median: the model pool, selection and bootstrap ensemble that produce it stay fixed, and only the uncertainty estimators and their calibration may change (patch audit).
- Do not change coverage targets, data, splits, target units, width formula or coverage guard; do not calibrate on hidden labels (frozen scorer and data; patch audit).
- Do not train or fit reusable model state or learned preprocessing on validation features or labels, including final train+val refits or validation-based distillation (patch audit).

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
