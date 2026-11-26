# ResCP: Reservoir Conformal Prediction

## Research question and experiment

Research question: around a fixed point forecaster on non-exchangeable time series, how can prediction intervals adapt to local dynamics, staying narrow while coverage stays near nominal?

The source paper proposes ResCP, which needs no training: an untrained echo state network is driven by past residuals, those residuals are resampled with weights from the similarity between their reservoir states and the query state, and quantiles of the sample give the interval. Its main table compares ResCP with SCP, NexCP, SPCI, HopCPT, CP-QRNN and ResCQR at alpha 0.1 by coverage gap, width and Winkler score. All methods calibrate the same saved residuals of the same base predictors on the same chronological segments; each tunes its own hyperparameters per cell on a validation set. Learned quantile models led on large datasets with informative exogenous variables, which are not supplied here, and fell behind on Exchange and ACEA.

This task takes the ResCP column: 12 cells, four datasets (`solar`, `beijing`, `exchange`, and `elec`, the paper's ACEA) crossed with three frozen point predictors (`rnn`, `transformer`, `arima`). A candidate is compared with ResCP on a hidden later segment.

## Metric and guards

`normalized_winkler`, lower is better. Per interval, Winkler = (upper - lower) + 20 * (distance of y outside the interval), in float32. Average over a cell's unmasked rows, divide by the cell's frozen denominator (val values are in `data/val/replay/labels.npz`), then average the 12 ratios equally.

Measured baseline on val, three seeds at 128 CPUs: mean 1.0075, seed standard deviation 0.0056; cell coverage 0.838 to 0.947 (nominal 0.90).

| Guard / rule | Enforced by |
| --- | --- |
| Each cell's coverage at least 0.86 on test and 0.83 on val; a failed cell gives Infinity | scorer |
| No repeated prefetch, missing intervals, nonfinite or reversed bounds, or incomplete execution | scorer replay |
| Artifact is a stored zip with `manifest.json` and `code/policy.sh` | validator |
| Future rows and labels not mounted, no network, one isolated worker per cell | sandbox |
| Forbidden substitutions and validation use below | patch audit |

Infinity on any seed fails the submission. Clearing the lower val floor does not show that the test floor clears.

## Interface

`train.sh` fits calibration artifacts using `data/train/`. `predict.sh` exports a private-policy-v2 artifact; the request is a descriptor only. The trusted scorer starts isolated `policy.sh` workers separately for each cell, with `BENCH_SEED` from the artifact manifest; private future histories and evaluation labels are not mounted.

The worker receives a cell descriptor, then calls `private_rpc.call('next')`. A prediction event provides `query`, `point` of shape (nodes,1) and a list `id`, plus newly available `residual`, `target` and `date` for rows from `history_start` up to but excluding `query`. Submit an object with `lower` and `upper` arrays (one value per node) using `call('submit', response)` before asking for the next event. The final `next` returns `op=done`; then write a completion file to `--output`. Current query truth is unavailable until a later query makes it past history. The full permitted history remains available; no history window is shortened by the protocol. Candidate code must preserve inherited PYTHONPATH.

## Baseline

`code/online_rescp.py` incrementally computes the original reservoir states and the released CPU sampling rule: cosine similarity to pool states, softmax at temperature T, a linear ramp over pool position (newest largest), seeded resampling, then the narrowest of 100 quantile pairs at levels (b, 0.9+b) added to the point forecast. It preserves the window+1 rule, draw counts and eta=0. The `predict` stage of `bench_main.py` is not used by the scorer. `code/method_config.json` holds the authors' per-cell values, chosen on a 10% validation tail of calibration.

## Data

`data/train/CELL.npz` holds `residual` and `target` of shape (rows, nodes, 1), `date`, `cal_index` (calibration rows), `fit_index` (scaler rows) and `encode_start`. `residual` is `target` minus the frozen forecast, NaN where unavailable. These rows are the calibration segment without its last 10%, which is val; the block that fit the base predictors is not supplied. Val is `data/val/request.json` and `data/val/replay/` (`input.npz`, `labels.npz`), 184739 intervals.

## Resources

128 CPUs and 64 GB (memory declared, not a measured peak). The default implementation uses at most 12 concurrent cells and at most four numeric threads per cell, fewer on a smaller CPU allocation. Measured baseline per seed at 128 CPUs: train about 10 s, artifact export about 5 s, val replay 397 to 425 s. The judge replays all seeds under one overall time budget, so a much slower policy risks exhausting it.

## Research scope

Only the listed components may change. Supplied observations, trusted evaluation and information boundaries remain fixed, so every method calibrates the same forecasts on the same segments.

**Objective.** Calibrate prediction intervals around the supplied frozen point forecasts for twelve time-series cells, minimizing normalized Winkler score while preserving every cell's coverage.

**Fixed research objects.** Four datasets crossed with the three frozen RNN/Transformer/ARIMA point-predictor outputs and chronological observations/splits; alpha=0.1, all twelve cells, fixed baseline normalization denominators, per-cell coverage>=0.86 on test and >=0.83 on val and original float32 Winkler arithmetic; query-before-future-observation release and independent per-cell policy state.

**Editable methods.** Conformal calibration model, residual/history features, training surrogate losses and numerical optimizers; interval construction, calibration hyperparameters and adaptation from legally preceding observations. The eta=0 condition describes the released baseline adapter, not an extra restriction on all candidate methods.

**Forbidden substitutions.** Every random draw, including reservoir weights, must derive from BENCH_SEED, and seeds or reservoir draws may not be selected by validation score (patch audit). Retraining or replacing the fixed base point predictors or altering their supplied forecast values; changing coverage/normalization/targets, future-information access or memorized intervals; training or fitting reusable model state or learned preprocessing on validation features or labels, including final train+val refits or validation-based distillation. Use only the supplied data; external copies of these series, including any copy from an installed package, are not allowed.

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
