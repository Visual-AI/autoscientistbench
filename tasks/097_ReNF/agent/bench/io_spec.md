# 097_ReNF: private forecast protocol v2

## Outer input and output
`input.json` has exactly protocol, task_id, windows, horizons, windows_per_horizon and channels. The protocol is `private-forecast-v2`. It contains no observations or answers. `predict.sh` must produce the task-tagged policy artifact defined by `private_eval.py`; ordinary prediction arrays are rejected.

The artifact contains the candidate code (including policy.sh), the trained model, required dependency files and any explicitly allowed public resources. Package it with `forecast_export.py`. Artifact paths, members, types and size are validated without executing their contents.

## Isolated worker input and output
One NPZ record with id=["window"], x of shape (1,L,7), history_len=[L], and channels. L=min(original history_len,720); no earlier history is supplied. Output exactly id=["window"], pred of shape (1,96,7), scalar integer seq_len in [1,L], and scalar string features="M". seq_len must be consistent across the suite.

Channel order is HUFL,HULL,MUFL,MULL,LUFL,LULL,OT. Arrays contain numeric or string values only and are loaded with allow_pickle=False. Write outputs to the exact requested path using an open binary file handle with numpy.savez.

The worker model is read-only. A fresh process and fresh writable scratch/output directories are used for every original window. The worker sees no other record. There is no online accumulation of prior test windows: each window is an independent inference problem.

## Trusted evaluation directory
The scorer receives a directory containing input.npz (all original windows) and labels.npz (all original targets and frozen scoring metadata). The hidden directory is never mounted into the outer candidate or window worker. Validation uses the same layout in public data/val/evaluation.

## Scoring
The scorer checks each one-window result, restores original opaque IDs, assembles the per-horizon prediction arrays and invokes legacy_metric.py. Predictions must be finite floating-point arrays in original data units. Horizon-96 multivariate MSE averages squared residuals divided by the squared frozen training standard deviation over all windows, forecast steps and channels. Lower is better. MAE is the mean absolute standardized residual and is diagnostic. features must equal M. seq_len must be a positive integer no greater than 720 or the supplied allowed history and must be consistent across windows. Invalid artifacts or forecasts receive infinite MSE.

The scorer preserves the supplied evaluation observations, targets, horizons, sample counts and numeric metric rules. Validation reference measurements are listed in `../README.md`.

The supplied neural architecture and training recipe are a baseline. A replacement forecasting method, model, training surrogate, optimizer or code-internal preprocessing is allowed under the same input, data-provenance, horizon and output rules.

## Research scope

Objective: Forecast all seven ETTh1 variables for horizon 96 from the allowed history of each window.

Fixed research objects: The supplied chronological split, all supplied evaluation windows, seven channels, horizon 96 and maximum 720-step lookback remain fixed. Training-standardized multivariate MSE is the fixed objective; MAE remains diagnostic.

Editable methods: The entire forecaster, internal feature representation, training surrogate, optimizer, regularizer and training procedure may change within the specified input and data-provenance constraints. The evaluated prediction target and loss formula remain fixed.

Forbidden substitutions: Do not change the time series/targets, access earlier than the allowed history or other windows, change standardization, or substitute a different horizon/metric.

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
