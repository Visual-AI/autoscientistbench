# 100_xPatch: private forecast protocol v2

## Outer input and output
`input.json` has exactly protocol, task_id, windows, horizons, windows_per_horizon and channels. The protocol is `private-forecast-v2`. It contains no observations or answers. `predict.sh` must produce the task-tagged policy artifact defined by `private_eval.py`; ordinary prediction arrays are rejected.

The artifact contains the candidate code (including policy.sh), the trained model, required dependency files and any explicitly allowed public resources. Package it with `forecast_export.py`. Artifact paths, members, types and size are validated without executing their contents.

## Isolated worker input and output
One NPZ record with id=["window"], horizon=[H], x of shape (1,96,7), start=[the timestamp of the first row of x], and channels, with H in {96,192,336,720}. Output exactly id_H=["window"] and pred_H of shape (1,H,7), finite floating point, in original physical units. No other horizon or record may be output.

Channel order is HUFL,HULL,MUFL,MULL,LUFL,LULL,OT. Arrays contain numeric or string values only and are loaded with allow_pickle=False. Write outputs to the exact requested path using an open binary file handle with numpy.savez.

The worker model is read-only. A fresh process and fresh writable scratch/output directories are used for every original window. The worker sees no other record. There is no online accumulation of prior test windows: each window is an independent inference problem.

## Trusted evaluation directory
The scorer receives a directory containing input.npz (every 8th original window) and labels.npz (their targets and frozen scoring metadata). The hidden directory is never mounted into the outer candidate or window worker. Validation uses the same layout in public data/val/evaluation.

## Scoring
The scorer checks each one-window result, restores original opaque IDs, assembles the original per-horizon prediction arrays and invokes legacy_metric.py. The original four-horizon equal-weight standardized MSE/MAE calculations apply, with the mandatory guard of mean MAE <= 0.4275 on test and <= 0.72 on validation. Guard failure yields Infinity.

The supplied neural architecture and training recipe are a baseline. A replacement forecasting method, model, training surrogate, optimizer or code-internal preprocessing is allowed under the same input, data-provenance, horizon and output rules. A submission whose only change is new values for existing settings of train.sh or run.py (batch size, learning rate, learning-rate schedule, epochs, patience, moving-average type, alpha, beta, RevIN, patch length, stride) does not count as a method improvement; this rests on patch audit. Switching the training loss to the unused mse_criterion, or to the commented MSE or CARD variants in exp/exp_main.py, does not on its own count as a method improvement; this rests on patch audit.

## Research scope

Forecast all seven ETTh1 variables at the four fixed horizons while satisfying the MAE guard.

Fixed research objects and target: The original split, every 8th of the first 2048 windows per horizon, seven channels and at most 96 input observations remain fixed. The equal-horizon training-standardized mean MSE and mean MAE at most 0.4275 on test and 0.72 on validation remain fixed.

Editable methods: The entire forecaster, internal feature representation, training surrogate, optimizer, regularizer and training procedure may change within the specified input and data-provenance constraints. The evaluated prediction target and the evaluation loss formula remain fixed.

Forbidden substitutions: Do not change/drop windows, use extra history or cross-window state, introduce external pretrained weights, or alter standardization, MSE or the MAE guard.

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
