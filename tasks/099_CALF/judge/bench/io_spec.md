# 099_CALF: private forecast protocol v2

## Outer input and output
`input.json` has exactly protocol, task_id, windows, horizons, windows_per_horizon and channels. The protocol is `private-forecast-v2`. It contains no observations or answers. `predict.sh` must produce the task-tagged policy artifact defined by `private_eval.py`; ordinary prediction arrays are rejected.

The artifact contains the candidate code (including policy.sh), the trained model, required dependency files and any explicitly allowed public resources. Package it with `forecast_export.py`. Artifact paths, members, types and size are validated without executing their contents.

## Isolated worker input and output
One NPZ record with id=["window"], horizon=[H], x of shape (1,96,7), and channels, with H in {96,192,336,720}. The original per-window start timestamp is included only when present in the scientific input; this task's inputs have none. Output exactly id_H=["window"] and pred_H of shape (1,H,7), finite floating point, in original physical units. No other horizon or record may be output.

Channel order is HUFL,HULL,MUFL,MULL,LUFL,LULL,OT. Arrays contain numeric or string values only and are loaded with allow_pickle=False. Write outputs to the exact requested path using an open binary file handle with numpy.savez.

The worker model is read-only. A fresh process and fresh writable scratch/output directories are used for every original window. The worker sees no other record. There is no online accumulation of prior test windows: each window is an independent inference problem.

## Trusted evaluation directory
The scorer receives a directory containing input.npz (every 8th original window) and labels.npz (their targets and frozen scoring metadata). The hidden directory is never mounted into the outer candidate or window worker. Validation uses the same layout in public data/val/evaluation.

## Scoring
The scorer checks each one-window result, restores original opaque IDs, assembles the original per-horizon prediction arrays and invokes legacy_metric.py. The metric is the float64 training standardization and the equal-weight four-horizon MSE. MAE is diagnostic; there is no MAE guard.

The supplied CALF design and training recipe are a baseline. The forecaster must keep the supplied pre-trained GPT-2 layers as its backbone and fine-tune them for the time-series input; a method that does not use them does not answer this task's question. Around that backbone, a different alignment, token construction, adapter, fine-tuning scheme, training surrogate, optimizer or code-internal preprocessing is allowed under the same input, data-provenance, horizon and output rules. For a submission that keeps the CALF model, the supervised loss stays `--task_loss l1`: switching this existing flag does not count as a method change. For a submission that keeps the CALF model, the loss weights `--feature_w 0.01` and `--output_w 1.0`, the learning rate `--learning_rate 0.0005` and the schedule `--tmax 20` stay as supplied, so that a gain comes from the method and not from a constant. A submission whose only change is new values for other existing settings of `code/bench_run.py` or `code/run.py` (`--batch_size`, `--lradj`, `--train_epochs`, `--patience`, `--n_heads`, `--d_ff`, `--dropout`, `--gpt_layers`, `--r`, `--lora_alpha`, `--lora_dropout`, `--cos`, `--task_w`, `--feature_loss`, `--output_loss`) does not count as a method improvement.

## Research scope

Forecast all seven ETTh1 variables at the four fixed horizons from each 96-observation window.

Fixed research objects and target: The original split, every 8th of the 2881-H windows at each H in 96/192/336/720, all seven channels and 96-step context remain fixed. The float64 training standardization and equal-horizon mean MSE are fixed; MAE is diagnostic. The supplied pre-trained GPT-2 layers stay as the forecaster's backbone and are fine-tuned for the time-series input (patch audit). The supplied PCA word embeddings are a permitted method input.

Editable methods: Everything around the GPT-2 backbone may change: alignment and token construction, the fine-tuning scheme, internal feature representation, training surrogate, optimizer, regularizer and training procedure, within the specified input and data-provenance constraints and the rules on constants. The evaluated prediction target and loss formula remain fixed.

Forbidden substitutions: Do not replace or drop the GPT-2 backbone, change observations/targets, add external data or pretrained resources, use cross-window information, or alter horizons, standardization or the MSE objective.

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
