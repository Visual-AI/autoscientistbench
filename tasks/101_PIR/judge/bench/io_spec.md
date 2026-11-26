# 101_PIR: private forecast protocol v2

## Outer input and output
`input.json` has exactly protocol, task_id, windows, horizons, windows_per_horizon and channels. The protocol is `private-forecast-v2`. It contains no observations or answers. `predict.sh` must produce the task-tagged policy artifact defined by `private_eval.py`; ordinary prediction arrays are rejected.

The artifact contains the candidate code (including policy.sh), the trained model, required dependency files and any explicitly allowed public resources. Package it with `forecast_export.py`. Artifact paths, members, types and size are validated without executing their contents.

## Isolated worker input and output
One NPZ record with id=["window"], horizon=[H], x of shape (1,96,7), and channels, with H in {96,192,336,720}. The record also carries `start`, the timestamp string of the window's first input row. Output exactly id_H=["window"] and pred_H of shape (1,H,7), finite floating point, in original physical units. No other horizon or record may be output.

Channel order is HUFL,HULL,MUFL,MULL,LUFL,LULL,OT. Arrays contain numeric or string values only and are loaded with allow_pickle=False. Write outputs to the exact requested path using an open binary file handle with numpy.savez.

The worker model is read-only. A fresh process and fresh writable scratch/output directories are used for every original window. The worker sees no other record. There is no online accumulation of prior test windows: each window is an independent inference problem.

## Trusted evaluation directory
The scorer receives a directory containing input.npz (every 8th original window) and labels.npz (their targets and frozen scoring metadata). The hidden directory is never mounted into the outer candidate or window worker. Validation uses the same layout in public data/val/evaluation.

## Scoring
The scorer checks each one-window result, restores original opaque IDs, assembles the original per-horizon prediction arrays and invokes legacy_metric.py. The original float32 standardized errors, per-horizon MSE/MAE and equal-weight four-horizon mean remain unchanged. MAE remains diagnostic.

This protocol changes execution and submission format while preserving the horizons and numeric metric rules on every 8th original window.

`prediction_wall_s` includes reading input, all isolated model executions and
output assembly; `artifact_setup_s` reports new artifact setup separately.


The PatchTST backbone and its pretraining recipe are fixed research objects. Revision architecture, training surrogate and optimizer may change, while the pretrained backbone parameters and buffers remain unchanged.

## Research scope and backbone probe

Improve post-hoc revision of the fixed PatchTST forecasts for the four ETTh1 horizons. The backbone is trained in `train.sh` with the authors' recipe and is fixed from then on. The PatchTST architecture and authors pretraining recipe are fixed; the selected pretrained parameters and normalization buffers must remain unchanged during revision. The 96-step inputs, four horizons, original observations, splits and every-8th windows, float32 training standardization and equal-horizon mean MSE remain fixed. Revision retrieval history contains training-split observations/targets only.

The revision architecture, retrieval/weighting method, quality estimator, revision training surrogate, optimizer and hyperparameters may change. The revision may be replaced completely while still operating on the fixed backbone forecasts and permitted context. A pass must come from a change to the revision method: changing only existing constants or switches of the baseline revision (`--retrieval_num`, `--refine_lr`, `--refine_epochs`, `--refine_d_model`, `--refine_d_ff`, `--refine_layers`, `--including_time_features`, `--patience`) does not count (patch audit).

Do not substitute or fine-tune another backbone, change pretraining settings/initialization selection, update backbone buffers during revision, use validation/test history for retrieval, replace forecasting targets or change the metric. Revision models must take the fixed backbone forecast as an input and may not include a trainable copy of PatchTST, another model from `code/models`, or additional PatchTST seeds; those count as substituting or fine-tuning a backbone (patch audit). For horizon H, only the fixed backbone trained for H supplies the backbone forecast; forecasts of the other horizons' backbones may not be blended in (patch audit).

Run `python bench/check_model.py MODEL` in the candidate execution sandbox. The probe compares actual tensor keys, shapes, dtypes, values and normalization buffers before and after revision against the fixed PatchTST state layout. The baseline predictor also checks its requested horizon. Preserve both pretrained and post-revision artifacts. Alternative revision implementations may supply `MODEL/research_backbone.json` with `horizons` keyed by horizon, each containing `pretrained_state`, `final_state`, `pretrained_config` relative paths and optional `final_prefix`. Tensor files use safe `weights_only` loading. These exports do not force the baseline PIR revision layout.

Matching artifacts cannot prove that authors pretraining/data were used, that the checked backbone actually supplied the revised forecasts, or that retrieval history is legitimate. Those properties rest on patch audit.

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
