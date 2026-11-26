# xPatch: Dual-Stream Time Series Forecasting with Exponential Seasonal-Trend Decomposition

## Goal and scientific data
Research question: given only the latest 96 hours of a multivariate series, how accurately can a method forecast the next 96, 192, 336 and 720 hours?

The xPatch paper (AAAI 2025) studies this in a unified long-term forecasting experiment. It compares xPatch with CARD, TimeMixer, iTransformer, RLinear, PatchTST, MICN, DLinear, TimesNet and ETSformer. All methods get lookback 96, the same horizons, the same chronological split, training-based standardization and the same MSE and MAE, so differences come from the method. Each method keeps its own configuration, such as its epochs and learning-rate schedule. Longer lookbacks are a separate experiment in the paper and are outside this task.

This task keeps that setting on ETTh1, with xPatch as the baseline. Forecast all seven ETTh1 channels from exactly 96 hourly observations at horizons 96, 192, 336 and 720. Minimize mean standardized MSE while keeping mean standardized MAE at or below 0.4275 on test and 0.72 on validation.

The ETTh1 chronological split is train [0,8640), validation [8640,11520), test [11520,14400), in hourly rows. Windows at the start of a split take their input from the rows just before it. Every 8th of the first 2048 windows at each horizon is evaluated, 256 per horizon, in validation and test. The first 2048 follow the paper's test loader (batch 2048, drop_last, no shuffle). Only every 8th is used because each window runs in a fresh process.

data/train/ETTh1_train.csv and data/val/ETTh1_val.csv hold the original train and validation rows.

## Private forecast protocol v2
The commands are train, predict, validate and scorer. Prediction exports a **policy artifact**, not a table of completed forecasts. The outer prediction input is a JSON descriptor containing only the task/protocol identifier, number of windows, horizons and channel count. It contains no observations, targets, timestamps, generation seed or original row identifiers.

The trusted scorer holds the evaluation windows and labels. It runs `code/policy.sh` in a fresh isolated process for **one window at a time**, with a read-only model and only that window's permitted observations: its 96-step `x`, its horizon, its start timestamp and the channel names. Workers do not share output directories, temporary files or writable state. Original row identifiers are replaced by the constant `window` inside the worker; the scorer restores alignment after inference. No complete test window batch is mounted into an outer candidate process.

- `data/val/input.json`: public validation descriptor.
- `data/val/evaluation/input.npz` and `labels.npz`: public validation observations and targets, for offline evaluation and selection only. No model or learned preprocessing may be fitted on them.
- The hidden evaluation directory is private to the scorer and is never mounted into candidate execution.
- `bench/forecast_eval.py` performs isolated replay and aggregation; `bench/legacy_metric.py` retains the original numeric scoring implementation.
- `bench/private_eval.py` validates and runs the artifact without executing candidate code in the private scorer process.

## Metric and reference
For each horizon the scorer standardizes predictions and labels with the fixed per-channel mean and standard deviation of the training rows, then takes MSE and MAE over all windows, steps and channels. The four horizons have equal weight. The score is the mean MSE; lower is better.

The MAE guard is mandatory: mean MAE at most 0.4275 on test and 0.72 on validation. Guard failure yields Infinity, and Infinity on any evaluation seed fails the run. Finite output values, exact one-window shapes and identifiers are checked on every worker result; a failed or invalid window also yields Infinity. A plain NPZ prediction table is not a valid outer submission. The scorer enforces all of this.

Measured baseline on validation over three seeds: MSE mean 1.18408, std 0.00094; MAE mean 0.7127. At 8 CPUs, one seed takes 465 to 751 s to train all four horizons, about 5 s to export the artifact and 485 to 553 s to replay the 1024 validation windows. Most of the replay time is worker start-up. There is no per-window time limit; replay counts against the overall evaluation budget.

## Baseline
xPatch normalizes each series with RevIN and handles every channel independently. An exponential moving average splits the input into trend and seasonal parts. A linear MLP stream takes the trend, a CNN stream over patches takes the seasonal part, and a final linear layer combines the two.

The original xPatch model, optimizer and training settings are retained: one model per horizon, batch 2048, AdamW at 0.0005, sigmoid schedule, arctangent-weighted MAE, patch length 16/stride 8, EMA alpha/beta 0.3, up to 100 epochs and patience 10. Validation loss selects the checkpoint and triggers early stopping.

`train.sh` retains the original training behavior and model layout. `policy.sh` is the original numerical prediction entry point, adapted only where needed to consume a single allowed window. `predict.sh` packages committed code, model files, required dependencies and explicitly permitted public resources (none in this task). Candidate checkpoint code is loaded only inside the isolated worker, never in the trusted scorer.

## Interfaces
- `bash code/train.sh --out DIR`
- `bash code/predict.sh --model DIR --input DESCRIPTOR.json --output FILE`: export the artifact before private observations are released.
- `bash code/policy.sh --model DIR --input WINDOW.npz --output FILE`: worker-only numerical prediction entry; see `bench/io_spec.md` for the one-window NPZ format.
- `python bench/validate.py ARTIFACT DESCRIPTOR.json`: structural validation without executing candidate code.
- `python bench/scorer.py ARTIFACT EVALUATION_DIR`: trusted replay followed by the original metric.

Honor `BENCH_SEED` and `BENCH_CPUS`. Up to 8 workers run at once, each on one CPU, so `BENCH_CPUS` is 1 inside a worker. Worker code/model/resources are under `/ws/code`, `/ws/model`, `/ws/resources`; `BENCH_DATA_DIR` points to the explicitly packaged public resources. This task packages none, so `/ws/resources` is empty and `data/` is not visible to the worker. Anything the worker needs must be in `code/` or in the model directory written by `train.sh`. Use the supplied paths rather than host paths or caches. The worker cannot access the parent scorer, its private inputs, other windows or the network. The runtime is Python 3.9 with the packages pinned in `code/env.lock`.

## Rules and limitations
The model, loss, optimizer and hyperparameters under code/ remain editable; fit only on the supplied training data and use validation only for evaluation and selection. No external pretrained weights.

The supplied neural architecture and training recipe are a baseline. A replacement forecasting method, model, training surrogate, optimizer or code-internal preprocessing is allowed under the same input, data-provenance, horizon and output rules.

A submission whose only change is new values for existing settings of train.sh or run.py (batch size, learning rate, learning-rate schedule, epochs, patience, moving-average type, alpha, beta, RevIN, patch length, stride) does not count as a method improvement; this rests on patch audit. Switching the training loss to the unused mse_criterion, or to the commented MSE or CARD variants in exp/exp_main.py, does not on its own count as a method improvement; this rests on patch audit.

Everything required to execute the submitted model must be included in the artifact. Frozen data, scoring code and protocol semantics cannot be changed by candidate patches. Do not use outside data, hidden-target lookup, cross-window state, fabricated resource declarations, or sample identifiers as features.

Enforcement: the scorer enforces the one-window input of 96 steps, worker isolation, the output checks, the metric and the MAE guard. The validator checks the artifact's paths, members, types and size, and the scorer repeats this. Training-data provenance, external weights and the use of identifiers as features rest on patch audit.

The task is limited to 8 CPUs and a configured 8 GB memory allowance. The memory allowance is metadata, not a hard limit.

## Research scope

Objective: Forecast all seven ETTh1 variables at the four fixed horizons while satisfying the MAE guard.

Fixed research objects: The original split, every 8th of the first 2048 windows per horizon, seven channels and at most 96 input observations remain fixed. The equal-horizon training-standardized mean MSE and mean MAE at most 0.4275 on test and 0.72 on validation remain fixed.

Editable methods: The entire forecaster, internal feature representation, training surrogate, optimizer, regularizer and training procedure may change within the specified input and data-provenance constraints. The evaluated prediction target and the evaluation loss formula remain fixed. New values for existing settings alone, or a switch to the existing unused losses, do not count; see Rules and limitations.

Forbidden substitutions: Do not change/drop windows, use extra history or cross-window state, introduce external pretrained weights, or alter standardization, MSE or the MAE guard. Do not train or fit reusable model state or learned preprocessing on validation features or labels, including final train+val refits or validation-based distillation.

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
