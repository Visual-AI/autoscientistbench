# CALF: Aligning LLMs for Time Series Forecasting via Cross-modal Fine-Tuning

## Goal and scientific data
The CALF paper asks whether LLM-based forecasting improves when time-series tokens are aligned with the LLM's textual distribution at the input, feature and output levels. Its long-term forecasting experiment compares CALF with LLM-based, Transformer, CNN and MLP forecasters. Every method gets input length 96, fixed for a fair comparison, and is scored by MSE and MAE averaged over horizons 96, 192, 336 and 720.

This task is the ETTh1 part of that experiment, scored by MSE. Forecast all seven ETTh1 channels from 96 hourly observations for horizons 96, 192, 336 and 720. Minimize the equal-weight mean standardized MSE across these horizons. CALF is the baseline to beat. The forecaster must keep the supplied pre-trained GPT-2 layers as its backbone and fine-tune them for the time-series input; a method that does not use them does not answer this task's question.

The ETTh1 chronological split is train [0,8640), validation [8640,11520), test [11520,14400). Every 8th of the original 2881-H evaluation windows for each horizon H is evaluated, because each window runs in a fresh process. On validation that gives 1276 windows: 349, 337, 319 and 271 for H = 96, 192, 336 and 720.

data/train/ETTh1_train.csv and data/val/ETTh1_val.csv hold 8,640 training and 2,880 validation rows: a date column and the channels HUFL, HULL, MUFL, MULL, LUFL, LULL and OT. data/pretrained holds the offline GPT-2 weights and PCA word embeddings. These are generic resources, not fitted on ETTh1.

## Private forecast protocol v2
The commands are train, predict, validate and scorer. Prediction exports a **policy artifact**, not a table of completed forecasts. The outer prediction input is a JSON descriptor containing only the task/protocol identifier, number of windows (total and per horizon), horizons and channel count. It contains no observations, targets, timestamps, generation seed or original row identifiers.

The trusted scorer holds the evaluation windows and labels. It runs `code/policy.sh` in a fresh isolated process for each window, with a read-only model and only that window's permitted observations. Window inputs carry no timestamps. Up to 32 such processes run side by side, each on its own 2 CPUs. Workers do not share output directories, temporary files or writable state. Original row identifiers are replaced by the constant `window` inside the worker; the scorer restores alignment after inference. No complete test window batch is mounted into an outer candidate process.

- `data/val/input.json`: validation descriptor.
- `data/val/evaluation/input.npz` and `labels.npz`: public validation observations and targets, for evaluation and selection only.
- The hidden evaluation directory is private to the scorer and is never mounted into candidate execution.
- `bench/forecast_eval.py` performs isolated replay and aggregation; `bench/legacy_metric.py` holds the numeric metric.
- `bench/private_eval.py` validates and runs the artifact without executing candidate code in the private scorer process.

`predict.sh` packages the public pretrained resource directory with the model artifact. It does not package private test windows.

## Metric and measured baseline
The scorer standardizes predictions and labels in float64 with the frozen mean and std of the training rows, computes one MSE per horizon over all windows, steps and channels, and averages the four horizons with equal weight. Lower is better. MAE is diagnostic; there is no MAE guard. Finite output values, exact one-window shapes and identifiers are checked on every worker result (scorer); an invalid result or a failed worker scores Infinity. A plain NPZ prediction table is not a valid outer submission (validator).

Baseline validation MSE over three training seeds on 64 CPUs: mean 1.13239, std 0.00048.

Measured times on 64 CPUs decide how many full runs fit in your deadline:

- **Training takes about 50 minutes per seed** (2968, 3025 and 3213 s).
- Artifact export takes 12 to 13 s.
- Validation replay of the 1276 windows takes 335 to 351 s, plus 15 to 17 s of artifact setup. Most of this is per-window process and model start-up.

One full baseline train and validation run therefore takes about 57 minutes. Training, prediction and scoring have no time cap of their own; your work has an overall deadline and the judge's rerun has a separate overall budget.

## Baseline
CALF runs two branches over pre-trained GPT-2 layers. Each channel's 96 values become one temporal token. A cross-modal match module applies cross-attention over PCA-reduced GPT-2 word embeddings to produce aligned text tokens for the textual branch. The temporal branch is fine-tuned with LoRA. Training adds a feature regularization loss between the branches' intermediate layers and an output consistency loss to the supervised loss. Only the temporal branch predicts at inference.

Configuration: GPT-2 first six layers, width 768, LoRA rank 8/alpha 32/dropout 0.1, PCA cross-modal alignment, batch 256, Adam learning rate 0.0005, cosine schedule, up to 100 epochs and patience 10. Four validation-selected checkpoints, one per horizon, are saved by train.sh.

`train.sh` trains the four independent models at once, at 16 threads each on 64 CPUs, with no fixed core per model. `policy.sh` is the numerical prediction entry point for a single allowed window. `predict.sh` packages committed code, model files, required dependencies and explicitly permitted public resources. Candidate checkpoint code is loaded only inside the isolated worker, never in the trusted scorer.

## Interfaces
- `bash code/train.sh --out DIR`
- `bash code/predict.sh --model DIR --input DESCRIPTOR.json --output FILE`: export the artifact before private observations are released.
- `bash code/policy.sh --model DIR --input WINDOW.npz --output FILE`: worker-only numerical prediction entry; see `bench/io_spec.md` for the one-window NPZ format.
- `python bench/validate.py ARTIFACT DESCRIPTOR.json`: structural validation without executing candidate code.
- `python bench/scorer.py ARTIFACT EVALUATION_DIR`: trusted replay followed by the metric.

Honor `BENCH_SEED` and `BENCH_CPUS`. Inside a worker `BENCH_CPUS` is 2. Worker code/model/resources are under `/ws/code`, `/ws/model`, `/ws/resources`; `BENCH_DATA_DIR` points to the explicitly packaged public resources. Use the supplied paths rather than host paths or caches. The worker cannot access the parent scorer, its private inputs, other windows or the network.

## Rules and limitations
The method under code/ is editable within these rules. The forecaster must keep the supplied pre-trained GPT-2 layers as its backbone and fine-tune them for the time-series input; a method that does not use them does not answer this task's question. Around that backbone, a different alignment, token construction, adapter, fine-tuning scheme, training surrogate, optimizer or code-internal preprocessing is allowed under the same input, data-provenance, horizon and output rules.

For a submission that keeps the CALF model, the supervised loss stays `--task_loss l1`: switching this existing flag does not count as a method change. For a submission that keeps the CALF model, the loss weights `--feature_w 0.01` and `--output_w 1.0`, the learning rate `--learning_rate 0.0005` and the schedule `--tmax 20` stay as supplied, so that a gain comes from the method and not from a constant. A submission whose only change is new values for other existing settings of `code/bench_run.py` or `code/run.py` (`--batch_size`, `--lradj`, `--train_epochs`, `--patience`, `--n_heads`, `--d_ff`, `--dropout`, `--gpt_layers`, `--r`, `--lora_alpha`, `--lora_dropout`, `--cos`, `--task_w`, `--feature_loss`, `--output_loss`) does not count as a method improvement. These rules rest on patch audit.

Use only the supplied data and pretrained resources (patch audit).

Everything required to execute the submitted model must be included in the artifact. Frozen data, scoring code and protocol semantics cannot be changed by candidate patches. Do not use outside data, hidden-target lookup, cross-window state, fabricated resource declarations, or sample identifiers as features. The isolated worker blocks cross-window state, hidden targets and row identifiers at inference; everything else relies on patch audit.

The task runs on 64 CPUs with a configured 120 GB memory allowance. The memory allowance is metadata, not a hard limit. Python 3.9 and the numerical pins in `code/env.lock` are the declared environment.

## Research scope

Objective: Forecast all seven ETTh1 variables at the four fixed horizons from each 96-observation window.

Fixed research objects: The original split, every 8th of the 2881-H windows at each H in 96/192/336/720, all seven channels and 96-step context remain fixed. The float64 training standardization and equal-horizon mean MSE are fixed; MAE is diagnostic. The scorer enforces the evaluated windows, channels, context, standardization and metric. The supplied pre-trained GPT-2 layers stay as the forecaster's backbone and are fine-tuned for the time-series input (patch audit). The supplied PCA word embeddings are a permitted method input.

Editable methods: Everything around the GPT-2 backbone may change: alignment and token construction, the fine-tuning scheme, internal feature representation, training surrogate, optimizer, regularizer and training procedure, within the specified input and data-provenance constraints and the rules on constants. The evaluated prediction target and loss formula remain fixed.

Forbidden substitutions: Do not replace or drop the GPT-2 backbone, change observations/targets, add external data or pretrained resources, use cross-window information, or alter horizons, standardization or the MSE objective. Do not train or fit reusable model state or learned preprocessing on validation features or labels, including final train+val refits or validation-based distillation. External data and validation use rest on patch audit.

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
