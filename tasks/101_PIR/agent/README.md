# PIR: Improving Time Series Forecasting via Instance-aware Post-hoc Revision

## Goal and scientific data
Research question: forecasting models that do well on average still fail on particular instances, because of distribution shift, missing data or rare patterns. Can a post-hoc step that estimates how reliable each forecast is, then revises it from context, lower the error of an already trained forecaster?

The PIR framework answers this with the formula y = y_bar + alpha * y_local + beta * y_global, where y_bar is the backbone forecast:
- Failure identification: a small network estimates an uncertainty per channel from the input, y_bar and a channel embedding. An auxiliary MAE loss trains it to predict the forecast's MSE.
- Local revision: attention across channels over embedded y_bar and a timestamp embedding gives y_local.
- Global revision: the top-K training windows by cosine similarity of instance-normalized inputs are retrieved per channel. The softmax-weighted sum of their targets gives y_global.
- alpha comes from the uncertainty; beta comes from the uncertainty and the similarities. The training loss is the forecast MSE plus the auxiliary loss.

The original experiment compares a backbone with and without PIR on the same inputs and splits. This task keeps one case, PatchTST on ETTh1, and compares your revision with the baseline PIR revision. Both revise a PatchTST trained by the same recipe.

Forecast all seven ETTh1 channels from exactly 96 historical rows at horizons 96, 192, 336 and 720, by revising the fixed PatchTST forecasts using permitted training data. Minimize mean standardized MSE.

The ETTh1 chronological split is train [0,8640), validation [8640,11520), test [11520,14400). Every 8th original window is evaluated, out of 2,785/2,689/2,545/2,161 at horizons 96/192/336/720, because each window runs in a fresh process. Validation has 349/337/319/271 windows, 1,276 in total.

data/train/series.csv and data/val/series.csv hold the public train and validation rows. Retrieval history is copied from training only into each model/hH/history/train/series.csv.

## Private forecast protocol v2
The outer commands are train, predict, validate and scorer. Prediction exports a **policy artifact**, not a table of completed forecasts. The outer prediction input is a JSON descriptor containing only the task/protocol identifier, number of windows, horizons and channel count. It contains no observations, targets, timestamps, generation seed or original row identifiers.

The trusted scorer holds the original evaluation windows and labels. It runs `code/policy.sh` in a fresh isolated process for **one window at a time**, with a read-only model and only that window's permitted observations. At 16 CPUs it runs 16 such workers at once, one CPU each. Workers do not share output directories, temporary files or writable state. Original row identifiers are replaced by the constant `window` inside the worker; the scorer restores alignment after inference. No complete test window batch is mounted into an outer candidate process.

- `data/val/input.json`: public validation descriptor.
- `data/val/evaluation/input.npz` and `labels.npz`: public validation observations and targets, for offline evaluation and selection only.
- The hidden evaluation directory is private to the scorer and is never mounted into candidate execution.
- `bench/forecast_eval.py` performs isolated replay and aggregation; `bench/legacy_metric.py` holds the numeric scoring implementation.
- `bench/private_eval.py` validates and runs the artifact without executing candidate code in the private scorer process.

The baseline carries its retrieval history in the model artifact. The worker also receives the window's `start` timestamp, which is allowed for deriving that window's calendar features; other windows and timestamps are not supplied.

## Metric and reference
The metric is the equal-weight four-horizon mean of per-horizon MSE, computed on float32 errors standardized with the training mean and std. Lower is better. MAE is diagnostic. Finite output values, exact one-window shapes and identifiers are checked on every worker result. A plain NPZ prediction table is not a valid outer submission.

Baseline on validation, three seeds at 16 CPUs: MSE mean 1.1399, std 0.0033; per horizon about 0.707/1.005/1.294/1.554. Training takes 243 to 258 s, artifact export about 3.5 s and validation replay about 875 s, so one `bench eval-dev` takes about 19 minutes. Replay costs about 11 s of one CPU per window. Most of that is process and model startup, and the baseline rebuilds its retrieval index in every worker. Time added per window is paid 1,276 times.

## Baseline
`train.sh` runs four independent horizon chains. Each trains PatchTST from scratch with the authors' pretraining recipe, seeded by `BENCH_SEED`, and then trains the PIR revision modules. No pretrained weights are supplied. Revision freezes the pretrained backbone parameters and normalization buffers, and its optimizer updates revision parameters only, as this task requires. The authors' released code also updates the backbone during revision; this task does not. Both stages use validation loss for early stopping and checkpoint selection only. Models, configuration, scaler and training-only retrieval history use the model/hH layout.

`policy.sh` is the numerical prediction entry point. The prediction driver schedules only the horizon present in its one-window input. `predict.sh` packages the code directory, model files and required dependencies. Candidate checkpoint code is loaded only inside the isolated worker, never in the trusted scorer.

## Interfaces
- `bash code/train.sh --out DIR`
- `bash code/predict.sh --model DIR --input DESCRIPTOR.json --output FILE`: export the artifact before private observations are released.
- `bash code/policy.sh --model DIR --input WINDOW.npz --output FILE`: worker-only numerical prediction entry; see `bench/io_spec.md` for the one-window NPZ format.
- `python bench/validate.py ARTIFACT DESCRIPTOR.json`: structural validation without executing candidate code.
- `python bench/scorer.py ARTIFACT EVALUATION_DIR`: trusted replay followed by the metric.

Honor `BENCH_SEED` and `BENCH_CPUS`. Worker code/model/resources are under `/ws/code`, `/ws/model`, `/ws/resources`. `BENCH_DATA_DIR` points to `/ws/resources`, which is empty in this task because the exporter packages no extra resources. Use the supplied paths rather than host paths or caches. The worker cannot access the parent scorer, its private inputs, other windows or the network.

## Rules and limitations
Everything required to execute the submitted model must be included in the artifact. Frozen data, scoring code and protocol semantics cannot be changed by candidate patches. Do not use outside data, hidden-target lookup, cross-window state, fabricated resource declarations, or sample identifiers as features.

The task is limited to 16 CPUs and a configured 16 GB memory allowance. The memory allowance is metadata, not a hard limit. The declared runtime is Python 3.10.22 with `code/env.lock`.

How rules are enforced:
- Scorer: the 96-step input shape, and each worker result's keys, shape, identifier and finite floating values. A violation scores Infinity.
- Validator: descriptor fields and artifact structure, without running candidate code.
- Worker sandbox: fresh process per window, read-only model, no network, no other windows.
- Patch audit: everything else in the research contract, including backbone provenance and use, retrieval history, seed behavior and validation use. Process isolation alone cannot establish them.

## Research contract

**Objective.** Improve post-hoc revision of the fixed PatchTST forecasts for the four ETTh1 horizons. The backbone is trained in `train.sh` with the authors' recipe and is fixed from then on.

**Fixed research objects.** These stay fixed so that a gain comes from the revision and not from a different forecaster or an easier problem.
- The PatchTST architecture and authors' pretraining recipe are fixed: seq_len 96, e_layers 3, n_heads 4, d_model 16, d_ff 128, dropout 0.1, factor 1, gelu activation; at most 100 epochs, batch 32, learning rate 0.0001, patience 3, MSE loss, lradj type1. `bench/backbone_contract.json` has the full list. The selected pretrained parameters and normalization buffers must remain unchanged during revision.
- The 96-step inputs, four horizons, original observations, splits and every-8th windows, float32 training standardization and equal-horizon mean MSE remain fixed.
- Revision retrieval history contains training-split observations/targets only.
- Supplied data/splits, evaluation metric, trusted evaluator and lawful information access.

**Editable methods.** The revision architecture, retrieval/weighting method, quality estimator, revision training surrogate, optimizer and hyperparameters may change. The revision may be replaced completely while still operating on the fixed backbone forecasts and permitted context. A pass must come from a change to the revision method: changing only existing constants or switches of the baseline revision (`--retrieval_num`, `--refine_lr`, `--refine_epochs`, `--refine_d_model`, `--refine_d_ff`, `--refine_layers`, `--including_time_features`, `--patience`) does not count (patch audit).

**Forbidden substitutions.** Do not substitute or fine-tune another backbone, change pretraining settings/initialization selection, update backbone buffers during revision, use validation/test history for retrieval, replace forecasting targets or change the metric. Revision models must take the fixed backbone forecast as an input and may not include a trainable copy of PatchTST, another model from `code/models`, or additional PatchTST seeds; those count as substituting or fine-tuning a backbone (patch audit). For horizon H, only the fixed backbone trained for H supplies the backbone forecast; forecasts of the other horizons' backbones may not be blended in (patch audit). Do not train or fit reusable model state or learned preprocessing on validation features or labels, including final train+val refits or validation-based distillation.

**Backbone self-check.** Run `python bench/check_model.py MODEL` in the candidate execution sandbox. The probe checks the pretrained configuration and compares actual tensor keys, shapes, dtypes, values and normalization buffers before and after revision against the fixed PatchTST state layout. The baseline predictor also checks its requested horizon. Preserve both pretrained and post-revision artifacts. Alternative revision implementations may supply `MODEL/research_backbone.json` with `horizons` keyed by horizon, each containing `pretrained_state`, `final_state`, `pretrained_config` relative paths and optional `final_prefix`. Tensor files use safe `weights_only` loading. These exports do not force the baseline PIR revision layout.

Matching artifacts cannot prove that authors' pretraining/data were used, that the checked backbone actually supplied the revised forecasts, or that retrieval history is legitimate. Those properties rest on patch audit.

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
