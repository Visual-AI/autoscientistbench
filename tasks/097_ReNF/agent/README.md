# ReNF: Rethinking the Design of Neural Long-Term Time Series Forecasters

## Goal and scientific data
Forecast the next 96 hourly observations of all seven ETTh1 channels. Minimize standardized MSE. A prediction may use at most the last 720 observations of its own window; the baseline uses 512.

The supplied chronological split is fixed. Public training rows are [0,8640), and public validation rows are [8640,11520). Validation contains 2,785 horizon-96 evaluation windows. Before execution, the trusted scorer truncates each context to its last min(history_len,720) rows.

`data/train/train.csv` supplies designated training observations. `data/val/val.csv` supplies validation observations for evaluation and method, hyperparameter or checkpoint selection only.

## Private forecast protocol v2
Prediction exports a **policy artifact**. The outer prediction input is a JSON descriptor containing only the task/protocol identifier, number of windows, horizons and channel count. It contains no observations, targets, timestamps, generation seed or original row identifiers.

The trusted scorer holds the original evaluation windows and labels. It runs `code/policy.sh` in a fresh isolated process for **one window at a time**, with a read-only model and only that window's permitted observations. Workers do not share output directories, temporary files or writable state. Original row identifiers are replaced by the constant `window` inside the worker; the scorer restores alignment after inference. No complete test window batch is mounted into an outer candidate process.

- `data/val/input.json`: public validation descriptor.
- `data/val/evaluation/input.npz` and `labels.npz`: public validation observations and targets for evaluation and method, hyperparameter or checkpoint selection only.
- The hidden evaluation directory is private to the scorer and is never mounted into candidate execution.
- `bench/forecast_eval.py` performs isolated replay and aggregation; `bench/legacy_metric.py` retains the original numeric scoring implementation.
- `bench/private_eval.py` validates and runs the artifact without executing candidate code in the private scorer process.

Earlier context and padding are removed before a worker starts. Input history_len therefore describes only the allowed context; it cannot disclose the original window position.

## Metric and reference status
Minimize horizon-96 multivariate standardized MSE with features=M. Submit predictions in original data units. MSE averages squared residuals divided by the squared frozen training standard deviation over all windows, forecast steps and channels. MAE averages absolute standardized residuals and is diagnostic. seq_len must be a positive integer no greater than 720 or the supplied allowed history, and must be consistent across windows. Worker outputs must contain finite floating-point forecasts with exact one-window shapes and identifiers. Invalid artifacts or forecasts receive infinite MSE. The outer submission must be a policy artifact.

Across the three reference seeds, validation MSE has mean 0.6947337806 and population standard deviation 0.0008509897. Training took 154.03, 172.23 and 178.20 seconds. Isolated validation replay took 1350.92, 1349.30 and 1348.57 seconds, respectively, excluding artifact setup.

## Baseline
The baseline is ReNF-alpha, an MLP forecaster with reversible instance normalization and cascaded forecasts of increasing horizon length. Each stage receives the observed history and preceding forecasts. Training supervises these stages with a stage-weighted absolute-error loss; its frequency-domain term has weight `alpha_freq=0`. `train.sh` saves the EMA checkpoint selected by validation MSE and its training-fitted scaling statistics. `policy.sh` loads that checkpoint and returns the final-stage forecast in original data units.

`predict.sh` packages committed code, model files, required dependencies and explicitly permitted public resources. Candidate checkpoint code is loaded only inside the isolated worker.

## Interfaces
- `bash code/train.sh --out DIR`
- `bash code/predict.sh --model DIR --input DESCRIPTOR.json --output FILE`: export the artifact before private observations are released.
- `bash code/policy.sh --model DIR --input WINDOW.npz --output FILE`: worker-only numerical prediction entry; see `bench/io_spec.md` for the one-window NPZ format.
- `python bench/validate.py ARTIFACT DESCRIPTOR.json`: structural validation without executing candidate code.
- `python bench/scorer.py ARTIFACT EVALUATION_DIR`: trusted replay followed by the original metric.

Honor `BENCH_SEED` and `BENCH_CPUS`. Worker code/model/resources are under `/ws/code`, `/ws/model`, `/ws/resources`; `BENCH_DATA_DIR` points to the explicitly packaged public resources. Use the supplied paths rather than host paths or caches. The worker cannot access the parent scorer, its private inputs, other windows or the network.

## Rules and limitations
Model, loss and optimizer under code/ remain editable, and hyperparameters may be retuned with such a change, while honoring the maximum 720-step lookback and multivariate protocol.

Training uses the supplied training rows; validation remains available for early
stopping and model selection. The original restrictions on external data and
pretrained weights remain binding.

Everything required to execute the submitted model must be included in the artifact. Frozen data, scoring code and protocol semantics cannot be changed by candidate patches. Do not use outside data, hidden-target lookup, cross-window state, fabricated resource declarations, or sample identifiers as features.

The task uses 8 CPUs and a configured 16 GB memory allowance. The memory allowance is harness metadata. Method, training provenance and algorithm-seed compliance require review.


The supplied neural architecture and training recipe are a baseline. A replacement forecasting method, model, training surrogate, optimizer or code-internal preprocessing is allowed under the same input, data-provenance, horizon and output rules.

## Research scope

Objective: Forecast all seven ETTh1 variables for horizon 96 from the allowed history of each window.

Fixed research objects: The supplied chronological split, all supplied evaluation windows, seven channels, horizon 96 and maximum 720-step lookback remain fixed. Training-standardized multivariate MSE is the fixed objective; MAE remains diagnostic.

Editable methods: The entire forecaster, internal feature representation, training surrogate, optimizer, regularizer and training procedure may change within the specified input and data-provenance constraints. The evaluated prediction target and loss formula remain fixed.

Forbidden substitutions: Do not change the time series/targets, access earlier than the allowed history or other windows, change standardization, or substitute a different horizon/metric. When retaining the supplied ReNF training pipeline, changing only learning_rate=0.0006, gamma=0.3, StepLR step_size=2, r_ema=0.996, alpha_freq=0, seq_len=512, d_layers=3, d_ff=2048, batch_size=128, train_epochs=10 or patience=3 does not count as a method improvement.

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
