# KAN-AD: Time Series Anomaly Detection

The judge retrains your code from scratch and compares its anomaly scores with an archived baseline.

## 1. Goal

Detect point anomalies in univariate time series from AIOPS (KPI), TODS, WSD and UCR. Improve `avg_f1e`, the event F1 averaged equally over datasets and five independent trials, while maintaining mean delay F1 of at least 0.727 on test and 0.59 on val. Higher is better. You may change the method within the rules below.

## 2. Data

- `data/train/input.npz` and `labels.npz`: 457 training series, 8,440,253 raw observations. The anomaly labels are provided, although the baseline trains by forecasting the next observed value.
- `data/val/input.npz` and `labels.npz`: the last floor(20% × original training length) observations of every original training series, totaling 2,110,003 observations. Each segment is processed separately. Validation labels are public.
- The test split is held by the judge and is unavailable in this workspace.
- Inputs are concatenated raw series with offsets and opaque curve ids. Labels are binary anomaly indicators. See `bench/io_spec.md` for the precise layout. Curve ids link the same source across training and validation; they contain no location or label information.
- Use only the supplied data. Do not change the splits or labels, download external data, or add pretrained weights from outside the task.

## 3. Metric and scoring

The frozen scorer reproduces EasyTSAD EventF1PA in squeeze mode and PointKthF1PA with k=5, including post-event margins, stable tie handling and skipping all-normal curves. Event F1 replaces each adjusted anomaly interval with one positive item scored by its maximum anomaly score. Each normal point remains a negative item. Delay F1 uses the maximum score within the first k points of each interval and weights that interval by its length. Each metric uses its best F1 over the frozen threshold sweep. The scorer aligns scores with observations starting at index 96. Scores are averaged over curves within a dataset, then over the four datasets and five independent trials. The reported `avg_f1e` equals mean event F1 when mean delay F1 is at least 0.727 on test or 0.59 on val, otherwise **0**, as specified by the scoring rule. Invalid output returns **Infinity** and fails validation.

Threshold sweeps belong to the frozen evaluation metric. The predictor submits continuous anomaly scores and has no access to evaluation labels. Any model, hyperparameter or deployment-threshold selection must use only the supplied training/validation data.

| Constraint | Enforcement class |
|---|---|
| Five-trial mean delay F1 ≥ 0.727 on test, ≥ 0.59 on val | scorer; otherwise metric is zero |
| Exact curve ids, five complete trial rows, correct lengths, finite floating scores | scorer pipeline and validator; violations return Infinity |
| Fit only on supplied training data; validation is for evaluation/selection; no memorized hidden results | patch audit |
| Five independent fits controlled by the supplied seed; no duplicating one fit to claim five trials | patch audit; tensor shape alone cannot establish independence |

UCR validation contains no anomalies, so its event and delay F1 are undefined under the original evaluator. For development scoring, an empty dataset contributes zero to the fixed four-dataset average and appears in `empty_datasets`. Use `avg_f1e_unguarded`, `avg_f1d` and `per_dataset` diagnostics when the gate returns zero. This is an explicit development convention; do not invent anomaly labels or replace the validation split.

Measured validation `avg_f1e`: **0.632079595**, σ = **0.000865835**. The baseline passes the validation delay-F1 guard for every external seed. Ungated validation F1e equals `avg_f1e`; mean validation delay F1 is **0.593075610**. Delta is **0.20%**, calibrated from measured seed variability. The judge uses three external seeds, with five internal trials per seed. PASS requires relative improvement of at least `max(delta, 2 × reference standard deviation / reference mean)`.

## 4. Baseline

Official code: https://github.com/CSTCloudOps/KAN-AD, upstream commit `7c3a0d96b77e0d609660a240c2b11db084ba523f`. `code/kanad/kanad.py` contains the original model, loss and training loop. The active entry-point configuration is `PARAMS` in `code/bench_main.py`, matching the upstream configuration.

The baseline has a 96-point window, Fourier order 2, batch size 1024, MSE training loss, Adam learning rate 0.01, StepLR step 5 and factor 0.75, up to 100 epochs, and validation early stopping with patience 3 and delta 0.0001. It scores the final early-stopped weights, preserving the released behavior. The `torchinfo.summary` forward pass is retained because it affects RNG and batch normalization. Each split is differenced separately, then standardized using only training statistics. Anomaly scores are absolute one-step forecast errors.

Protocol adaptation supplies isolated data entry, explicit checkpoint/scaler saving, separate prediction, opaque ids, environment-controlled seeds and process counts. The required EasyTSAD dependency classes are vendored without changing their scientific operations. Progress display is disabled. The model and training operations are unchanged.

The declared budget is **64 CPUs and 128 GB memory**. Memory is a budget, not a measured peak. Measured train time: **14.01–14.16 min** per external seed; validation prediction: **15.7–16.5 s**. Each external seed includes all five internal trials.

## 5. Interface (frozen)

- `bash code/train.sh --out DIR`: train five models per curve and save everything needed for prediction.
- `bash code/predict.sh --model DIR --input PATH --output FILE`: write the five-trial NPZ score tensor specified in `bench/io_spec.md`.
- Environment: `BENCH_SEED`, `BENCH_CPUS`, `BENCH_DATA_DIR`, `BENCH_BENCH_DIR`, `BENCH_CODE_DIR`. Honor seed and CPU budget. All argument paths may be absolute.
- Commands: `bench status`, `bench eval-dev --seed S`, `bench validate-output FILE`, `bench submit --final -m MESSAGE`.
- Dependencies are pinned in `code/env.lock`. Python 3.11, torch 2.7.0+cpu and NumPy 1.26.4 are used. Only PyPI / PyTorch CPU `name==version` pins are rebuildable; submission freezes the installed environment.

## 6. Rules

The research scope below determines which components may change. Supplied data and splits, trusted tests, information boundaries and the evaluation metric remain fixed.


- You may change the anomaly estimator, derived representations, training surrogate losses and optimization under `code/`, within the fixed anomaly-detection target and evaluation protocol.
- `data/`, `bench/`, this README and the prediction format are frozen.
- The CPU budget is 64 and the memory budget is 128 GB.
- Every external seed must independently generate five internal trials. Honor `BENCH_SEED`; seed variance affects PASS. Do not substitute repeated copies of one fitted trial.
- Train and predict have no network access. Save all prediction dependencies into the model directory or committed code.
- Reading or guessing hidden labels, embedding memorized paper results, hard-coding predictions, and using external evaluation data are disqualifying.

## 7. Known pitfalls

- NPZ paths may have no extension. Pass an open file handle to NumPy when writing predictions.
- An input series includes observed values at scored timestamps: the target is the anomaly indicator, not a future value. Do not submit raw value forecasts in place of anomaly scores.
- The initial 96 points of each segment provide context and have no submitted scores. Do not prepend training history to validation segments.
- Each one-thread worker handles one independent curve/trial; total concurrent workers must not exceed `BENCH_CPUS`. Large thread pools inside every worker can make the run much slower.
- `PYTHONSAFEPATH=1` is enabled; entry scripts explicitly set `PYTHONPATH` for local imports.
- Preserve finite outputs even for extreme values. All five trial rows must cover every supplied curve, including all-normal curves; only the scorer decides which curves to exclude from F1 averages.
- The baseline passes the validation gate. One dataset has no validation events; inspect the per-dataset diagnostics.

## Research scope

**Objective.** Detect anomalies in the four supplied univariate time-series datasets, maximizing five-trial dataset-macro event F1 while retaining the original delay-F1 guard.

**Fixed research objects.** raw series, split-local segments, anomaly labels and four-dataset population; 96-point score alignment/context, continuous anomaly-score outputs, original EasyTSAD margins/ties/threshold-sweep rules and independent five-fit aggregation.

**Editable methods.** anomaly-detection architectures, forecasting/reconstruction/discriminative estimators, derived features and training surrogate losses; optimization on the training data and model selection using validation.

**Forbidden substitutions.** replacing the anomaly target with value forecasting, changing event/point labels, score alignment or dataset denominator; using evaluator threshold sweeps for candidate test-label selection, prepending another split history or duplicating fitted trials. When retaining the supplied KAN-AD forecasting pipeline, changing only order=2, batch_size=1024, epochs=100, lr=0.01, StepLR step_size=5 or gamma=0.75, or early-stopping patience=3 or delta=0.0001 does not count as a method improvement.

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
