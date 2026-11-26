# BRIDGE: Predicting Human Task Completion Time From Model Performance

The judge retrains your code from scratch with its own seeds and compares hidden evaluation performance against an archived baseline.

## 1. Goal

Predict human task completion time from sparse binary model success/failure observations. BRIDGE jointly estimates task difficulty and model ability with a hierarchical two-parameter logistic IRT model, then calibrates difficulty to log human minutes using METR tasks. Improve `bucket_accuracy` (higher is better), subject to the quality thresholds below. Evaluation tests transfer to a different benchmark distribution; validation is held out from METR and is not an estimate of in-distribution evaluation performance. You may change the method freely within the frozen protocol.

## 2. Data

- `data/train/input.csv` and `labels.csv`: 142 calibration tasks from 40 task families. Inputs contain `id,responses`; labels contain `id,y`, where y is human minutes.
- `data/train/auxiliary.csv`: 349 unlabeled context tasks from the authors' other benchmarks, with the input schema and a separate ID namespace. No auxiliary human times are provided.
- `data/val/input.csv` and `labels.csv`: 28 tasks from 10 disjoint families, created by holding out 20% of METR task families with seed 20261030. No family overlaps training.
- Hidden evaluation inputs, task identities and human labels are unavailable in this workspace.
- A response cell is `[subject,block,rank,binary_response]`; absence means missing, not failure. Subject is a stable model index; block/rank preserve within-subject observation order, and must not be used as predictive features. IDs are source row positions and are identifiers only. See `bench/io_spec.md`.
- Both baseline and candidates fit reusable model state on training data only, including final models. Validation is for evaluation and model selection. Do not change the frozen split files, relabel examples, or add outside data. The task-defined transductive inference on the current unlabeled prediction batch is allowed only for that batch; do not retain its fitted state for later batches or use validation labels to fit it.

## 3. Metric and scoring

`bucket_accuracy` is four-bucket accuracy in percent, provided linear-weighted Cohen kappa is at least 0.22 and macro F1 is at least 0.27. Otherwise it is **0**. The buckets in minutes are [0,15], (15,60], (60,240], (240,infinity); boundaries follow the authors' right-closed binning. Macro F1 includes all four classes with zero division set to zero. Scoring prints raw accuracy, F1 and kappa as diagnostics.

| Guard | Enforcement | Consequence |
| --- | --- | --- |
| Linear weighted kappa ≥ 0.22 | scorer | Score 0 if below threshold |
| Four-class macro F1 ≥ 0.27 | scorer | Score 0 if below threshold |
| Exact prediction contract, finite nonnegative minutes, complete ordered IDs | scorer and validator | Invalid; scorer emits Infinity |
| Allowed data only; no hard-coded truth, identity lookup or cross-call retention | patch audit | Protocol violation |

There are no model-class or parameter-count constraints and no additional model certificate is required.
Baseline validation score: **55.952381**, population σ **8.417938**, over three seeds. The judge compares the hidden-score mean to its archived reference. PASS requires relative improvement at least `max(delta, 2 × reference_sigma / abs(reference_mean))`; delta is 2.00%. `bench status` reports the configured baseline.

## 4. Baseline

Official code: https://github.com/McGill-NLP/BRIDGE, upstream commit `6a1b821bb75e27b21f71982af609401e6ec74aef`. The model is in `two_param_logistic.py`; `fit_irt.py` runs py-irt SVI for 1000 epochs with hierarchical priors, Adam learning rate 0.1 and exponential decay 0.9999. Parameters are selected by the best training ELBO. `bridge_entry.py` fits `log(minutes) = slope * difficulty + intercept` using training human times and SciPy linear regression.

The protocol adapter `bridge_entry.py` reads materialized data and honors environment seeds and threads. Training saves its context, fitted parameters and calibration into the model directory. Prediction refits the same joint IRT on saved training context plus the current unlabeled input batch, then recalibrates using saved training labels. This is the authors' transductive method; its prediction stage has fitting cost. No released fitted models, results or query labels are supplied.

Measured on 8 CPUs: training mean 0.295 min (range 17.09–18.89 s), validation prediction mean 0.283 min (range 16.79–17.24 s). Memory allowance is 8 GB; a peak-memory measurement was not collected.

## 5. Interface (frozen)

- `bash code/train.sh --out DIR` writes all state needed for prediction under DIR.
- `bash code/predict.sh --model DIR --input PATH --output FILE` writes exactly one CSV table with header `id,pred`, in numeric ID order. `pred` is minutes, not a bucket label. The output filename may have no extension.
- Honor `BENCH_SEED`, `BENCH_CPUS`, `BENCH_DATA_DIR`, `BENCH_BENCH_DIR` and `BENCH_CODE_DIR`. Paths passed by the judge are absolute.
- `bench status`, `bench eval-dev --seed 1`, `bench validate-output FILE`, and `bench submit --final -m MESSAGE` are the development interfaces. Evaluation runs train → predict → validate → score.
- Install dependencies using `uv pip install` in the supplied environment; submission freezes them into `code/env.lock`. Only pinned packages from PyPI or the PyTorch CPU index are rebuildable. Python is 3.11; editable, git and local-file installations are unsupported.

## 6. Rules

The research scope below determines which components may change. Supplied data and splits, trusted tests, information boundaries and the evaluation metric remain fixed.


- Editable surface: human-time estimators, representations, calibration, training surrogate losses and optimization under `code/`, within the fixed transfer-prediction problem.
- Frozen surface: `data/`, `bench/`, `README.md`, the prediction contract and the script arguments.
- Resource allowance: 8 CPUs and 8 GB memory. No network during either stage.
- Three training seeds are used; every stochastic component must honor `BENCH_SEED`. Seed variance affects the PASS threshold.
- Never reconstruct or memorize hidden examples or public benchmark time tables, use IDs/order metadata to retrieve labels, embed paper results, or use outside data. Prediction may fit the current batch jointly but must not retain inputs or fitted state across prediction calls. Prediction may use only the supplied current unlabeled input and designated training context. Validation is only for evaluation and method, hyperparameter or checkpoint selection.

## 7. Known pitfalls

- Ordinary inductive lookup of a previously fitted item difficulty does not handle new tasks. The baseline deliberately refits inside predict; account for that cost when changing the implementation.
- Preserve missing observations and the authors' aggregation semantics. Zero-filling the sparse response matrix changes the data.
- Numeric ID spaces overlap between distinct source tables. Keep separate namespaces when combining calibration, auxiliary and query items; never accidentally join query IDs to calibration labels.
- Values exactly 15, 60 and 240 minutes belong to the lower interval. Finite nonnegative outputs and strict numeric ID order are required.
- The evaluation distribution differs from METR validation. Inspect raw accuracy, F1 and kappa, and run multiple seeds before trusting improvements; a quality threshold failure makes the score zero.
- `PYTHONSAFEPATH=1` removes the script directory from automatic imports. The provided shell wrappers export the code directory in PYTHONPATH; keep equivalent behavior.
- Data and benchmark helpers are read-only. Write model state under `--out`, output at `--output`, and temporary files under TMPDIR. `env.lock` must remain tracked despite the original repository's ignore rules.

## Research scope

**Objective.** Predict human task completion minutes from the supplied sparse model success/failure observations and transfer calibration, maximizing the original four-bucket accuracy with kappa/F1 guards.

**Fixed research objects.** supplied binary response observations and missingness, labeled calibration tasks, unlabeled auxiliary tasks and family-disjoint validation split; human-time target and transfer setting, right-closed four-bucket definition, weighted-kappa/F1 constraints and no cross-call retention.

**Editable methods.** completion-time estimators, task/model representations, calibration functions and training surrogate losses; optimization and joint fitting on the current unlabeled batch within the allowed transductive protocol.

**Forbidden substitutions.** replacing the human-time target with model success accuracy or alternate time buckets; filling missing responses as observed failures, using source identifiers/order as time features, or retaining hidden batches between calls. When retaining the supplied IRT pipeline, changing only epochs=1000 or priors=hierarchical does not count as a method improvement.

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
