# Spatialproteomics: an interoperable toolbox for analyzing highly multiplexed fluorescence image data

## 1. Goal
Predict seven major cell types from multiplexed fluorescence images of B cell non-Hodgkin lymphoma tissue cores, using the supplied cell segmentation. Improve `pooled_kappa` (higher is better) against a three-annotator majority consensus. The baseline uses the paper's automatic default-quantile thresholding and argmax pipeline without human threshold review. You may change the method within §6.

## 2. Data
- `data/train/` contains a README only; the baseline requires no training.
- `data/val/input/` contains development cores as lossless NPZ files. `data/val/labels.csv` supplies public consensus cell labels. There are 20 development cores, 267,883 QC cells and 2.590 GB of input images.
- Each core contains all 56 raw fluorescence channels, the published cellpose segmentation mask, channel names, pixel coordinates, and a fixed QC cell set. See `bench/io_spec.md` for shapes and encoding.
- The patient-group split is fixed. Input IDs do not reveal source or patient identity.
- No reviewed thresholds, precomputed expression matrices or deposited cell labels are prediction inputs. Use public development labels for analysis and method selection. Do not re-split, relabel or supplement the fixed data with external samples, annotations or thresholds.

## 3. Metric and scoring
`pooled_kappa` is sklearn Cohen's kappa between predictions and majority consensus, pooled over every fixed QC cell. The seven classes are B cell, T cell, Myeloid cell, Dendritic cell, Macrophage, Stromal cell and Endothelial cell. If all three annotators disagree, Annotator 1 wins. This is a pooled cell metric, not a mean of per-core kappas. The upper bound is 1.

Every missing or out-of-class label counts as a mismatch. Invalid values on the two sides receive distinct sentinels; they can never match each other. Output one row per QC cell; use `-1` or an empty label to abstain. The direct scorer also fills omitted predictions with mismatches, so omitting difficult cells cannot reduce the scoring population.

| Guard | Enforcement class | Effect |
| --- | --- | --- |
| Fixed cell population; valid CSV; no duplicate or extra IDs; no NaN/Inf tokens | scorer, also validator | Malformed output fails; direct scoring returns `Infinity`. Validator requires a complete ID set. |
| No prediction lookup tables keyed by opaque IDs; renaming input IDs must preserve deterministic predictions | patch audit | Violations disqualify the submission; stochastic methods require seed-aware review. |
| Use only permitted inputs; no hidden annotations, source-specific answer tables or memorized paper outputs | patch audit | Disqualification; output alone cannot establish compliance. |

Baseline development `pooled_kappa` is **0.97652847913449**, with seed standard deviation **0** (deterministic). The judge compares its mean score against the archived baseline; PASS requires relative improvement of at least 0.0001 (0.01%). Equal scores are ties. The deterministic baseline is evaluated with one seed; its seed standard deviation is zero. The scorer alone does not enforce patch-audit rules.

## 4. Baseline
Official code: https://github.com/sagar87/spatialproteomics, upstream commit `2a8ff158460bc5c58d0da227407740d68b0c7e41` (0.9.0). The local base is a sanitized source snapshot; bundled data, notebook outputs, figures and upstream history were omitted. All retained official algorithm files are unchanged.

`code/bench_predict.py` calls the released preprocessing and label accessors: float32 image → observations → `remove_outlying_cells()` → quantile threshold → `intensity_mean` → arcsinh (cofactor 5) → `predict_cell_types_argmax`. Default quantiles are PAX5 0.8, CD3 0.7, CD11b 0.8, CD11c/CD68 0.9, and CD90/Podoplanin/CD31/CD34 0.95. Marker order and class mapping are preserved.

The baseline uses no-op training, NPZ input, CSV output and pinned dependencies. Four independent single-thread workers run within the 8 CPU allowance. At 8 CPUs, measured no-op training takes 0.06 s and validation prediction takes 63.67 s. Memory allowance is 40 GB; worker peaks are logged during prediction, not a measured whole-process-tree bound.

## 5. Interface (frozen)
- `bash code/train.sh --out DIR` creates the model directory; the baseline writes `NO_TRAINING`. Any learned state you introduce must be saved in DIR.
- `bash code/predict.sh --model DIR --input PATH --output FILE` reads an arbitrary number of core NPZ files and writes `id,label` CSV at exactly FILE, including when FILE has no extension.
- Environment: `BENCH_SEED`, `BENCH_CPUS`, `BENCH_DATA_DIR`, `BENCH_BENCH_DIR`, `BENCH_CODE_DIR`. Honor seed and thread limits whenever adding randomness or parallelism.
- Use `bench status`, `bench eval-dev [--seed S]`, `bench validate-output FILE`, and `bench submit --final` for budgeting, development evaluation, format checking and submission.
- Dependencies use Python 3.12 and PyPI. `uv pip install` may update the environment; submission freezes `code/env.lock`. Only pinned `name==version` packages are rebuildable; editable, Git and local-path dependencies are unsupported.

## 6. Rules

The research scope below determines which components may change. Supplied data and splits, trusted tests, information boundaries and the evaluation metric remain fixed.

- Editable surface: cell-type classification features, threshold-selection methods and learned estimators under `code/`, within the fixed cell population and consensus-label research scope below.
- Frozen surface: `data/`, `bench/`, this README, the prediction format, script signatures, segmentation and fixed QC scoring population.
- Resource allowance: up to 8 CPUs and 40 GB memory.
- Train and predict run without network. All required state must be reproducible from the permitted data, submitted code and pinned packages.
- The judge uses one seed because the baseline is deterministic; stochastic submissions must honor `BENCH_SEED`.
- Reading or inferring hidden annotations, hard-coding predictions, embedding data or label lookup tables, memorizing source-specific thresholds or paper results, and changing the population being scored are disqualifying. All pixels and cells of a prediction input may inform its predictions; opaque identifiers are correspondence keys, not predictive features.

## 7. Known pitfalls
- The input images and float32 intermediates are large. Four baseline workers can use tens of GB; increasing workers may exhaust memory.
- Quantile thresholds are computed from each complete image, before per-cell means. This order describes the baseline; alternative feature pipelines are permitted.
- The raw mask includes cells subsequently removed by QC. Emit exactly the supplied `qc_cells`; do not add excluded cells or silently skip difficult cells.
- Prediction class IDs are the fixed 0–6 encoding in `bench/io_spec.md`, not the accessor's internal numeric label IDs. Convert via its label-name mapping.
- `PYTHONSAFEPATH=1` is set; preserve the explicit local `PYTHONPATH` export. Ensure all files needed at runtime are committed under `code/`.
- Missing and out-of-class labels count as errors. NaN and infinity tokens fail validation. CSV row order does not affect scoring.
- Segmentation is supplied and fixed; cellpose and PyTorch are not runtime dependencies. The source package contains optional APIs whose extra dependencies are not installed.

## Research scope

**Objective.** Assign the seven specified cell types to every supplied QC cell and maximize pooled agreement with the fixed consensus annotations.

**Fixed research objects.** The supplied 56-channel tissue images, segmentation masks, QC cell population and patient-group split; seven-class ontology, annotator-consensus rule and pooled Cohen kappa. These keep the population and prediction target comparable across methods.

**Editable methods.** Features, fluorescence normalization, threshold-selection methods and classifiers. There is no training split. Validation supports evaluation and method selection only. Do not fit reusable models or learned preprocessing on validation features or labels. Per-core computations using that core's input image, such as the baseline's image quantiles, remain permitted.

**Forbidden substitutions.** Resegmenting cells, changing QC eligibility or replacing the consensus/class definition; dropping difficult cells, using source-specific answer tables or exploiting opaque identifiers. When retaining the supplied threshold-and-argmax pipeline, changing only PAX5=0.8, CD3=0.7, CD11b=0.8, CD11c=0.9, CD68=0.9, CD90=0.95, Podoplanin=0.95, CD31=0.95, CD34=0.95 or shift=True does not count as a method improvement.

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
