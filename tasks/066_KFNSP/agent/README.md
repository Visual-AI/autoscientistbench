# Extending Fair Null-Space Projections for Continuous Attributes to Kernel Methods

## 1. Goal

Predict the normalized violent-crime rate of a community from census attributes,
while controlling dependence on the continuous protected attribute racepctblack.
The baseline applies fair kernel decomposition (FKD) to an RBF kernel on the
Communities & Crime data, then fits support vector regression (SVR) to the
transformed kernel.
Improve **fair_mae** (lower is better) on unseen communities, subject to all three
fairness limits. The judge retrains your code from scratch and compares its score
with the archived baseline. You may change the method within the rules below.

## 2. Data

- `data/train/train.csv`: 1197 training communities, opaque `id`, 98 numeric
  feature columns, target `y`, protected attribute `p`, and three public `fold`
  groups (399 rows each). Feature headers carry their original names.
- `data/val/input.csv` and `data/val/labels.csv`: 399 independent validation
  communities, the existing fold `4ec78c74` (lexicographically first public fold).
  These rows are absent from training. Both development and final models fit
  only the other three folds, using the same 1197 training communities.
- The evaluation split is held by the judge. It is fixed and uses the same
  feature schema and preprocessing; its rows are unavailable in this workspace.
- Administrative identifiers, the target, protected column and columns with
  missing values were removed from features. Preserve the supplied float32
  rounding; the baseline performs no further scaling. Targets are normalized
  ViolentCrimesPerPop values, not counts.
- Use only the supplied data. Keep fold membership, IDs and labels unchanged.
  Model selection may use the independent validation results.

## 3. Metric and scoring

`bench/scorer.py` computes mean absolute error (MAE), HGR and GDP demographic
parity scores, and PF equal opportunity scores per fold, then takes an equal-weight
fold mean. The score is mean MAE if HGR, GDP and PF are each no greater than
their frozen caps in the labels for that evaluation distribution.
Otherwise **fair_mae = 0.1785**, the specified DUMMY fallback. Invalid output
returns Infinity and cannot pass. The preserved PF implementation and its
ordering behavior are part of the metric definition.

| Guard | Enforcement |
| --- | --- |
| HGR no greater than the frozen baseline cap | scorer |
| GDP no greater than the frozen baseline cap | scorer |
| PF no greater than the frozen baseline cap | scorer |
| A prediction uses its own features and a fixed trained model; model is unchanged during prediction | manual subset-consistency and artifact-hash checks; patch audit |
| Exclude the validation fold from every fit, including final retraining; no memorized predictions or outside data | patch audit |

The measured validation reference has mean fair_mae 0.10085594654083252
and standard deviation 0.0 over one seed. Frozen validation fairness caps are
HGR 0.421795420349, GDP 0.032299573999 and PF 0.197910189629.
These caps are fixed constraints and are not recalculated from new models.
The judge compares the mean score over its configured seeds; the required relative
improvement is **0.1%** (delta=0.001), with the harness's variance adjustment.
`bench status` reports the baseline and threshold.

## 4. Baseline

Official code: https://github.com/Felix-St/FairKernelDecomposition at
`26ffcefc79087fec101ff4059329921e99bc6262`.
The model is in `code/src/models/decomposition_models.py`.

SVR-FKD uses m=30, epsilon=0.01, RBF gamma=0.05, C=0.75,
alpha_prime=0.05, max_iter=50000, no Nystroem approximation. The adapter
`code/main.py` adds train-only fitting, checkpointing and
CSV prediction. Model equations, solver and hyperparameters are unchanged.
It drops unused intermediate transform histories only when saving the model.

The baseline fits one model on the 1197 training rows, then uses that same
model for validation and new communities. Numerical libraries use one thread.
The reference measured 29.61 seconds for training and 2.62 seconds for
validation prediction in its single seed run.
Memory budget is 16 GB; a precise peak-memory measurement was not collected.

## 5. Interface (frozen)

- `bash code/train.sh --out DIR` trains from `BENCH_DATA_DIR/train/train.csv`
  and writes all model artifacts into DIR.
- `bash code/predict.sh --model DIR --input PATH --output FILE` writes CSV
  with the exact header `id,pred`, in input row order. The output may have no
  filename extension. See `bench/io_spec.md`.
- Honor `BENCH_SEED`, `BENCH_CPUS`, `BENCH_DATA_DIR`, `BENCH_BENCH_DIR`
  and `BENCH_CODE_DIR` where provided. Paths passed to the scripts are absolute.
- `bench status`, `bench eval-dev [--seed S]`, `bench validate-output FILE`,
  and `bench submit [--final] [-m MSG]` are the development interfaces.
- Run `python bench/check_model.py MODEL --code code --input data/val/input.csv
  --pred PRED` after a local evaluation. This manually checks subset consistency
  and model immutability; patch review is also required.
- Python 3.12.13; `code/env.lock` pins the rebuildable runtime and scorer
  dependencies. Use PyPI or the PyTorch CPU index, without editable/local installs.

## 6. Rules

The research scope below determines which components may change. Supplied data and splits, trusted tests, information boundaries and the evaluation metric remain fixed.


- Editable surface: fair-regression estimators, derived features, training surrogate losses
  and optimization under `code/`, within the fixed fairness problem and inductive protocol.
- Frozen: `data/`, `bench/`, this README, the input/output schema and script
  signatures. Do not patch them.
- The allocation is 8 CPUs and 16 GB memory. No network during training or prediction.
- The judge uses one seed for this deterministic baseline; your code must honor
  BENCH_SEED, including if your replacement method introduces randomness.
- Fit only on the 1197 designated training rows, including final retraining.
  Use the same fitted model for validation and new communities. Do not identify the evaluation split
  by its path, size, or reconstructed source indices.
- No external datasets, including a copy of this data from an installed package, embedded paper predictions, hard-coded targets,
  recovery of unavailable rows, or batch-dependent/transductive prediction.
  Predict must not alter the model files. IDs are bookkeeping only and cannot
  route special validation models, be predictive features or store target predictions.

## 7. Known pitfalls

Validation is a disjoint existing fold: fitting on the full 1596-row public
pool, including the validation fold, is invalid even for the final submission. The protected attribute is available
for training and scoring, but absent from inference inputs. Fairness caps can
be exceeded even when MAE improves; inspect all three diagnostics. Preserve row
order because the released PF routine uses it. The model uses float64 internally
and float32 predictions; changing dtypes changes the baseline. Export PYTHONPATH
in shell entry points because the harness enables PYTHONSAFEPATH. Torch is
needed by the trusted fairness scorer even though the SVR itself uses NumPy and
scikit-learn. Declared dependencies cover the task entry points; unused upstream
neural/plotting modules are not part of the runnable baseline.

## Research scope

**Objective.** Predict normalized community violent-crime rate while satisfying the original three continuous-protected-attribute fairness constraints.

**Fixed research objects.** observed community features, crime target, protected racepctblack attribute and fixed independent training/validation partitions; inductive per-row prediction with the protected attribute absent at inference; original MAE/HGR/GDP/PF computation and frozen fairness caps.

**Editable methods.** fair regression/representation methods, training surrogate losses, optimization and legal derived features; model architecture and fitting procedures respecting train-only fitting for development and final submission.

**Forbidden substitutions.** changing the target/protected attribute, fairness definition or caps, or fitting any model or learned preprocessing on the validation fold; using prediction-batch transduction, protected attributes at inference, administrative identities or recovered source rows. When retaining the supplied SVR-FKD pipeline, changing only m=30, epsilon=0.01, gamma=0.05, C=0.75 or alpha_prime=0.05 does not count as a method improvement.

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
