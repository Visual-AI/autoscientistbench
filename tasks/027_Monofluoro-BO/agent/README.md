# Bayesian optimization- and high-throughput-guided synthesis of monofluoroalkenes via defluorinative C-C bond formation

The judge retrains submitted code from scratch and scores predictions on data held by the judge.

## 1. Goal
Predict reaction productivity and yield from the SMILES of two substrates and their proposed monofluoroalkene product,
under the fixed Pd168/KHCO3/DCE/80 C reaction condition. Improve `r2` (higher is better), while maintaining
positive-class classification F1 of at least 0.912 on test and 0.80 on val. The proposed product is an input, not a product-generation target.
This benchmark compares supervised productivity and yield prediction methods on the same reaction observations. Keeping the chemical condition fixed isolates the effect of molecular structure. The separate Bayesian optimization campaign is outside this task.

## 2. Data
- `data/train/input.csv` and `data/train/labels.csv`: 696 training reactions.
- `data/val/input.csv` and `data/val/labels.csv`: 88 validation reactions with public labels.
- Inputs contain `id,sub_smiles1,sub_smiles2,prod_smiles`; labels contain `id,y,cls_label`. Yield `y` is in percent;
  `cls_label = 1` exactly when yield is positive. Zero-yield reactions remain in both training and scoring.
- Validation holds out 20% of the product Bemis–Murcko scaffold groups from the supplied training population,
  using fixed seed 20261030. Unequal group sizes mean the row fraction differs from 20%. Train and validation
  product scaffolds do not overlap. IDs are original row numbers; files are sorted by ID.
- The baseline and candidates fit only on train, including final models. Validation may be used for evaluation and selection.
  Do not change the evaluation files, relabel data, recover omitted data, or add external data.

## 3. Metric and scoring
`bench/scorer.py` computes R2 over **all** yields: `1 - sum((y-pred)^2)/sum((y-mean(y))^2)`.
Classification F1 uses the positive class and the submitted `pred_cls` column. A failed guard returns `Infinity`,
which the harness treats as an invalid score and ranks last for either metric direction.

| Guard or rule | Enforcement class |
|---|---|
| Positive-class F1 >= 0.912 on test, >= 0.80 on val | scorer |
| Exact IDs, finite yield predictions, binary classification, valid CSV | scorer (also checked by validator) |
| Fit models and learned preprocessing only on train; use validation only for evaluation and selection; no memorized predictions or reconstruction of omitted rows | patch audit |

There is no model-family or stage-count restriction.

Measured reference validation: F1 **0.8039215686**, R2 mean **-0.2024346262**, and R2 standard deviation **0.0** from one run.
This F1 passes the 0.80 validation guard with little margin, so the guarded score is that R2.

The judge uses one seed for this deterministic baseline. Relative improvement is compared against
`max(delta, 2 * reference_std / abs(reference_mean))`, with `delta = 0.001` (0.1%).
A valid score must first pass the F1 guard. `bench status` exposes the configured baseline and budgets.

## 4. Baseline
Official code: https://github.com/TheLiaoGroup/defluorinative_C-C_bond_formation at
`f8df83988dcd6ebb148ecd05d4c2058b5900cfdd`.

The baseline concatenates three radius-2, 2048-bit Morgan fingerprints into 6144 float32 features.
XGBoost classification uses 200 trees with depth 6; regression uses 300 trees, depth 6, learning rate 0.05,
and squared-error loss, fitting only positive-yield training rows. Predictions classified inactive are zero;
productive predictions are clipped below at zero. Other estimator defaults are unchanged.
The method primitives are in `code/src/reaction_discovery/prediction/staged_model_cli.py`.

The baseline trains both stages on the designated training data and saves the fitted models for prediction. Seeds and thread counts are supplied through the environment.

Measured reference runtime at 8 CPUs: train **5.22 s**, validation prediction **3.72 s**.
The memory budget is 4 GB; the reference record does not report peak memory.

## 5. Interface (frozen)
- `bash code/train.sh --out DIR` writes the artifacts required for prediction; the baseline saves a classifier, regressor and metadata.
- `bash code/predict.sh --model DIR --input PATH --output FILE` writes CSV `id,pred_cls,pred`.
  The output filename may have no extension. See `bench/io_spec.md`.
- Environment: `BENCH_SEED`, `BENCH_CPUS`, `BENCH_DATA_DIR`, `BENCH_BENCH_DIR`, `BENCH_CODE_DIR`.
  All external paths are supplied by the caller. Scripts resolve relative arguments before changing directory.
- Commands: `bench status`, `bench eval-dev [--seed S]`, `bench validate-output FILE`, `bench submit [--final] [-m MSG]`.
- Dependencies are pinned in `code/env.lock`. Use `uv pip install` inside the workspace environment;
  submissions freeze dependencies. Only version-pinned PyPI packages are rebuildable. Python is 3.13.

## 6. Rules

The research scope below determines which components may change. Supplied data and splits, trusted tests, information boundaries and the evaluation metric remain fixed.

- You may change the productivity/yield prediction method, derived features and training surrogate losses under `code/` within the research scope below.
- `data/`, `bench/`, this README, prediction format and script signatures are frozen.
- Resource allocation: up to 8 CPUs and 4 GB memory.
- Honor `BENCH_SEED` and `BENCH_CPUS`. Randomness you introduce must be seeded.
- Train and prediction run without network. Save required fitted artifacts under `--out` and predictions under `--output`.
- Recovering hidden data from external repositories, using IDs to reconstruct omitted reactions, hard-coding predictions,
  or embedding results from the article is disqualifying.

## 7. Known pitfalls
- Validation product scaffolds do not occur in training, so validation is hard for this baseline.
  The baseline's validation F1 is only just above the 0.80 guard, so a change that lowers classification can fail it.
- Classification labels cannot be omitted: a strong yield R2 alone does not pass the scoring rule.
- `PYTHONSAFEPATH=1` is set by the harness; retain the scripts' explicit module search path if adding imports.
- RDKit parses the three supplied SMILES. Output must still have exactly one prediction for every input ID.
- Inputs and labels are read-only. The judge rebuilds from a clean code commit, so ignored or uncommitted artifacts are absent.
- The trimmed lock includes cloudpickle because joblib requires it. GPU/NCCL packages are unnecessary for this CPU path.

## Research scope

**Objective.** Predict productivity and percent yield for the supplied substrate/product reactions under the fixed reaction condition, maximizing all-row R2 subject to positive-class F1>=0.912 on test and >=0.80 on val.

**Fixed research objects.** reaction observations, fixed Pd168/KHCO3/DCE/80C condition, supplied product as an input, scaffold split and zero-yield rows; yield/productivity target definitions, all-row R2 and fixed classification-F1 guard.

**Editable methods.** chemical features derived from the supplied SMILES, classifiers/regressors, joint or staged prediction architectures; training surrogate losses, optimization and hyperparameters of a changed method; the Bayesian experimental campaign is outside this task.

**Forbidden substitutions.** generating different reactions/products, changing chemical conditions or replacing the productivity/yield target; discarding zero yields, relaxing the F1 guard or recovering omitted reactions from row identifiers. When retaining the supplied staged XGBoost pipeline, changing only Morgan radius=2 or bits=2048, classifier n_estimators=200 or max_depth=6, or regressor n_estimators=300, max_depth=6 or learning_rate=0.05 does not count as a method improvement. Selecting the existing `rf` or `svm` branch of `build_model_bundle` with no other change also does not count.

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
