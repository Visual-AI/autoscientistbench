# LI-DTI numerical interaction prediction

## 1. Goal

Predict missing interactions in the supplied Luo/DTINet binary matrix, whose rows are drugs and columns are protein targets. Improve AUPR under the fixed pair-holdout protocol. The seven side-information sources describe chemical similarity, side effects, drug-drug interactions, drug-disease associations, protein sequence similarity, protein-protein interactions and protein-disease associations. The task retains ten repetitions of ten author folds, with permanent outer masking to prevent cross-fold disclosure.

## 2. Data

`data/train/full.mat` contains the filtered 549×424 observed interaction matrix, supplied side information, and 100 label-free author fold masks. A fixed hash of pair coordinates assigns 70% of pairs to training and 10% to validation. All pairs outside training are zeroed in the observed matrix before every fold fit. Author fold masks are additionally applied at fitting time. The source’s feature preprocessing is preserved.

`data/val/input.csv` and `data/val/labels.csv` contain all validation occurrences in every author fold, including repeated pair occurrences and every negative. Fold masks contain only sorted coordinates; neither the label column nor positive-first row ordering is exposed. `data/train/smoke.mat` is a small real-data fixture selected only during explicit harness smoke checks.

Do not resplit the frozen data or use external interaction labels. Fit models and learned preprocessing only on the designated training observations. Use validation only for evaluation and method, hyperparameter or checkpoint selection. The outer partition changes the training and evaluation populations relative to the original author folds.

## 3. Metric and scoring

`bench/scorer.py` computes trapezoidal precision-recall area using one point per distinct score threshold, without an artificial initial `(recall=0, precision=1)` point. It averages 10 fold AUPRs within each run, then the 10 run means. Every occurrence of every negative in the outer intersections is included; there is no negative undersampling. This matches the supplied `AUPR_perfcurve.m` convention.

The validator and scorer require exactly one finite prediction per input sample key, with no missing, extra or duplicated keys. A pair may occur repeatedly under distinct sample IDs. These structural guards are checked by the scorer; invalid predictions score 0. Data isolation, the fixed evaluation population, use of supplied data only, and absence of memorized outputs are enforced by sandboxing and patch review.

Measured validation AUPR is 0.4100615826445977. The reference standard deviation is 0.0 from one execution; this is separate from the scorer's variation across author runs.

## 4. Baseline

The baseline is the official GNU Octave LI-DTI implementation from `https://github.com/paccanarolab/LI-DTI`, source commit `24c7e45a3bb2d8ae53f3e39bbd18e519e8407db0`. The numerical functions in `code/code (Octave)/` remain editable and execute in training and prediction. LI-DTI combines drug-based and target-based similarity predictions using learned nonnegative weights. It minimizes squared reconstruction error with penalties on interaction-derived and side-information weights. The baseline uses the original gradients, normalizations, seven thresholds `[0.6,0.5,0.7,0,0.3,0.4,0]`, penalties `lambda1=1`, `lambda2=0.1`, nonnegative bound, random weight initialization and optimizer settings (`sqp`, gradient enabled, 10000 iteration and evaluation limits).

`bench_train.m` fits 100 fresh fold weight vectors by default. `bench_predict.m` recomputes fold predictions with those weights using `get_predictions.m`; it does not load cached predictions. Measured training took 143.72 seconds, and validation prediction took 5.97 seconds.

## 5. Interface (frozen)

- `bash code/train.sh --out DIR` saves trained artifacts; the baseline writes `DIR/model.mat`.
- `bash code/predict.sh --model DIR --input PATH --output FILE` writes the CSV defined in `bench/io_spec.md`.
- `BENCH_SEED` controls initialization. `BENCH_CPUS` and the harness’s numerical-library environment variables bound threads. Data and evaluator paths are supplied through `BENCH_DATA_DIR` and `BENCH_BENCH_DIR`.
- The standard `bench status`, `bench eval-dev`, `bench validate-output` and submission interfaces apply.

The fixed runtime is a copied GNU Octave 10.2.0 installation with optim 1.6.3, statistics 1.7.7 and struct 1.0.18. Python tools use only the system standard library. Runtime snapshots, dependencies and `env.lock` are fixed.

## 6. Rules

The research scope below determines which components may change. Supplied data and splits, trusted tests, information boundaries and the evaluation metric remain fixed.


Interaction-prediction models, numerical methods and training surrogates under `code/` are editable within the research scope below; entrypoint signatures and prediction format remain fixed. Data, scorer, validator, documentation, runtime and the outer partition are frozen. Do not recover hidden labels from external copies, embed predictions, replace computation with lookup tables, or access unavailable source files.

The full evaluation retains 100 fold groups and all their supplied sample occurrences. The allocation is up to 8 CPUs and 48 GB memory. Execution is offline.

## 7. Known pitfalls

The paper’s Octave `AUCDownSample.m` prepends an artificial PR point; the benchmark instead uses the reproduction’s `AUPR_perfcurve.m` convention. Both tie grouping and the missing initial point matter. Pair occurrences in author folds can repeat, so sample IDs must be preserved. The unmodified source driver expects its old directory layout; use the provided entrypoints. Smoke selects a 32×24 real-data subproblem and a single original fold, while the default path always selects the full 100-fit dataset.

## Research scope

**Objective.** Predict the fixed observed binary interaction relation under the globally masked outer holdout and original 100-fold scoring protocol.

**Fixed research objects.** supplied interaction/side-information matrices and permanent outer masks plus author fold masks; all positive and negative sample occurrences, threshold/tie convention and original nested mean trapezoidal AUPR.

**Editable methods.** numerical interaction-prediction models, representation learning, training surrogates and optimizers; internal feature transformations respecting every fold exclusion.

**Forbidden substitutions.** When retaining the supplied LI-DTI pipeline, changing only the similarity thresholds [0.6,0.5,0.7,0,0.3,0.4,0], lambda1=1 or lambda2=0.1 does not count as a method improvement. Also forbidden: changing interaction truth, fold/outer populations or negative sampling; recovering held-out labels from other folds/external copies or replacing computation with pair-ID answer tables. Patch audit enforces a ban on using how often a pair occurs in query rows or fold masks as label evidence or as a prediction feature.

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
