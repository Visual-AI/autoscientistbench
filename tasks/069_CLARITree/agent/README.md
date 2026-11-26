# CLARITree: piecewise-linear regression trees

Improve mean test R² across the five original released folds of a hidden regression dataset. Submit one piecewise-linear regression tree per fold, depth at most four, with the original twenty-quantile threshold rule and linear leaves. The trusted scorer fits each fold independently and reconstructs predictions from its checked tree certificate.

## Research question and experiment

The source paper asks how to learn near-optimal sparse piecewise-linear regression trees at a cost that scales to large datasets. Its objective is squared error, plus a ridge penalty kappa on each leaf's linear coefficients, plus a penalty lambda per leaf. Exact dynamic programming (STreeD) reaches the optimum when it finishes, but on this task's hidden dataset it frequently exhausts a 10-minute budget before reaching depth four. Greedy induction (GUIDE, M5, PILOT, the authors' CholeskyTree) is fast but can land far from the optimum. CLARITree tries every candidate split at a node, completes each side with a greedy subtree, keeps the split with the best completed objective, and repeats in both children. Rank-one Cholesky updates of the regularized Gram matrix keep that scan cheap and numerically stable.

The paper's Table 2 compares these methods by test R² in one tree class: depth 4, at most 20 quantile thresholds per feature, five outer 80/20 folds, and binary features used for splits only. The authors sweep lambda and select kappa by 3-fold cross-validation inside each training fold. They report that CLARITree is never more than about 0.01 below the optimal solver's test R². This task keeps one cell of that table, the hidden dataset, and asks whether a better method in the same tree class raises its test R².

## Metric and guards

`mean_test_r2` is the unweighted mean of the five fold test R² values, higher is better, with R² = 1 - SSE/TSS in float64. Test features and targets stay with the scorer.

The scorer enforces the tree class on every fitted certificate. One tree per fold, maximum leaf depth four, at most 31 nodes. Every branch threshold must exactly equal a value in that fold's global pool. Every leaf holds exactly one intercept and one coefficient per feature, all finite, with a coefficient of exactly zero on every binary column, meaning a column that holds only 0 and 1 in that fold's training x. Extra or duplicate keys, a certificate over 128 KiB and nonfinite predictions are rejected. A rejected certificate, a failed fit or an exhausted replay budget scores the whole run -Infinity. See [the interface](bench/io_spec.md) for certificate details.

## Research contract

**Objective.** Improve mean test R² over the five released outer folds of the hidden dataset within the fixed depth-four piecewise-linear tree class.

**Fixed research objects.**
- One piecewise-linear regression tree per fold, depth at most four, with linear leaves and the permitted threshold pool built from the twenty-quantile rule (scorer).
- Leaf coefficients of binary columns stay zero, as in the paper's experiments: binary features are for splits only (scorer).
- The data population, the five original folds, feature and target meaning, and equal-weight mean test R² (scorer).
- The trusted evaluator and lawful information access: each fit sees only its own fold's training x/y (sandbox).

**Editable methods.** Inside `code/` you may change tree construction and search, lookahead, Cholesky or other numerical leaf fitting, auxiliary training losses and optimization, within the fixed class. The baseline's lambda=.001, kappa=1e-5, automatic minimum leaf size and its specific CLARITree implementation are editable method choices; they cannot justify replacing the tree class. A change of lambda, kappa or the minimum leaf size alone does not count as an improvement; the gain must come from the search or the leaf fitting (patch audit). The baseline's leaves fit the continuous columns and emit zero coefficients for binary columns.

**Forbidden substitutions.**
- Forests, unrestricted or neural regressors, deeper trees and alternate split grids (scorer rejects the certificate).
- Changing the outer folds, or replacing test R² by another score such as a placeholder MSE (scorer fixes both). A squared-error training objective, as in the baseline, remains a method choice.
- Target memorization, exposing private observations or labels, and external real datasets, including any copy of the hidden dataset from a package or from memory (patch audit; the sandbox gives no network and no other fold's data).
- Fitting reusable model state or learned preprocessing on validation features or labels, including final train+val refits and validation-based distillation (patch audit).
- Presenting a Seoul Bike development score as performance on the hidden dataset.
- Editing `bench/` or fixed data.

## Baseline

`code/train.py` builds the native library and writes the parameters. `code/policy.py` fits the reproduced CLARITree C++ sources in `code/src` and writes the certificate. It uses depth 4, 20 quantile thresholds, lambda=.001, kappa=1e-5 and the automatic minimum leaf size (five times the number of continuous columns).

Development score: 0.7123, std 0 with one seed; the fit is deterministic. Its five fold values are 0.7236, 0.6405, 0.7428, 0.7255 and 0.7291. The low value comes from a few rows that extrapolate badly in a linear leaf.

## Data

The scored dataset is hidden and is not named here. It is a real regression dataset with thirteen features, continuous variables and binary indicators, and a real-valued target. Workers receive only numeric arrays, with no feature names. Its folds cannot be public because each row is a held-out observation in one of the five complementary folds.

There is no public training split: `data/train` holds no data. Models are fitted only inside the private policy runs, on that fold's training x/y.

`data/val/replay` is the development set: the real Seoul Bike dataset from the paper's Table 2, 8760 hourly records with nine continuous features and the rented-bike count as target, in its five original released folds (7008 training and 1752 test rows per fold). It shares no rows with the hidden dataset, so all of it is public, including its labels. It is for evaluation and for method, hyperparameter or checkpoint selection; no model or learned preprocessing may be fitted on it and carried into a policy (patch audit). `bench eval-dev` fits each fold's training part and scores its test part through the same private interface as the formal run: five sandboxes with one CPU each, the same tree rules and the same metric, with nine features instead of thirteen.

A Seoul Bike score is a guide, not an estimate of the hidden score. Compare a candidate with the baseline's score on the same set from `bench status`. The two datasets differ: Seoul Bike has fewer rows, so its mean R² carries more sampling noise, and it has no binary columns, so splits on binary features cannot show on it. A row bootstrap of the baseline's mean gives a standard error of about 0.012. Comparisons on the same folds have no run noise, but a gain below 0.01 may not transfer. The authors selected the baseline's lambda and kappa for the hidden dataset, not for Seoul Bike, so a gain from retuning them on Seoul Bike says little about the hidden dataset.

## Interface and resources

Run `bash code/train.sh --out MODEL`, then `bash code/predict.sh --model MODEL --input data/val/request.json --output policy.zip`. Validate with `python bench/validate.py policy.zip data/val/request.json` and score on the development set with `python bench/scorer.py policy.zip data/val/replay`; `bench eval-dev` runs all four steps. The validator checks only the inert archive; the scorer checks every fitted tree.

The outer train command prepares a method artifact and receives no fold. Actual learning happens within five isolated private policy runs, one after another, with one CPU each and no network. Each receives only its own training x/y, and starts with a fresh writable filesystem and process state; code and model are read-only. Your policy must take the feature count from the data, so that it works with both nine and thirteen.

The native build is serial and uses `c++`, active Python development headers, bundled Eigen 3.4.0/pybind11 3.1.0 headers and NumPy 2.4.6. It does not need a source checkout, the original environment or network. The compiled library is kept in MODEL, so source edits are rebuilt by training.

Allocation: 16 CPUs and 16 GB configured memory. Baseline times: `train.sh` 21 s, `predict.sh` 0.5 s, about 30 seconds per fold on the development set, about three minutes per run. There is no per-fold time limit, and the paper's 600-second limit does not apply; only your overall deadline and a separate overall budget for the formal replay do. The hidden dataset has more rows and more features than Seoul Bike, so each fit takes longer there.

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
