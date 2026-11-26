# CLARITree private five-fold interface

`train.sh --out DIR` prepares the candidate method and native library. It receives no fold. `predict.sh --model DIR --input REQUEST --output FILE` exports a `private-policy-v2` ZIP artifact. REQUEST is exactly `{"task":"icml-2217-CLARITree","protocol":"private-policy-v2"}`. Neither entrypoint receives a fold. Validation checks the inert archive; scoring performs all private fits.

The trusted scorer launches five fresh network-isolated sandboxes, sequentially and with one CPU each. Each runs `policy.sh --model /ws/model --input /ws/input --output /ws/output/pred.out`. Its only data are `/ws/input/train.npz` arrays `x` (float64, n by p) and `y` (float64, n), where p is 13 in the formal run and 9 on the development set. A policy must read p from `x`. There are no other-fold labels, test features or test targets. Code and model are read-only; the temporary filesystem and process state are new for every fold.

The policy writes strict JSON `{"schema":1,"tree":NODE}`. A leaf is exactly `{"coefficients":[intercept,beta_0,...,beta_(p-1)]}` with p+1 finite numbers in original feature/target units: fourteen in the formal run, ten on the development set. An internal node is exactly `{"feature":j,"threshold":t,"left":NODE,"right":NODE}` with zero-based feature j and routing x[j] <= t to the left. One tree is required; root depth is zero and maximum leaf depth is four. Extra keys, malformed/nonfinite coefficients, duplicate JSON keys and more than 31 nodes are rejected. The certificate carries one coefficient per original feature. Leaf coefficients of binary columns stay zero, as in the paper's experiments: binary features are for splits only (scorer). A column counts as binary when it holds only 0 and 1 in that fold's training x, the same test as the threshold pool, and a nonzero coefficient on it is rejected. Leaf sample-size choices, leaf solvers and regularization are editable. A change of lambda, kappa or the minimum leaf size alone does not count as an improvement; the gain must come from the search or the leaf fitting (patch audit).

The global threshold pool is constructed once from the entire authorized training fold, matching the released implementation: columns containing only 0/1 use 0.5; other columns with at most 21 distinct values use adjacent-value midpoints; otherwise use the twenty interpolated empirical quantiles k/21 for k=1,...,20, removing duplicates. Constant nonbinary columns have no thresholds. Submitted thresholds must equal a pool value exactly; the trusted `threshold_pools` helper is available for development.

The scorer validates each actual certificate and reconstructs all predictions independently. For each original test fold, R² = 1 - SSE/TSS, using float64; the released constant-target convention returns 1.0 when TSS=0. The primary metric `mean_test_r2` is the unweighted mean of the five fold values, higher is better. Rejected execution or certificates return -Infinity and a fixed failure string. Private worker logs, trees and per-row predictions are not returned.

`data/val/replay` is the development set: the real Seoul Bike dataset from the paper's Table 2, 8760 rows with nine continuous features (`Hour`, `Temperature`, `Humidity`, `Wind_speed`, `Visibility`, `Dew_point_temperature`, `Solar_Radiation`, `Rainfall`, `Snowfall`) and the target `Rented_Bike_Count`, in its five original released folds. Each `fold_K.npz` holds `train_x`, `train_y`, `train_id` (7008 rows) and `test_x`, `test_y`, `test_id` (1752 rows); every row is in the test part of one fold and the training part of the other four. It shares no rows with the hidden dataset, so all of it is public, including its labels.

`python bench/scorer.py policy.zip data/val/replay` and `bench eval-dev` run it through the same five private sandboxes, certificate rules and metric as the formal run, with p=9. Each sandbox still receives only that fold's training x/y. The score is marked `development_only=true`. It is a guide to whether a change helps, not an estimate of the hidden score. Formal testing retains all original released five-fold populations and all thirteen features.

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
