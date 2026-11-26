# I/O specification

## Input
CSV `id,responses`, sorted by numeric id. `id` is the original row position in the source task table.
Each `responses` cell is a JSON list of `[subject, block, rank, response]` observations.
`subject` is the stable integer index of a model. `response` is binary success after the authors' aggregation.
Absence means unobserved, never failure. `block` and `rank` preserve the original within-subject observation
order; sort by `(block,rank)` when combining tasks. They are ordering metadata, not predictors or labels.
Task IDs in different source tables are separate namespaces. No task names or human times occur in input.

## Public data
`data/train/input.csv` and `labels.csv` contain labeled calibration tasks; labels have header `id,y`,
where `y` is human completion time in minutes. `data/train/auxiliary.csv` contains additional unlabeled
context tasks in the same input format and its own ID namespace. `data/val/input.csv` and `labels.csv`
use the same format. Validation families are disjoint from training families.

## Output
CSV with exactly `id,pred`, sorted by numeric id, exactly one row for every input ID and no other IDs.
`pred` is finite nonnegative human minutes, not a bucket index. No duplicates, missing values, or extra columns.
Write to the exact output path, which need not have a file extension.

## Model and prediction protocol
`train.sh --out DIR` writes all required model state under DIR. The baseline writes a fitted IRT model,
calibration coefficients, seed and training context. Each `predict.sh` call may jointly fit the supplied
unlabeled batch and saved training context: BRIDGE is transductive. It must not retain information from a
previous prediction call. Only training human times calibrate the baseline; no query labels are available.

## Scoring
`scorer.py PRED LABELS` prints one JSON line. `bucket_accuracy` is accuracy in percent if linear-weighted
Cohen kappa is at least 0.22 and four-class macro F1 is at least 0.27; otherwise it is 0.
Buckets are [0,15], (15,60], (60,240], (240,infinity) minutes, matching the authors' right-closed `pd.cut`.
Invalid prediction contracts produce Infinity and are rejected. Diagnostics include raw accuracy, F1 and kappa.

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
