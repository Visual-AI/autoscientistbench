# 025_QUED interface

The objective is complete-population MAE in log10(mol/kg). RMSE is reported separately
as a diagnostic; it does not change the scalar ranking metric.

Run `bash code/train.sh --out DIR`, then
`bash code/predict.sh --model DIR --input PATH --output FILE`.
The outer input is `input.json` identifying this task and the evaluation population.
The outer output is an inert `private-policy-v2` ZIP archive containing source and a
training recipe. `validate.py PRED INPUT` checks this format. `scorer.py PRED LABELS`
receives a trusted replay directory and returns a JSON metric object.

Trusted replay invokes `bash code/policy.sh --model DIR --input BUNDLE --output FILE`.
The fit bundle has `train.h5`, `train_labels.csv` and `request.json` with phase `fit`.
It has no evaluation features or labels. Fitting returns an inert fitted-state
`private-policy-v2` archive. A new worker receives that archive and a prediction
bundle containing only `eval.h5` and `request.json` with phase `predict`. Each HDF5
contains only `id` (unique UTF-8 byte strings) and `X` (float64 N by 39553):
39340 original Bag-of-Bonds columns followed by 213 raw DQM columns. DQM order is
eight global scalars, dipole norm, eight orbital energies and 196 padded atomic charges.
The fixed definition is in `data/descriptor.json`. No labels occur in feature files.
Training labels have the exact CSV header `id,y`, matching training input order.

The prediction phase must return UTF-8 CSV with exact header `id,pred`, one finite prediction
for every evaluation id in its input order. Missing, extra, duplicated, reordered,
nonfinite or malformed rows fail. A row's prediction uses only the authorized training
observations and supplied descriptor values. Evaluation labels are never mounted in
the isolated worker, and its logs and failure text are not returned.

Development uses 4000 training rows and 1000 validation rows selected exclusively
from the original 5000-row training pool. Formal replay uses the same 4000
training rows and requires predictions for every designated evaluation observation.
The 1000 validation rows are excluded from both development and formal fitting.
Fresh preprocessing and fitting occur inside each replay; the outer recipe stage
does not replace this fit.
Use `BENCH_SEED`, `BENCH_CPUS`, `BENCH_DATA_DIR`, `BENCH_BENCH_DIR` and
`BENCH_CODE_DIR` for their documented runtime roles. Policies receive writable
temporary/output directories and read-only code, recipe, runtime and input files.

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
