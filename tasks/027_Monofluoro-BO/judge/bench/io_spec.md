# I/O specification

Input is CSV with the exact header `id,sub_smiles1,sub_smiles2,prod_smiles`.
The three SMILES describe the two substrates and the proposed product. No yields or class labels are included.
IDs are original row numbers represented as strings; treat IDs as identifiers only.

Training uses `$BENCH_DATA_DIR/train/input.csv` and `labels.csv`; validation uses
`$BENCH_DATA_DIR/val/input.csv` and `labels.csv`. Labels have header `id,y,cls_label`:
`y` is measured yield in percent, `cls_label` is 1 if yield is positive, otherwise 0.

Output is text CSV with exact header `id,pred_cls,pred`. Each input ID must occur exactly once;
order may differ. `pred_cls` is the predicted productive class (literal 0 or 1), and `pred` is
finite predicted yield in percent. The output filename need not have an extension.

`train.sh --out DIR` saves all required fitted state into DIR. The baseline writes separate
classifier and regressor pickle files plus metadata. `predict.sh` must work with only these
artifacts and the provided input.

`validate.py PRED INPUT` checks shape, headers, ID set, duplicates, class values and finite yields.
`scorer.py PRED LABELS` returns one JSON line. `r2` is R2 over all rows, including zero yields,
provided the positive-class F1 is at least 0.912 on test and 0.80 on val. A failed guard or malformed file returns
`Infinity`, the harness's failed-submission sentinel (rank last, irrespective of metric direction).
Diagnostics include `raw_r2`, `f1`, confusion counts, MAE and RMSE.

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
