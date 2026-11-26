# I/O specification

`train/train.csv` has `id`, 98 feature columns, `y`, `p`, `fold`. `y` is
ViolentCrimesPerPop; `p` is racepctblack. Features exclude the target, protected
attribute, administrative identifiers and columns containing missing values.
Values have the authors' float32 rounding. The supplied input has no fitted normalization; derived transformations inside the submitted method are allowed.
IDs are opaque strings; their numeric interpretation is forbidden.

`--input` is a CSV with `id` followed by the same 98 feature columns, in the
same column order. It contains no target or protected attribute.

`--output` is CSV with the exact header `id,pred`, one finite numeric prediction
per input row, in exactly the input's order. No missing, extra or duplicate IDs,
extra columns, NaN or Infinity are accepted. The filename may have no extension.

`labels.csv` contains `id,y,p,fold,hgr_cap,gdp_cap,pf_cap`. Labels and fold IDs
are used by the scorer, which evaluates each fold in its original order and
averages fold metrics with equal weight. Three caps are constant across a labels
file. The caps are fixed constraints and are not recalculated from new models.
The scorer casts predictions, labels and protected attributes to float32, runs
the released metrics and rounds each fold score to float32 before averaging in
float64. Fairness violation returns `fair_mae=0.1785`; malformed output returns
`fair_mae=Infinity`. Output is one JSON line, with MAE and fairness diagnostics.

Training contains 1197 rows from three existing folds. Validation contains the
399 rows of fold `4ec78c74` and is disjoint from training. The same training-only
model predicts both validation and new rows; final fits must also exclude all
validation observations. The baseline saves `all.joblib` and `model_card.json`
(format `training-only-refit/v1`) recording features and training IDs. Alternative
model artifacts are allowed. IDs cannot select validation-specific models. Never
route by path, input size, target, or a reconstructed external row index.

Each prediction must depend only on its own features and the trained model.
Prediction must leave model files unchanged. `check_model.py MODEL --code CODE
--input INPUT --pred PRED` checks sparse reversed-subset consistency (absolute
tolerance 1e-6) and model immutability. Run this check explicitly. Training
provenance and fold exclusion additionally require patch audit.

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
