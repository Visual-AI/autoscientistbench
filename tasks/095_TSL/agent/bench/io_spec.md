# Separable regression input and model contract

## Data and predictions
Input CSV header is exactly
`id,longitude,latitude,housingMedianAge,totalRooms,totalBedrooms,population,households,medianIncome`.
These are the original eight raw features, in that order. Training adds target
column `y` (median house value in USD). Input files and splits remain fixed.

Prediction is CSV `id,pred`, exactly one finite numeric prediction for every input
ID, in any order. The frozen scorer computes ordinary raw-USD RMSE. IDs are not
predictive features.

## Model card and rank budget
Train saves `MODEL_DIR/model_card.json`. The generic representation is:

```json
{"format":"separable-product-card/v2","n_features":8,
 "intercept":0.0,
 "terms":[{"coef":1.0,"factors":[{"feature":0,"kind":"executable","name":"longitude_factor"}]}]}
```

The predictor is intercept + sum over terms of coef * product of factor values.
There are at most four terms in total, counting a nonzero intercept as one term.
An empty factor list represents a constant product and also consumes one term.
Coefficients, factor outputs and final predictions must be finite. They may have
either sign. There is no mandatory ordering, positivity or pairing of terms.
`feature_names`, if supplied, must equal the eight input feature names in order.

Each factor reads one raw feature (`feature` is 0..7). A factor may implement any
univariate function, including a learned neural network or internal transformation
of that one feature. Repeated factors on the same coordinate are allowed: their
product is still univariate. Factors cannot read other columns, row identities,
other evaluation examples or labels.

The legacy `separable-product-card/v1` representation with `stages` and nested
`products` is still accepted. Stage grouping is descriptive only. The same total
four-term budget, including a nonzero intercept, applies across all stages.
Candidates are not limited to two stages or two products per stage.

## Factor representations
- `step`: `splits` is finite and strictly increasing, and `values` has one more
  entry than `splits`. Evaluation uses `searchsorted(...,side="right")`.
- `linear`: finite, strictly increasing `knots` and equally sized `values` define
  piecewise-linear interpolation with constant extrapolation (`numpy.interp`).
- `executable`: `name` identifies any trained univariate function implemented by
  candidate code. This is not restricted to the built-in factor families.

For executable factors, train must freeze `MODEL_DIR/factors.zip` with
`private_eval.export_artifact(code_dir,MODEL_DIR,MODEL_DIR/'factors.zip','icml-6321-TSL')`
from BENCH_BENCH_DIR. Include `code/policy.sh --model DIR --input REQUEST --output FILE`.
The artifact contains the same model card, learned parameters, code and pinned
dependencies. The output archive itself is excluded from its contents.

Each isolated factor invocation receives JSON with exactly
`{"feature":0,"factor":"longitude_factor","x":-122.0}`.
It returns JSON with exactly `{"value":FINITE_NUMBER}`. Only this scalar input,
function identity and read-only fitted model/code are available. No original ID,
other feature, row position or batch is supplied. Every invocation has a fresh
process and private writable storage. Equal (feature,name,x) queries may be cached
by the trusted checker. Factor state must come from the trained model, not other
invocations. The checker alone multiplies factors and combines the terms.

## Structural verification
`python bench/check_model.py MODEL_DIR --input INPUT.csv --pred PRED.csv` checks the
fixed four-term budget and reconstructs all predictions. Maximum
`abs(pred-reconstructed)/max(1,abs(pred))` must be less than 1e-6. Non-reproducing
cards, malformed factors, missing/extra IDs or nonfinite output are rejected.
The active scientific constraint is total separation rank <=4; baseline stage
organization and factor implementation are not additional restrictions.

The original baseline uses only the fast built-in factors. Generic executable
factors may have substantial startup overhead. Run the model probe separately
after a development evaluation. Passing the probe and a data/training provenance
audit is required for acceptance.

The task-specific Research contract in README is binding: output validity alone does not permit substituting a different research object or objective.

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
