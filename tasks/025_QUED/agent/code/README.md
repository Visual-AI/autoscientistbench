# QUED source snapshot and adapter

`qued/` retains the selected upstream descriptor implementations under their MIT license.
The benchmark supplies their frozen BOB+DQM values; the runtime does not import these
source files or receive raw molecular geometries. The adapter reconstructs a fresh
XGBoost fit from the published model's stored hyperparameters. The original fitting
history was not supplied, and no trained estimator or scaler is included.

`train.sh` writes a training recipe. `predict.sh` exports source plus that recipe.
The trusted evaluator invokes `policy.sh` in separate fit and prediction workers;
`main.py` fits on training inputs, exports inert fitted state, then predicts in a
fresh worker that receives evaluation features. Edit the regression method, loss,
optimization and internal preprocessing while preserving the fixed input/output
contract and using all authorized training observations for the formal fit.
