# ResCP: Reservoir Conformal Prediction

The task forecasts conformal intervals for 12 fixed cells: four datasets crossed with three frozen point predictors. Minimize the equal-weight normalized Winkler score at alpha 0.1. Each cell must retain coverage at least 0.86 on test and 0.83 on val; a failed cell gives Infinity. The datasets, splits, frozen predictors, normalization denominators, and scientific constraints remain unchanged.

`train.sh --out MODEL` fits calibration artifacts using public `data/train/`. `predict.sh --model MODEL --input REQUEST --output ARTIFACT` exports a private-policy-v2 artifact. The request is a descriptor only. The trusted scorer starts isolated `policy.sh` workers separately for each cell; private future histories and evaluation labels are not mounted into those workers.

The worker receives a cell descriptor, then calls `private_rpc.call('next')`. A prediction event provides the current query coordinate, point forecast and IDs, plus the newly available residual/target/date history with coordinates strictly below the query. Submit an object with `lower` and `upper` arrays (one value per node) using `call('submit', response)` before asking for the next event. The final `next` returns `op=done`. Write a completion file to `--output`. Only intervals accepted by the trusted transcript are scored. Repeated prefetch, missing intervals, nonfinite bounds and incomplete execution fail.

The baseline `online_rescp.py` incrementally computes the original reservoir states and the released CPU sampling rule. It preserves the window+1 rule, draw counts, quantile selection, seeded sampling and eta=0. Current query truth is unavailable until a later query makes it past history. The full permitted history remains available; no history window is shortened by the protocol.

`data/val/` contains the public validation request and replay directory. Use these files only for evaluation and method, hyperparameter or checkpoint selection. Measured baseline on val, three seeds at 128 CPUs: mean 1.0075 and seed standard deviation 0.0056; all cells pass the val coverage floor.

Events contain numeric NumPy arrays over the bounded RPC format. `history_start` specifies the first newly released row and `query` is its exclusive cutoff. `id` is a list; `point` has shape (nodes,1). Candidate code must preserve inherited PYTHONPATH. The authoritative scorer applies the unchanged float32 Winkler arithmetic and the 12 individual coverage gates.

Artifact preparation, worker startup and RPC durations are recorded as diagnostics. A failed coverage gate returns Infinity.


The task-specific research scope in README fixes the scientific object and objective; only the listed solution components may change. Supplied observations, trusted evaluation and information boundaries remain fixed.

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
