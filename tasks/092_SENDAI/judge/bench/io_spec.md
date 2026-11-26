# Sparse-field reconstruction I/O

`--input PATH` is an NPZ archive with `ids` (N unique opaque strings), `sensors`
(float64, N × 5 × 64, NDVI histories ending at the reconstructed frame), and
`sensor_locs` (int64, 64 × 2, zero-based row/column in a 64 × 64 field).
IDs have no temporal meaning. The physical sensor network is fixed.

`--output FILE` must be written as an NPZ archive **to the exact requested filename**,
using an open binary file handle. Exactly four keys are required:

- `ids`: identical to the input, including order.
- `pred`: finite floating-point array of shape (3, N, 64, 64), three independent
  fitted models' full-field NDVI predictions. Do not average the fields together.
- `sensor_locs`: distinct integer row/column pairs actually used by the method.
- `lags`: nonnegative integer scalar, the actual maximum observation history length (0 for no observations). A method may use an empty `(0,2)` sensor list and rely on its learned simulation prior.

`labels.npz` has `ids` and float32 `targets` (N × 64 × 64). Evaluation uses the
original state-scaler float32 roundtrip convention; no clipping is performed.

`data/train/simulation.npz` contains `fields`, raw simulated-season NDVI.
`data/train/observations.npz` contains `sensors` and `sensor_locs`, the sparse
adaptation observations. No real-period full-state training targets are provided.
Validation features and labels are for evaluation and method, hyperparameter or checkpoint selection only.
Fit reusable models and learned preprocessing only on the designated training observations.

Each outer `BENCH_SEED=s` specifies three independent fits with seeds
`3*s`, `3*s+1`, `3*s+2`. Train each from scratch using its assigned seed. Write
`model_card.json` with format `sendai-sparse/v1`, `sensor_locs`, `lags`,
`replicates: 3`, `outer_seed`, and the three `seeds`, in the model directory.
The certificate must describe actual observation use, not just a declared budget.
`bench/check_model.py MODEL_DIR --input PATH --pred FILE` checks it.

The scorer takes the arithmetic mean of the three per-run mean frame SSIMs.
Its guard uses the arithmetic mean of the three per-run global pixel RMSEs,
which must be at most 0.130. The budget is at most 64 sensors and at most 5 lags.
A guard violation has rule value -1; the scorer emits `ssim: Infinity` as the
harness failure sentinel and includes `rule_worst: -1` in its diagnostics.

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
