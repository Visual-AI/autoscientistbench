# I/O specification

Input is a CSV with exact header `id,s1,s2`, one positive material coefficient pair per row.
`id` is the original integer row identifier. Training input is `data/train/input.csv`;
training and validation labels have header `id,s1,s2,J_ref` and contain the authors'
high resolution reference Ritz energy. Coefficients are not normalized.

Predictions are CSV with this exact header:

```text
id,x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,y0,y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,train_s,inference_s,fem_solves
```

Keep the input row order and ids exactly; no duplicate, missing or extra rows.
All numeric fields must be finite. Each axis has exactly eleven strictly increasing
nodes in [0,1], including exactly 0, 0.5 and 1. Endpoints must be serialized exactly.
The tensor grid with the bottom-right quadrant removed defines the L-shaped domain.
Do not report energies: the frozen scorer computes them from nodes and coefficients.

`train_s` is nonnegative elapsed wall time for the entire training stage, including initialization and compilation; repeat it on all rows. `inference_s` is nonnegative elapsed time for this row's prediction. `fem_solves` must be exactly zero: prediction applies the trained parameter-to-mesh rule without a fresh PDE/FEM solve or instance-specific mesh optimization. Training-time FEM/optimization on permitted data is allowed. The evaluator independently computes FEM energy from the submitted mesh. Timing fields remain honest diagnostics; actual information provenance and the absence of undeclared prediction-time solves require source review.

The baseline also writes `model_card.json` with format `rfem-run/v1`, seed, sample_seed,
steps, train_s and weights_sha256 to bind its timing record to its saved parameters.
Alternative models may change their serialization while retaining honest runtime fields.

`scorer.py PRED LABELS` returns one JSON line with `mean_relative_ritz_error` (lower
is better). For admissible predictions this is mean sqrt(abs(J_ref-J_mesh)/abs(J_ref)).
A violation returns `score_rule_value: 1.0` and an Infinity main metric, which signals
an invalid submission to the harness. `validate.py PRED INPUT` exits nonzero for an
invalid schema, row set/order, nonfinite entry, inadmissible mesh or invalid diagnostic counters.

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
