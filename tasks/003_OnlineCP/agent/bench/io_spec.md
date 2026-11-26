# Causal Table-2 interface

The ordinary four calls are `train.sh --out DIR`,
`predict.sh --model DIR --input PATH --output FILE`,
`validate.py PRED INPUT`, and `scorer.py PRED LABELS`.
The public input JSON is exactly
`{"task":"030_Conformal-Optimistic-Prediction","protocol":"private-policy-v2"}`.
`LABELS` is a replay directory containing the frozen `labels.npz`.

Training may prepare method configuration; no reusable state may be fitted on validation data.
The baseline has no offline fit. Prediction packages candidate code and model
using `private_eval.export_artifact`. The output is a stored, unencrypted ZIP
with `manifest.json`, `code/policy.sh`, code, model, and any exported dependencies.
The frozen validator checks task/protocol identity, archive safety and entry
points without executing the candidate. Duplicate or unsafe paths, symlinks,
compressed members and malformed manifests are rejected. Exact baseline lock
equality permits frozen environment reuse; changed pins remain exportable.

The scorer holds each complete forecast and outcome trajectory. For each cell
it launches a fresh isolated `policy.sh --model DIR --input FILE --output FILE`.
The input descriptor identifies the task, protocol, cell, trajectory length `n`
and burn-in `burn`. The worker receives no trajectory files. Only the current
forecast and outcomes from its own completed submissions become accessible.
`BENCH_PRIVATE_EVAL=1` identifies this phase; import `call` from the frozen
`private_rpc` utility. The baseline interaction is:

```python
event = call('next')
# event: {"op": "predict", "t": 0, "forecast": 1.25}
feedback = call('submit', {"t": event['t'], "lower": lower, "upper": upper})
# feedback: {"op": "feedback", "t": 0, "y": observation}
```

`next` returns `{"op":"done"}` after the entire timeline. Each event requires
exactly one submission before another `next`. Bounds must be finite numeric
scalars, ordered, and have finite difference. The integer time index must match
the current event. Submission is recorded before its outcome is returned.
Incomplete transcripts, unsupported calls and malformed responses fail scoring.
The worker writes a completion file at its output path; its contents do not
replace the trusted interval transcript.

Every cell runs with separate process, output and temporary state, read-only
code/model/utilities, and network isolation. Outcomes shared by different
forecasters cannot pass between workers. Candidate output and exception text
are discarded; replay failures return a fixed failure record.

The scorer prints one JSON object containing `geomean_width`,
`all_constraints_pass` and causal verification status. Successful replay also
reports per-cell width, coverage, evaluation count and coverage threshold.
The score is the geometric mean of all twelve post-burn-in mean widths if every
cell has coverage at least `0.9 - 2*sqrt(0.09/n_eval)` and every interval width
is finite; otherwise it is `Infinity`. Both coverage and width start at index
`burn + 1`. The formal replay retains the original full timelines and burn-in.
Public development has twelve independent synthetic examples and is explicitly
identified by `synthetic_development=true`; its score is a diagnostic only.

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
