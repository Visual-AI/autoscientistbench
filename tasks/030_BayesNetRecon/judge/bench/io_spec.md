# Private sequential forecasting I/O (private-policy-v2)

The outer commands are train, predict, validate and scorer. `predict.sh`
exports a policy artifact; the scorer runs that policy and records forecasts itself.
A submitted forecast CSV is not a valid submission.

## Public data and outer input

`data/train/weekly.csv` and `pstar.csv` are the designated training observations.
`data/val/input.csv` and `labels.csv` are public validation tables with calibration
history. Use validation features and labels only for evaluation and method,
hyperparameter or checkpoint selection. Online calibration uses observe events
released earlier in the same replay, as specified below. The outer `--input` is
`request.json`, containing only task identity and protocol version. It contains no
hidden weekly observations, future bank counts or target links.

## Policy worker

The artifact includes the candidate code, model and declared runtime dependencies.
The isolated worker runs `bash policy.sh --model DIR --input FILE --output FILE`.
`policy.sh` may be changed under code/. It calls `private_rpc.call` from the frozen
bench directory. No private dataset or label directory is mounted into this worker.

`call('next')` returns exactly one event:

- `{"op":"observe","row":{"id":week,"N":banks,"L":links,"V":potential}}`.
  Acknowledge with `call('submit', {"observed": week})`.
- `{"op":"forecast","target":{"id":week,"N":banks,"V":potential}}`.
  Return `call('submit', {"id":week,"alpha":positive_number,"beta":positive_number})`.
  An optional `p` must match the trusted posterior probability; free `pred` values are rejected.
- `{"op":"done"}` ends the stream. Write a completion file to the requested output.

A second `next` before submitting the current reply is rejected. Each target's L
is released in an observe event only after that target's forecast is irrevocably
recorded. Later N/V/observations are never supplied early. The policy can retain
all legitimate preceding observations; there is no imposed lookback truncation.

## Artifact validation and score

`validate.py ARTIFACT REQUEST` checks archive structure, safe paths, task identity
and bounded size without executing code. The authoritative scorer takes
`ARTIFACT PRIVATE_REPLAY_DIR`, where the directory holds the original chronological
`input.csv` and target `labels.csv`. The public development replay directory uses
exactly the same structure. Hidden replay data remain outside candidate mounts.

For every requested week the policy supplies positive finite Beta prior parameters
calibrated from legally preceding observations. The trusted scorer takes the last
observed `L` and pair count `M=N*(N-1)/2`, computes the posterior mean
`(alpha+L)/((alpha+L)+(beta+M-L))`, clips it to `[1e-9,1-1e-9]` as in the original
solver, and multiplies by the requested week's supplied `V`. It uses zero when
the preceding node set has no possible pair, matching the original solver.
The optional claimed `p` is checked to numerical tolerance. Thus the trusted
transcript contains one model-consistent forecast per week in chronological order.
The metric is mean `abs(pred-y)/y`, with each y positive. Output includes `re_l`, `n` and `causality_guard_ok`. A malformed,
incomplete, out-of-order or prefetching policy fails closed with `Infinity`.
The scorer's own recorded decisions are authoritative; candidate completion files
or self-reported counters cannot substitute for them.

The BERM shared-edge model, Beta prior family, posterior-predictive mean, scientific metric and chronological observations are fixed. Hyperparameter calibration/window selection and numerical estimation may change. The prior must be fitted to the distribution of past weekly density estimates L/V. Do not choose alpha and beta by minimizing forecast error, scale their sum by a free concentration factor, or use fixed or near-zero values that remove the prior or the observed week from the posterior. The certificate proves posterior arithmetic only: Beta parameters must actually be calibrated from permitted history, not reverse-solved from desired predictions. Prior provenance, algorithm-seed compliance and these semantics remain subject to code review.

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
