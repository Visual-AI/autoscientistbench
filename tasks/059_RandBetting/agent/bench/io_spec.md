# Predictable betting policy interface

## Exported method
`train.sh --out DIR` saves `DIR/betting.json` and any model artifacts. The active
`predict.sh --model DIR --input PATH --output FILE` receives only
`{"protocol":"sta34-private-betting-v2"}` and exports a `private-policy-v2` ZIP
using `private_eval.export_artifact(code_dir,model_dir,output,'Sta-34')`.
The artifact includes editable code, learned model files and pinned dependencies.
Its algorithm seed must equal BENCH_SEED. No private observations are passed to
this export stage.

For arbitrary stateful methods, `betting.json` has exactly:

```json
{"version":2,"execution":"stateful","B":100,"theta":0.5,"trunc_scale":0.5,"combine":"max"}
```

B is a positive integer; theta is in [0,1]; trunc_scale is in (0,1]; combine is
`max` or `mixture`. These values and the global learned model are fixed before
private evaluation. Any candidate architecture, loss, optimizer or history-based
bet-selection algorithm may be implemented in `code/policy.sh` and its helpers.
There is no expression-depth or sufficient-statistic-only restriction on this mode.

The optional certificate fast path uses
`{"version":2,"execution":"certificate","policy":LEGACY_CERTIFICATE}`.
`betting_interface.DEFAULT` supplies the unchanged baseline certificate. This mode
uses the original checked expression representation and frozen arithmetic;
its grammar restrictions do not apply to stateful policies.

## One isolated permutation
The scorer invokes `policy.sh --model /ws/model --input /ws/input --output FILE`
in a new process and private filesystem for every permutation of every sample.
The initial request contains exactly `protocol`, `n`, `null_mean:0.5` and
`alpha:0.05`. The original public horizon n is available. IDs, parameter labels,
shuffle seeds, final U, other permutations and future observations are absent.
Model/code files are read-only; private writable storage is discarded between runs.

Before observing X_t, call:

```python
reply = private_rpc.call('bet', {'positive': lambda_plus, 'negative': lambda_minus})
x = reply['x']
```

Both bets must be finite real numbers computed from the saved model, n and the
already revealed prefix X_1,...,X_(t-1). The candidate can maintain arbitrary state
and access that entire prefix. The trusted side commits and clips both bets before
returning X_t. `reply['done']` is true after the final observation. Exactly n bets
are required; reads without a committed bet, extra calls or malformed bets fail.
Write `{"complete":true}` to FILE after the last response. Capital and rejection
are computed solely by the trusted side.

The clipping interval is [-2*trunc_scale, 2*trunc_scale]. Starting from unit
capital, update Kplus *= 1 + lambda_plus*(X_t-0.5) and
Kminus *= 1 - lambda_minus*(X_t-0.5), using the clipped bets. At theta=1 or 0,
use that side alone; otherwise combine terminal capital as
max(theta*Kplus,(1-theta)*Kminus), or the sum for `mixture`. This preserves the
original nonnegative betting construction and its null expectation bound.

## Randomization and score
The trusted side preserves the original in-place permutation schedule. The base
seed is the uint32 SHA-256-derived value of BENCH_SEED + ':' + opaque ID. For
zero-based sample-size index i and permutation l, shuffle in place after resetting
the NumPy seed to base_seed+i*l modulo 2**32. After averaging all B terminal
capitals, reset to base_seed and draw U. Reject iff e_value > U/0.05. These private
randomization values are never supplied to the stateful worker.

The original data files remain unchanged: NPZ contains id, n and float64 x rows
padded to length 2000; only the first n entries are observations. Public validation
labels contain id,n,x,b,cell,data_seed and baseline diagnostics. Private labels and
complete private observation arrays remain scorer-side.

The metric is equal-cell mean alternative rejection probability. Null type-I error
must be <=0.0695 at every n and <=0.0562 pooled. A valid method exceeding either
null limit receives zero; invalid artifacts or transcripts fail. Fixed mixture
parameters, truthful artifact provenance and absence of memorized answers also
require patch audit.

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
