# Private sequential CAST I/O (private-policy-v2)

Prediction exports a policy artifact; an order CSV is not accepted as a
submission.
The public outer request is a JSON task/protocol descriptor with no market prices.

## Data and worker events

Fit calibration parameters and learned preprocessing only on train/input.csv.
Use val/input.csv and val/labels.npz only for evaluation and method,
hyperparameter or checkpoint selection. Online state updates may use the
observations already released in the current panel; do not carry fitted state
from validation cases into other independent cases. The worker receives no full
price table, label file or future path.
It runs `policy.sh --model DIR --input FILE --output FILE` and uses frozen
`private_rpc.call`:

- `call('next')` first returns `{"op":"start","panel":name}`. Reply with
  `call('submit', {"started":name})`.
- Each subsequent event is `{"op":"step","id":id,"step":k,
  "prices":[30 positive prices],"active":boolean}`. Reply with
  `call('submit', {"id":id,"orders":[30 finite share changes]})` on an active
  decision; use `orders:null` on warm-up or settlement observations.
- `{"op":"done"}` ends the panel. Write a completion file to --output.

The runner starts a separate isolated worker for each of NASDAQ, CSI300, TPX100
and Global30. A worker may retain that panel's preceding observations and state.
The next close is released only after the current reply is irrevocably recorded.
Calling next twice, changing a reply's id, omitting an event or supplying extra
orders is a protocol failure. The last close settles the previous order only.

## Artifact and score

The validator checks bounded archive structure, safe paths and task identity
without executing candidate code. The scorer receives `ARTIFACT REPLAY_DIR`, whose
input.csv and labels.npz preserve the original observations, accounting and frozen
risk baselines. It writes its own order table from accepted RPC responses and then
uses the unchanged accounting.py; candidate files are not authoritative orders.

The original rules remain: independent accounts, initial cash 1000, next-close
fills, no transfers/costs, decision-close no leverage, permanent stop below NAV
200, and positive NAV. Score is mean four-panel Sharpe with sqrt(252), simple
returns and sample standard deviation. MDD must not exceed the frozen panel
baseline plus 0.01; turnover must not exceed twice frozen baseline. Terminal
settlement is excluded from turnover. Zero-variance returns fail.

A failed transcript or numerical guard yields Infinity. The result reports
causality_guard_ok for successful sequential execution, independently of numerical
guards. The no-lookahead boundary is enforced automatically by the scorer. The controller, forecasting model and calibration procedure may be replaced while obeying the same observation, accounting and risk constraints.

The allocation is up to 16 CPUs and 16 GB. Formal market data, scoring and accounting remain fixed. Algorithm-seed compliance requires source review.

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
