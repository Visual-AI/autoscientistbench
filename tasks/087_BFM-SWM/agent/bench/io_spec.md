# Private policy I/O, version 2

## Outer calls

`data/val/request.json` holds `{"task": "icml-4454-BFM-SWM", "protocol": "private-policy-v2"}` and nothing about instances. `train.sh --out MODEL` prepares the model directory. `predict.sh --model MODEL --input REQUEST --output FILE` seals `code/` and `MODEL` into a ZIP policy artifact through `bench/export_policy.py`; it runs no instance. `validate.py FILE REQUEST` checks the artifact without executing it. A winner table or any other prediction file is rejected.

`scorer.py FILE LABELS` extracts the artifact once and starts one fresh isolated `code/policy.sh` per instance, up to the task's CPUs at a time. Costs stay in the scorer process.

## The worker

Each worker runs `bash code/policy.sh --model /ws/model --input /ws/input --output /ws/output/pred.out` with no network, a private `/tmp`, and read-only `code/`, `model/`, `bench/` and graph files.

`--input` is one JSON object, written with Python's default separators:

```json
{"id": "val0007", "graph": "slashdot", "budget": 400.0, "algorithm_index": 7}
```

- `graph` names a directory under `$BENCH_DATA_DIR` holding `renum_node.txt` (id, weight, degree, group) and `renum_edge.txt` (u, v). Read whichever graph the input names; do not assume it is one you have seen. The worker's `BENCH_DATA_DIR` holds graphs only.
- `budget` is a positive multiple of 0.25, so it is exact in the `float` prices the authors use.
- `algorithm_index` is the instance's position. With `BENCH_SEED` it may seed the policy's randomness; it says nothing about the instance.

The valuation is public: `v(S) = |N(S)|`, the number of distinct out-neighbours of S, with unit node weights. The node table's `weight` and `group` columns are not used by it. A supplier is a node id `0..n-1`.

## The offer service

Reach it through `bench/offer_client.py` (Python) or `bench/offer_client.h` (C++, compile with `-I"$BENCH_BENCH_DIR"`). Both wrap one Unix socket at `$BENCH_RPC_SOCKET`, 4-byte big-endian length plus JSON, one call at a time.

`offer(ids, prices, stop_on_accept)` makes `prices[i]` to supplier `ids[i]`, in order. It returns a string with one `1` (accepted) or `0` (rejected) per offer actually made. A supplier accepts exactly when its private cost is at most the price; the answer carries nothing else. With `stop_on_accept` true, the offers after the first acceptance are not made and are not checked, so the string is shorter. Use it to send prices that hold until somebody accepts, instead of one call per offer.

`declare(winners)` commits the winner set, once. It returns `{"value": v(S), "payment": total payment}`. An empty set is valid and scores zero.

Rules. Breaking any one ends the instance, and the whole evaluation scores `Infinity`, even if the policy catches the error:

- Ids are integers in range and appear at most once per batch. Prices are finite and at least 0.
- The price offered to a supplier never rises.
- A supplier that rejected is out for good: no further offer, and it cannot win.
- A winner must have received at least one offer and never rejected. Its payment is its most recent offer, which it accepted.
- Total payment is at most the budget, and `v(S) - payment` is at least 0, each with a tolerance of `1e-9`.
- Exactly one `declare`, with no calls after it. A worker that exits without declaring, or exits non-zero, fails the evaluation.

There is no other method. `policy.sh` must also write something to `--output`; it is never scored, and only the oracle's transcript counts.

## Labels (public splits)

`data/train/labels/` and `data/val/labels/` each hold `manifest.json`, `costs.npz` (one float64 array per draw, indexed by supplier) and `instances.csv` with columns `id,graph,budget,draw,family,greedy,upper_bound`. `greedy` is the full-information benchmark, the denominator of the metric. `upper_bound` bounds `max v(O)-c(O)` subject to `c(O) <= budget`. Both are frozen numbers. Graph files for the public splits are under `data/graphs/`.

Public cost families:

- `lognormal_sqrtdeg`: `(1 + sqrt(degree)) * LogNormal(0, 0.5)`
- `uniform_10_100`: `Uniform(10, 100)`

You can run `python bench/scorer.py ARTIFACT data/train/labels` directly; `data/train/request.json` is the matching descriptor.

## Scorer output

One JSON line. `welfare_ratio` is the metric; `Infinity` means a rule was broken, and `reason` names it. Other keys are diagnostics: `per_group` by graph and family, `mean_welfare`, `mean_greedy`, `mean_upper_bound_ratio`, `mean_payment_share`, `offers_total`, `rpc_calls_total`, `worker_elapsed_mean_s`, `worker_elapsed_max_s`, `prediction_elapsed_s`, `artifact_prepare_s`, `workers`.

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
