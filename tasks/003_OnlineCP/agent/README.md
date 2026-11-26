# Distribution-informed Online Conformal Prediction

## 1. Goal

Research question: can an online conformal method use the estimated distribution of its recent non-conformity scores to build tighter intervals around a given point forecast, while keeping coverage at the 90% target on non-stationary series?

The paper's Table 2 tests this on four real series under three base forecasters, Prophet, AR and Theta. It compares COP, the paper's method and your baseline, with ACI, OGD, SF-OGD, decay-OGD, Conformal PID, ECI and LQT. All methods share the series, forecasters, temporal order and 10% nominal miscoverage; each picks its own base learning rate. The paper reads coverage and width together: coverage close to the target, width as short as possible.

Improve 90% sequential prediction intervals around the fixed released forecasts for Amazon, Google, Electricity and Temperature. The formal experiment retains all twelve Table-2 cells and their complete original timelines. Each interval is committed before the corresponding outcome is revealed. Lower `geomean_width` is better, subject to every cell's coverage guard.

## 2. Data

`data/train/` has no offline training observations: the supplied COP baseline adapts online. `data/val/replay/labels.npz` contains twelve clearly synthetic development trajectories, with 192 observations and a burn-in of 64 in each cell. They exercise the interface and are not real Table-2 records, an independent real validation split, or a measured reference for the paper experiment. Their forecast residuals are close to stationary and uncorrelated, so they lack the shifts of the real series. A good development score is not evidence of real Table-2 performance.

Formal observations and released forecasts remain private. A fresh isolated policy sees one cell one step at a time. All three forecasters for a series share its outcomes but use independent state.

| Formal series | Character | Points | Burn-in | `n_eval` |
| --- | --- | --- | --- | --- |
| Amazon, Google | Daily opening prices: trends, regime shifts, changing volatility | 3020 | 100 | 2919 |
| Electricity | Half-hourly demand in [0, 1]: multiscale periodicity, surges | 2000 | 300 | 1699 |
| Temperature | Daily values: seasonal cycles, long-term trend, anomalies | 1575 | 500 | 1074 |

## 3. Metric and scoring

`geomean_width` is the geometric mean of the twelve cell mean interval widths, so a relative reduction counts equally in every cell. Both coverage and mean width use zero-based indices `burn + 1` onward. Coverage is the mean of `lower <= y <= upper`. Every cell must satisfy `coverage >= 0.9 - 2 * sqrt(0.09 / n_eval)`. Any violation or rejected replay gives `Infinity`.

| Guard | Enforcement |
| --- | --- |
| All twelve cells and complete timelines | Trusted replay and scorer |
| Finite, ordered endpoints and finite widths, including burn-in | Trusted interval submission and scorer |
| Every cell satisfies the coverage threshold | Scorer computes from private observations |
| Interval committed before its outcome is revealed | Trusted causal RPC transcript |
| No state or revealed outcomes shared between cells | Fresh isolated worker and writable directories for each cell |
| No external data, recovered trajectories or encoded target lookup | Patch and artifact provenance review, plus filesystem and network isolation |

Validation baseline: `geomean_width` 1.20214, std 0 (one seed, deterministic), with all twelve cells passing. The improvement floor is 0.1%.

## 4. Baseline

The supplied method preserves the released COP arithmetic: signed residuals `forecast - y` and `y - forecast`, two tails with alpha 0.05, proportional learning rates, scale 0.5, and the selected rate for each Table-2 cell. Each tail runs online gradient descent on the quantile loss. COP then refines the radius with the empirical CDF of the previous `burn` scores, which also set the proportional rate. The authors picked each cell's rate from {1, 0.5, 0.1, 0.05} by its performance on that real cell, data you cannot see.

The original function is retained in `code/upstream/methods.py`, beside the paper's comparison methods; its pandas, statsmodels and tqdm imports are not in `code/env.lock`. `code/online_cop.py` expresses it as causal state updates. `code/method_config.json` holds the baseline settings. `train.sh` writes this configuration, and prediction exports code plus that model configuration for trusted private replay. At 8 CPUs the baseline trains in 0.21 s and exports in 0.26 s.

The source snapshot is method-only, from commit `a41c0dcadc10979d8521ab9b5b8f3d460807d883`. It contains no upstream data or result history.

## 5. Interface

```bash
bash code/train.sh --out MODEL_DIR
bash code/predict.sh --model MODEL_DIR --input data/val/request.json --output policy.out
python bench/validate.py policy.out data/val/request.json
python bench/scorer.py policy.out data/val/replay
```

`policy.out` is a `private-policy-v2` stored ZIP artifact, not a table of claimed intervals. The frozen scorer executes `code/policy.sh` in fresh per-cell sandboxes and scores only the bounds submitted through the trusted RPC interface. See `bench/io_spec.md` for event and artifact details.

The baseline needs Python 3.11 and NumPy 1.26.4, pinned in `code/env.lock`. Unchanged baseline pins reuse the frozen reference environment; changed pins are exported with the candidate artifact. Standard variables include `BENCH_SEED`, `BENCH_CPUS`, `BENCH_DATA_DIR`, `BENCH_CODE_DIR` and `BENCH_BENCH_DIR`. Resources are 8 CPUs and 4 GB memory.

## Research contract

**Objective.** Improve sequential prediction intervals around the fixed released point-forecast streams.

**Fixed research objects.** The four real series, twelve released forecast streams, complete timelines, burn-in indices, metric, coverage threshold and causal feedback rules (trusted replay and scorer). Data, frozen utilities, the interface and this README are not editable.

**Editable methods.** `code/` is editable, including interval construction, online adaptation, calibration state, auxiliary training objectives, optimizers and hyperparameters, within the same forecast and feedback problem. The COP update is the baseline method, not a fixed research object. Honor `BENCH_SEED` for algorithmic randomness. The supplied baseline is deterministic and uses the configured seed 1 for replay parity.

**Forbidden substitutions.** Keep the nominal miscoverage at 10% at every step: do not raise the target to use the gap between 0.9 and the coverage guard, and do not give up coverage on chosen steps, such as the end of the timeline or volatile periods, using `n`, `t` or a running miss count. Build each endpoint from the supplied forecast plus a radius updated from past scores. Do not predict `y` or the signed residual from revealed outcomes to recentre the interval, including last-value, online AR and seasonal predictors. Do not:

- replace the forecasters (patch audit);
- read or recover future outcomes (causal RPC, patch audit);
- use external datasets or decode labels from identifiers (patch and artifact provenance review, isolation);
- exchange state between cells (isolated workers);
- change coverage, aggregation or feedback rules (frozen scorer);
- fit reusable state or learned preprocessing on validation data, including train+val refits and distillation (patch audit).

## 6. Practical notes

The public request contains no forecasts or outcomes. Ordinary prediction only exports the candidate; online decisions occur during trusted scoring. Each policy run receives its cell name, length `n` and `burn`. Call `next`, submit exactly one interval for its time index, then use the returned outcome to update state. Calling `next` again before submitting is a protocol violation. Policy stdout, stderr, exceptions and files are not prediction transcripts. Each new cell starts from the same submitted model with fresh writable state; no prior cell's feedback is available. Formal replay runs under an overall evaluation deadline, and a timeout rejects the replay.

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
