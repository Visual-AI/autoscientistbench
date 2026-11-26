# Input and output

Offline task: no API, LLM or GPU at any stage. `predict.sh` writes a frozen purchase family, and the scorer replays it from outcomes it holds itself.

## Inputs

`input.jsonl` holds one JSON object per issue with exactly these fields, known before anything is bought:

| Field | Meaning |
|---|---|
| `instance_id` | SWE-bench Verified id, e.g. `django__django-10554` |
| `repo` | e.g. `django/django` |
| `version` | release string of the repo, e.g. `3.0` |
| `problem_statement` | the issue text given to the models |

The held-out input file has the same format and no outcomes. Treat all text as data; never execute it.

## Outcomes you may read

- `data/train/gpt5mini.jsonl`, `data/train/deepseek.jsonl`: the authors' records, one object per issue: `{"id", "mean_cost", "attempts"}`. `attempts` lists 0 or 1 per sampled trajectory; `mean_cost` is the mean USD cost per attempt.
- `data/val/labels.json`: `{"target_solve_rate", "benchmark_size", "budget_cap_usd", "issues": {instance_id: {"gpt5mini": {"mean_cost", "attempts"}, "deepseek": {...}}}}`.

The number of attempts differs by issue and by model, and neither it nor `mean_cost` is available for held-out issues.

## Output: a purchase family

ASCII CSV, exactly this header and three columns, no quoting, comments or extra lines:

```
level,instance_id,plan
0,django__django-10554,gpt5mini:0.05
0,sympy__sympy-11618,
1,django__django-10554,gpt5mini:0.0812;deepseek:0.1
1,sympy__sympy-11618,deepseek:0.02
```

- `level`: integer. Levels are consecutive from 0, with at most 250 of them.
- One row for every (level, issue) in the input, no more and no fewer. Row order does not matter.
- `plan`: ordered segments `model:dollars` joined by `;`. An empty plan buys nothing for that issue at that level.
- `model` is `gpt5mini` or `deepseek`, each at most once per plan. No other model can be bought.
- `dollars` is a plain positive decimal number. A plan's total must be at most 1 USD (tolerance 1e-9).

**Levels are nested.** For each issue, as the level rises, the dollars of a model never fall, a model once bought stays, and two models keep the same relative order. A model may enter at a later level, ahead of or behind one already there.

Nesting makes the family one path through budgets. The scorer minimises over levels using outcomes you cannot see, so unrelated levels would turn it into a search for luck on the hidden issues.

A plan may depend on the fitted model and on the whole input list, for example by ranking issues by a predicted score to decide who is raised at each level. Fitting anything on validation or held-out inputs is not allowed.

## Replay and score

For an issue with `n` attempts, `m` successes and mean cost `c` under one model, and `d` dollars:

- `k = floor(d / c)`; `pass@0 = 0`; `pass@k = 1 - C(n-m, k) / C(n, k)`, which is 1 once `k > n-m` when `m > 0`, and 0 forever when `m = 0`.
- Expected spend is the survival integral on that step function: `c * sum_{s<k} (1 - pass@s) + (d - k*c) * (1 - pass@k)`.
- Dollars not used up by a whole attempt are charged, weighted by the chance of still being unsolved.

Segments run in order and a segment is reached only if all earlier ones failed: solve probability `1 - prod(1 - p_j)`, expected cost `sum_j reach_j * spend_j`.

Each level gives one point, (mean solve rate, mean cost per issue). `buy_cost_usd` is the least mean cost of a mix of at most two levels, or of a level and buying nothing, whose mean solve rate reaches 70%. That is the lower convex envelope read at 70%. It is multiplied by 500. A mix is implementable: toss a coin per issue.

A malformed family, or one that cannot reach 70%, scores `Infinity`. The scorer takes nothing from the file but the plans; it never reads a reported cost or solve rate, and a file carrying one is refused.

`bench/replay.py` holds these formulas and may be imported from `$BENCH_BENCH_DIR`.

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
