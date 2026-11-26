# Computational Arbitrage in AI Model Markets

Task statement for the agent. Everything below is binding: the judge reruns your `code/` from the prescribed
starting state with its own seeds, replays your purchase plans on hidden issues, and compares with the archived baseline.

## 1. Goal

An arbitrageur buys repeated attempts from model providers to solve software issues, stopping at the first success. Two providers are on sale, GPT-5 mini and DeepSeek v3.2 Thinking, at up to 1 USD per issue. Decide how much to spend on each, in what order, on each SWE-bench Verified issue.

Improve `buy_cost_usd` (lower is better): the expected cost of reaching a 70% solve rate on hidden issues, scaled to the 500-issue benchmark. The hidden issues appear in neither `data/train/` nor `data/val/`, so a policy must generalize to issues whose outcomes it has not seen.

The baseline is a cascade: GPT-5 mini up to a fixed cap, then DeepSeek, with the cap fitted by mean profit margin. You may replace it with any purchase policy within the Research contract below.

The target is fixed at 70%. A family that cannot reach it has no finite score.

## 2. Metric and guards

`buy_cost_usd`: `bench/scorer.py` replays each level of your family with the paper's pass@k and survival formulas, from attempts and mean costs it holds itself. Each level gives a point (mean solve rate, mean cost per issue). The score is the lower convex envelope of those points at 70%, times 500. `bench/io_spec.md` has the exact formulas.

| Guard | On violation | Enforced by |
|---|---|---|
| Plans buy only `gpt5mini` and `deepseek`, each at most once per plan | Infinity, invalid output | scorer, validate.py |
| At most 1 USD per issue in every plan | Infinity, invalid output | scorer, validate.py |
| One row per (level, issue); consecutive levels, at most 250 | Infinity, invalid output | scorer, validate.py |
| Exactly the three columns; no reported cost, solve rate or other field is read or accepted | Infinity, invalid output | scorer, validate.py |
| Levels are nested: dollars never fall and model order never changes as the level rises | Infinity, invalid output | scorer, validate.py |
| The family reaches a 70% solve rate on the scored issues | Infinity | scorer |
| Fit only on `data/train`; validation is for evaluation and selection only | Disqualified | patch audit |
| No outcomes, gold patches, test lists or per-issue results of benchmark issues from memory or outside sources; no per-issue tables for issues outside `data/train` | Disqualified | patch audit |

Archived baseline on `data/val`: mean `buy_cost_usd` **$42.6929**, standard deviation **$0.0000**, across **1** seed. `bench status` shows the archived value. The judge compares your score with the archived baseline on the hidden issues. PASS needs a relative improvement of at least **10%** (`delta=0.10`).

## Research contract

### Objective

Reduce the expected cost of reaching a 70% solve rate on issues unseen in fitting, scaled to the 500-issue benchmark.

### Fixed research objects

- The market: GPT-5 mini and DeepSeek v3.2 Thinking only, with the authors' released attempts and mean costs.
- The purchase rule: repeated sampling until first success, at most 1 USD per issue, each model at most once per plan, outcomes of different models treated as independent.
- The estimators: unbiased pass@k, `k = floor(dollars / mean_cost)`, expected spend by the survival identity, computed by the scorer.
- The target and scale: 70% solve rate, read from the lower convex envelope of a nested family, cost times 500.
- Inputs at prediction time: only the fields in `input.jsonl`. Outcomes of hidden issues stay hidden.

These objects keep policies comparable in the same market with the same cost and success definitions. The replay formulas are in `bench/io_spec.md`.

### Editable methods

- The purchase policy as a whole. The fixed cap is a baseline choice, not a requirement: a new policy may change per-issue or per-repo allocation, model order, skipping of issues and how levels are built.
- Anything fitted at train stage from `data/train`: predictors of solve probability or cost built on problem statements, repo and version, with any model class and objective.
- Dependencies, as `name==version` pins in `code/env.lock`.

### Forbidden substitutions

Buying any other model, changing the 1 USD total budget cap, estimators, target or scale, or reporting figures instead of plans is not allowed. The data and information-access restrictions in the guards also apply. When retaining the supplied issue-independent cascade, changing only the model order gpt5mini then deepseek, the fitted cap of $0.0812, budget_levels=logspace(1e-4, 1, 250), performance_grid=linspace(0.68, max_perf, 1000) or compute_first=logspace(0.01, 0.5, 100) does not count as a method improvement.

## 5. Data

- `data/train/input.jsonl`: 222 issues with `instance_id`, `repo`, `version`, `problem_statement`.
- `data/train/gpt5mini.jsonl`, `data/train/deepseek.jsonl`: the authors' per-issue `{"id", "mean_cost", "attempts"}` for those issues.
- `data/val/input.jsonl`, `data/val/labels.json`: 88 issues, with the same fields and their outcomes.
- The hidden issues are held by the judge. Their input file has the same fields and no outcomes.

The problem statements come from the public SWE-bench Verified dataset; attempts and costs come from the authors' release. Do not re-split, relabel, or bring in outside data.

## 6. Baseline

Method: the GPT-5 mini then DeepSeek cascade. Official code: https://github.com/RicardoDominguez/computational-arbitrage at `f14d5fdabbddd8fbfa08f1b4fa7b92d54db75881`. The analysis lives in `code/notebooks/utils.py` and `code/notebooks/swebench_gpt5_deepseek.ipynb`.

`code/bench_cascade.py` fits the cap by grid search on `data/train`. For each level with budget `b`, model `i` receives `min(max(b - sum(previous caps), 0), cap_i)`. It uses the notebook's 250-point budget grid. The plans are the same for every issue and ignore the input fields. The fitted cap on `data/train` is $0.0812 on GPT-5 mini.

## 7. Interface

- `bash code/train.sh --out DIR`: fits on `data/train` and writes everything `predict.sh` needs.
- `bash code/predict.sh --model DIR --input PATH --output FILE`: writes the family specified in `bench/io_spec.md`.
- Environment: `BENCH_DATA_DIR`, `BENCH_BENCH_DIR`, `BENCH_CODE_DIR`, `BENCH_CPUS`, `BENCH_SEED`.
- `bench status`, `bench eval-dev`, `bench validate-output FILE`, `bench submit --final` are the development commands.

`data/`, `bench/`, this README, the script signatures and the output format are frozen. New data files cannot enter a patch, so fit at train stage.

## 8. Resources

4 CPUs, 4 GB, no GPU. No network during train or predict, and no API or LLM at any stage. In the archived baseline run on 4 CPUs, training took 6.03 s and prediction on `data/val` took 0.52 s. A candidate has the overall time budget of the task; there is no separate per-stage limit.

## 9. Seeds

The baseline is deterministic, and the judge runs one seed. If your method is random, honor `BENCH_SEED` so a rerun gives the same result.

## 10. Notes

- A budget below an issue's mean cost buys no attempt and is still charged. So is the remainder after the last whole attempt. Mean costs vary widely between issues and are not available for hidden issues.
- An issue that a model never solved in its attempts has pass@k = 0 at any budget, so all the money spent there is lost.
- Attempt counts can be as low as 2, so per-issue pass@k is coarse. Do not read small differences as signal.
- Val is small and noisy near the target. Small differences on val may not generalize.
- Levels only need to be dense near 70%.
- `code/notebooks/swebench_gpt5_deepseek.ipynb` is kept for reference; it will not run here because the authors' data folder is not supplied.

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
