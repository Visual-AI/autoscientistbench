# AutoScientistBench — Evaluation Protocol & Harness Architecture

## 1. Core Protocol

Every task in AutoScientistBench enforces strict data isolation, clean-room reproducibility, and task-specific research contracts:

1. **Workspace isolation**: The agent operates inside a bubblewrap (`bwrap`) sandbox at `/ws` containing only:
   - `/ws/README.md` — task specification, metric, baseline validation score, rules, and research contract.
   - `/ws/code/` — writable baseline implementation (`train.sh`, `predict.sh`, `env.lock`).
   - `/ws/data/` — read-only training and validation data (never the hidden test split).
   - `/ws/bench/` — read-only `validate.py`, `scorer.py`, and `io_spec.md`.
2. **Validation-only selection**: Both the baseline and candidate solutions may fit model parameters and learned preprocessing **only on designated training observations**. Validation data is strictly for evaluation and method/hyperparameter/checkpoint selection — never merged into final retraining.
3. **Explicit patch submission**: The agent evaluates locally with `bench eval-dev` and submits its edits to `code/` via `bench submit [--final]`. Only unified text diffs of `code/` (<= 2 MB, no binaries, no new data files) are accepted; checkpoints are discarded.
4. **Network closure**: Inside the sandbox, direct network access is disabled (`--unshare-net`). Outbound traffic is routed strictly through an operator-side allowlist proxy (permitting only configured model endpoints and PyPI/PyTorch CPU indexes for `venv` tasks). During `bench judge` execution, **all network access is completely disabled**.
5. **Independent clean-room judge**: `bench judge` takes the submitted patch, applies it onto a fresh checkout of `tasks/<pid>/agent/code`, rebuilds the Python environment from the submitted `code/env.lock` (or mounts the immutable runtime snapshot), runs `train.sh` and `predict.sh` from scratch across all configured evaluation seeds, validates predictions with `judge/bench/validate.py`, and scores against `judge/test/` with `judge/bench/scorer.py`.

## 2. Frozen Interface Contract

Every task exposes the exact same four-stage contract:

| Interface | Requirement |
|---|---|
| `bash code/train.sh --out DIR` | Train from the task-prescribed starting state using `$BENCH_DATA_DIR`, `$BENCH_SEED`, and `$BENCH_CPUS`; write inference artifacts to `DIR`. |
| `bash code/predict.sh --model DIR --input PATH --output FILE` | Generate predictions for `PATH` using only `DIR` and write them to `FILE` (which may have no extension). |
| `bench/validate.py <pred> <input>` | Verify schema, alignment keys, row/instance completeness, shapes, and finiteness. |
| `bench/scorer.py <pred> <labels>` | Compute the task metric and task-specific scientific guards, printing a single JSON line `{"<metric>": value, ...}`. |

## 3. Scoring & Leaderboard Metrics

AutoScientistBench reports **two complementary headline metrics** for every evaluated model/agent: **Tasks Passed (`Passed / 104`)** and **Normalized Improvement Score (`%`)**.

### 3.1 Per-Task Relative Gain & `PASS` Criterion

Given the archived baseline held-out test mean $\mu_{\text{ref}}$ and across-seed standard deviation $\sigma_{\text{ref}}$ measured under the exact same protocol (`judge/reference/result.json`), and the candidate held-out test mean $\mu_{\text{cand}}$ across the same seeds:

- **Relative gain** (positive = improvement):
  $$\text{gain} = s \cdot \frac{\mu_{\text{cand}} - \mu_{\text{ref}}}{|\mu_{\text{ref}}|}, \qquad s = \begin{cases} +1 & \text{if higher is better} \\ -1 & \text{if lower is better} \end{cases}$$
- **PASS threshold**:
  $$\tau = \max\!\left(\delta_{\text{task}},\; \frac{2\,\sigma_{\text{ref}}}{|\mu_{\text{ref}}|}\right)$$
  A candidate **passes** task $i$ iff every evaluation seed succeeds, all task-specific scientific guards pass (a non-finite / `Infinity` scorer output indicates a guard violation), and $\text{gain} \ge \tau$.
- **Diagnostic harm floor**: For per-task relative-gain diagnostics (`gain_clipped`), failed builds, timeouts, invalid patches, validation failures, or severe regressions are clipped at `-0.10` (`-10%`).

### 3.2 Normalized Leaderboard Improvement Score (`0%` Baseline → `100%` Target Ceiling)

Because raw relative gains have vastly different natural scales across metrics (e.g., an accuracy task bounded near `1.0` vs. an iteration-count task reducible by `90%`), overall leaderboard ranking uses a calibrated per-task `target_ceiling` ($c_i$, stored in [`benchmark_tasks.json`](benchmark_tasks.json) under `tasks_meta.<task_id>.target_ceiling`):

1. **Per-task Improvement Score (`%`)**:
   - If task $i$ has `status == "scored"` with a finite held-out test mean $x_i = \mu_{\text{cand}, i}$:
     $$\text{score}_i = \max\!\left(0,\; \frac{x_i - b_i}{c_i - b_i} \times 100\right)$$
     where $b_i = \mu_{\text{ref}, i}$ is the reproduced baseline (`0%`) and $c_i$ is the calibrated target ceiling (`100%`). Note that $(x_i - b_i)$ and $(c_i - b_i)$ have the same sign whenever $x_i$ improves over $b_i$, whether higher or lower is better.
   - **Clipping & ceiling policy**: Negative scores (regressions) are clipped to `0%`. Solutions that surpass $c_i$ remain **uncapped above `100%`**.
   - **Unresolved / failed tasks**: Tasks with no valid submission before the deadline, judge timeouts, build failures, validation errors, or guard violations receive $\text{score}_i = 0$.
2. **Aggregate Improvement Score**:
   $$\text{Improvement Score} = \frac{1}{|T|} \sum_{i \in T} \text{score}_i$$
   where $|T| = 104$ for the full benchmark (or $|T| = N_d$ when filtering to a single scientific discipline $d$). Compute both `Passed` and `Improvement Score` automatically via `python3 -m bench report`.

## 4. Research Scope & Guard Classes

Each task's `task.json` (`research_contract` and `validation_policy`) and `agent/README.md` explicitly specify:
- **Fixed research objects**: problem formulation, target observables, data splits, evaluation metrics, and fixed reference objects (e.g., a specified target model being pruned/unlearned or a specified posterior distribution being sampled).
- **Editable components**: algorithms, architectures, feature representations, training surrogates, solvers, and implementations that may be freely replaced or improved.
- **Forbidden substitutions**: shortcuts that alter the research object, tune only trivial baseline constants where disallowed, leak validation/test labels, or bypass real computation.

Guards are classified into three enforcement tiers:
1. **Scorer / validator guards** — automatically checked on every prediction output.
2. **Judge probes** — e.g., subset-consistency / locality checks and private policy replay workers.
3. **Patch audit rules** — enforced by source/diff review of `final.patch`.

## 5. Budgets

- **Agent budget**: wall-clock deadline configured via `bench run --budget-hours H`. Explicit submission before deadline is required.
- **Judge budget**: independent overall wall-clock budget configured via `--judge-budget-hours H` (or `bench judge --budget-hours H`), shared across environment build, all evaluation seeds, validation, and scoring.
- **CPU budget**: fixed per task in `task.json` (`cpus`), enforced via `taskset` and BLAS/OpenMP/Torch thread environment variables.
