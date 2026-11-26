# AutoResearch task: {{pid}}

You are working inside an isolated workspace at `/ws`. Your job is to **improve the method** of this task so that it
scores better on the hidden test split, within a wall-clock budget of **{{budget_h}} h** (hard deadline; run `bench status`
to see the remaining time). You have **{{cpus}} CPU cores** and no GPU.

## What is in the workspace

- `/ws/README.md` — the task statement (metric, data, protocol, baseline), reproduced in full below.
- `/ws/code/` — the full baseline implementation (a git repository; its HEAD is the baseline). You may improve or
  replace the whole method within the editable scope specified by the task statement below. Model architecture, loss,
  optimizer, hyperparameters and preprocessing are editable only when the task permits those changes.
  Preserve the optimization objective, fixed research objects, supplied inputs and splits, evaluation metrics,
  experimental constraints and allowed information access. A specified model being pruned, explained or unlearned,
  or a specified objective function being optimized, can itself be a fixed research object. Do not substitute a
  different object or an easier problem. Distinguish required experimental controls from baseline implementation choices.
  `code/train.sh` and `code/predict.sh` are the two entry points the judge calls.
- `/ws/data/` — train and validation data (read-only). The test split is **not** here and cannot be reached.
  Validation is for evaluation and method/hyperparameter/checkpoint selection only. Neither the baseline nor
  a candidate may fit model parameters or learned preprocessing on validation features or labels, including
  during final retraining. Never merge validation into training or use it for distillation or label lookup.
  Task-defined per-instance inference and causal feedback must follow the stated observation roles; they do
  not permit fitting a reusable model on validation cases or carrying state between independent cases.
- `/ws/bench/` — read-only: `scorer.py` (the exact metric the judge uses), `validate.py` (output-format checker),
  `io_spec.md` (input/output formats).
- The task runtime is available. For Python package environments, `uv pip install ...` can install packages and
  `bench submit` freezes `code/env.lock`. Tasks with a supplied runtime snapshot use the environment described below.
- **Scratch space.** `/ws` (except `data/` and `bench/`), `/tmp` and `$HOME` are writable and on a local disk, so keep
  experiments, logs and intermediate files there, for example in `/ws/.work` or `/tmp`. Only `code/` is submitted;
  nothing else reaches the judge.
- **Network.** There is no internet access, even if the tool's own environment notes say otherwise. Only the Python
  package indexes are reachable, so `uv pip install <name>==<version>` works; other downloads, repositories, datasets
  and pretrained weights are blocked. The judge runs `train.sh` and `predict.sh` with no network at all, so anything
  they need must be in `code/` or pinned in `code/env.lock`.

## Rules (the judge enforces them)

1. **Frozen interface.** The judge runs exactly
   `bash code/train.sh --out <dir>` then `bash code/predict.sh --model <dir> --input <path> --output <file>`,
   with the environment variables `BENCH_SEED` (use it to seed everything) and `BENCH_CPUS`. The prediction file
   must pass `bench/validate.py` (see `io_spec.md`). Do not change these signatures or the output format.
2. **Frozen scorer and data.** `bench/` and `data/` cannot be modified (they are read-only and the judge uses its own
   copies). Do not hard-code outputs, do not look for test data: there is none here.
3. **Reproducible execution.** The judge reruns your entry points from a clean checkout with its own seeds on the
   same hardware class, using the task-prescribed starting state, including any fixed pretrained model.
   Keep anything the training needs inside `code/` (patch limit 2 MB, no binaries, no data files).
4. **Submit a patch, not a model.** `bench submit --final` records a diff of `code/` against the baseline commit
   (new files included) plus the frozen `code/env.lock`. The judge applies it to a clean copy, rebuilds the env,
   runs the entry points and scores the hidden test outputs against the baseline executed under the same protocol.

## Workflow that works

- `bench eval-dev` runs the *same* train → predict → validate → score pipeline on the validation split and
  prints the metric; the baseline gets **{{baseline_val}}** there (metric `{{metric}}`, {{direction}}).
  You pass if the test metric improves by more than **{{delta_pct}} %** relative to the baseline (and more than the
  seed-to-seed noise). Changes that help on val but not on test do not count, so prefer robust improvements.
- Iterate: edit `code/`, `bench eval-dev`, repeat. Submit early with `bench submit` (you may submit many times; the
  judge takes the last valid `--final`, else the last valid submission).
  **If no valid submission is recorded before the time budget expires, the run is a failure.**
- **Use the whole budget to get the best result you can.** Do not stop at your first improvement: record a valid
  submission early as insurance, then keep going — try further ideas, check them with `bench eval-dev`, and resubmit
  whenever one is better, since only your last valid submission counts. Spending the remaining time on another
  attempt costs you nothing, so keep improving for as long as the {{budget_h}} h allow, keeping back only the margin
  you need to record the final submission and leave the workspace clean.
- Leave the workspace in a state where `code/train.sh` and `code/predict.sh` work from a clean checkout using the prescribed inputs and starting state.

---

{{readme}}
