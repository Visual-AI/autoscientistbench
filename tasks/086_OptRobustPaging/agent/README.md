# Towards Optimal Robustness in Learning-Augmented Paging

Task statement for the agent. Everything below is binding: the judge reruns your `code/` from the prescribed
starting state with its own seeds, replays your policy on hidden traces, and compares with the archived baseline.

## Research contract

### Objective

Decide which cache line to evict, online, on memory-access traces of SPEC CPU 2006 programs.
The cache has 2 MB, 64-byte lines and 16 ways: 2048 sets, k = 16. A policy sees each access only after
all earlier evictions are committed, so it cannot know future accesses. Improve `mean_cost_ratio` (lower is better,
bound 1.0) on the hidden traces.

The hidden traces come from programs that appear in neither `data/train/` nor `data/val/`, so a policy must
generalize across programs. The supplied baseline is the paper's RPB-OM with the POPU predictor at τ = 4. You may
replace it with any online policy within the rules below.

### Metric and guards

`mean_cost_ratio`: per trace, misses divided by the misses of Belady's offline rule run per set, then the unweighted
mean over traces. A miss that fills an empty way counts. The scorer replays the trace, keeps the true cache state, and
counts misses and the optimum itself. A count reported by the candidate never enters the score.

| Guard | On violation | Enforced by |
|---|---|---|
| The victim is a resident line of the missing access's set; every miss loads the requested line (no bypass) | Infinity | scorer |
| Accesses are revealed in order; the next access is released only after the pending eviction is committed | Infinity | scorer |
| The whole trace is replayed and closed with `finish` | Infinity | scorer |
| `--output` equals the trusted record | Infinity | scorer |
| The output of `predict.sh` is a sealed policy artifact, not a file of counts or decisions | invalid | `validate.py` |
| One fresh worker per trace: no network, no trace or data files, nothing survives to the next trace | not possible | isolated worker |
| Models and learned preprocessing are fitted only on `data/train/`, never on `data/val/` | disqualified | patch audit |
| No trace-specific tables for programs outside `data/train/`, whether embedded, recalled or obtained elsewhere | disqualified | patch audit |

The judge scores k seeds on the hidden traces and compares the mean with the archived baseline. PASS means a relative
improvement of at least delta; `bench status` shows both.

### Fixed research objects

These define the problem; a patch that changes them is invalid:

- the cache geometry above, with `line = address >> 6` and `set = line & 2047`;
- the paging model: a miss always loads the requested line, and there is no bypass or prefetch;
- the cost (trusted misses) and the optimum (per-set Belady), and the supplied traces and split;
- causal access with a fresh state per trace: no horizon, trace identity or program name is given;
- `data/`, `bench/`, this README, the artifact format and the script signatures.

### Editable methods

Under `code/`, the whole eviction policy may change. RPB-OM, its budget rule, the OnlineMin priorities and POPU are
baseline choices, not requirements. You may fit a predictor on `data/train/` in `train.sh`, use any online learning
within a trace, including tables shared across the sets of that trace, use `pc` and address history as features, and
add dependencies through `env.lock`.

The score does not measure the paper's worst-case robustness bound, and a candidate need not prove one.

### Forbidden substitutions

Changing any fixed research object; using future accesses through any channel; carrying state between
traces; fitting on validation data, including a final train+val refit or distillation from validation; hard-coding
results or trace-specific tables for programs outside `data/train/`. Do not try to identify the programs: names are
withheld on purpose.

## 5. Data

- `data/train/P*.npz`: full traces of the training programs, about 3.3 million accesses in the largest file and
  6.2 million in total. Arrays `pc`, `addr` (uint64, trace order) and `segment_end`.
- `data/val/input.json` and `data/val/labels/`: validation traces, with `labels/manifest.json` and one
  `V*.npz` per trace (`pc`, `addr`). Use them to evaluate and select, not to fit.
- Hidden traces use the same format. They are available only through trusted evaluation
  and are never readable from this workspace.

Program behavior varies widely, and three validation traces cover
it only thinly. Do not expect a gain measured on validation to carry over in size.

## 6. Baseline

- Method: RPB-OM with POPU, τ = 4, `max_support_factor` = 100. Official code:
  https://github.com/Natureal/ICML-Cache-Coliseum at commit `7f0299c12ff1bdd02ba28f3f98a98c28ccbd8478`, under `code/`.
  The algorithm is `cache/evict/algorithms.py`, class `PredictiveRPBOnlineMinAlgorithm`; the predictor is
  `cache/evict/predictor.py`, class `POPUPredictor`.
- `train.sh` calls `train_policy.py` to write the configuration; nothing is fitted. `predict.sh` seals the policy.
  `policy.sh` calls `policy.py`, which drives one algorithm instance per set through `paging_env`.
  POPU estimates next-arrival times from the page frequencies observed so far within that trace.
- On validation, seeds 1, 2 and 3: `mean_cost_ratio` 1.13565, 1.13558, 1.13565
  (mean 1.13562457, standard deviation 0.00003494 across seeds).
- For those seeds in order, measured training times were 0.21, 0.21 and 0.21 s; validation artifact sealing took
  0.61, 0.51 and 0.52 s. Replaying all validation traces took 20.57, 20.23 and 20.12 s with 3 workers.

## 7. Interface

- `bash code/train.sh --out DIR` writes everything the policy needs into DIR. The baseline writes `config.json`.
- `bash code/predict.sh --model DIR --input PATH --output FILE` seals `code/` and DIR into a policy artifact.
- `bash code/policy.sh --model DIR --input FILE --output FILE` is started by the scorer once per trace.
- In the worker, `paging_env.step(victim)` returns the accesses up to and including the next miss on a full set, and
  `paging_env.finish()` closes the replay. The full protocol is in `bench/io_spec.md`.
- Environment: `BENCH_SEED`, `BENCH_CPUS` (1 in a worker), `BENCH_DATA_DIR` (`data/`, train stage only),
  `BENCH_BENCH_DIR` (`bench/`), `BENCH_MODEL_DIR` (worker).
- `bench status` · `bench eval-dev [--seed S]` (train, seal, validate, replay on validation) ·
  `bench validate-output FILE` · `bench submit [--final] [-m MSG]`.
- Dependencies: `uv pip install ...` into the venv; `bench submit` freezes them into `code/env.lock`. Only `name==version`
  pins from PyPI are rebuildable.

## 8. Resources

4 CPUs and 8 GB of memory. Up to four single-CPU workers replay traces in parallel. No network during train,
predict or replay. Use the overall deadline as you see fit: there is no separate stage limit, and evaluation has its
own overall budget, so a very slow policy can exhaust it. Submit a valid patch with `bench submit` before the deadline;
no valid submission means failure.

## 9. Seeds

The judge runs k seeds and every run must honor `BENCH_SEED`. RPB-OM is randomized: the baseline seeds Python's
`random` from `BENCH_SEED` in each worker and is exactly reproducible for a given seed. Variance across seeds counts
against you.

## 10. Notes

- Keep your resident lines exactly in step with the trusted state; a victim that is not resident scores Infinity.
- A worker restarts for each trace, so keep its start-up light. `cache.evict` needs only numpy; `benchmark` and
  `model` import torch and tensorflow, which are not in `env.lock`.
- The authors' scripts under `scripts/` expect CSV traces under `traces/`, which are not shipped. Load `data/train/*.npz`
  with numpy instead.
- There is one RPC round trip per eviction, so hits cost nothing and most of the time is your policy's.
- A copy of a validation trace in your model can reach 1.000 on validation. That is fitting on validation, and it
  tells you nothing about the hidden traces.

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
