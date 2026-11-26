# Paging policy I/O, private-policy-v2

The scorer holds every evaluated trace, keeps the true cache contents and counts misses itself. A candidate
submits a sealed policy and never receives a trace file for evaluation.

## Cache model

2 MB capacity, 64-byte lines, 16 ways: 2048 sets, k = 16. For a byte address, `line = address >> 6` and
`set = line & 2047`. A line is the page of the paper's paging model. A miss always loads the requested line
(no bypass, no prefetch); when its set is full, one resident line of that set must be evicted first. The cost
is the number of misses, including misses that fill an empty way.

## Data

- `data/train/P*.npz`: full traces of the training programs. Arrays `pc` and `addr` (uint64, trace order) and
  `segment_end` (int64): the row where each of two consecutive segments ends. `data/train/manifest.json`
  lists the sizes. Fitting on these files is allowed.
- `data/val/input.json`: the public request, `{"task": ..., "protocol": "private-policy-v2"}`.
- `data/val/labels/`: the validation container: `manifest.json` (`task`, `protocol`, `traces`) and one
  `<id>.npz` per trace with arrays `pc` and `addr`. For evaluation and selection only.
- Program names are withheld. The hidden split has the same container format and comes from other programs.

## Outer commands

- `train.sh --out DIR` writes everything the policy needs into DIR. It may read `$BENCH_DATA_DIR/train`.
- `predict.sh --model DIR --input REQUEST --output FILE` seals `code/` and DIR into a ZIP artifact through the
  frozen `bench/export_policy.py`. It reads no trace.
- `validate.py FILE REQUEST` checks the artifact without running it. Anything else, such as a file of claimed
  counts, is rejected.
- `scorer.py FILE LABELS` replays the artifact and prints one JSON line.

## Worker

The scorer starts `code/policy.sh --model DIR --input FILE --output FILE` once per trace, in a fresh isolated
worker: no network, no trace or data files, `DIR` read-only, a private `/tmp` that does not survive the worker,
one CPU. `--input` holds only the public request, so the worker is not told which trace it is replaying or how
long it is. `BENCH_SEED`, `BENCH_CPUS`, `BENCH_MODEL_DIR` and `BENCH_BENCH_DIR` are set. Keep the inherited
`PYTHONPATH`, which provides the packaged dependencies and `bench/`.

Import `paging_env` from `bench/`:

- `paging_env.step(victim=None)` returns a `Batch` with `pcs`, `addresses`, `needs_victim`, `done`. It delivers
  every access up to and including the next miss on a full set. Earlier accesses in the batch were hits or
  fills of an empty way, so they required no decision.
- When `needs_victim` is true, the next `step` must carry the victim: `paging_env.line_of(address)` of a line
  resident in the set of the last access. The next access is released only after that. Otherwise pass `None`.
- `done` is true when the trace is exhausted and no decision is pending. Then call `paging_env.finish()` for
  the trusted `{"accesses", "misses"}` record and write it with `paging_env.write_output(args.output, record)`.

There is one round trip per eviction and none per hit. Replies are JSON; a batch is two lists of Python ints.

The batch boundary reveals nothing a correct policy does not already know: it knows its own cache, so it can
tell which accesses hit. Keep your resident lines exactly in step with the trusted state.

## Violations

Each of these sets `mean_cost_ratio` to Infinity, even if the worker catches the RPC error and exits cleanly:

- a victim that is not a resident line of the set, including the requested line itself;
- `step` without a victim while a decision is pending, or with a victim when none is pending;
- any other RPC method or argument shape, including `step` after the trace is exhausted;
- `finish` before the trace is exhausted, or a worker that ends without `finish`;
- an `--output` that differs from the trusted record, a missing output or a non-zero exit.

## Score

`mean_cost_ratio`: per trace, trusted misses divided by the misses of Belady's rule run independently in each
set, computed by the scorer; the score is the unweighted mean over traces. The output also carries `mean_hit_ratio`
and, per trace: `id`, `accesses`, `misses`, `opt_misses`, `cost_ratio`, `hit_ratio`, `policy_elapsed_s` and
`rpc_calls`. Up to four workers run at once.

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
