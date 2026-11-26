# Transformer Circuits Can Realize Clustering Algorithms

## 1. Goal
**Research question.** With constructed weights, one transformer layer of attention and residual connections performs exactly one iteration of Lloyd's k-means algorithm. What clustering algorithm results when those weights are learned instead? The paper trains one layer on a distribution of clustering tasks and applies it repeatedly to new tasks, one application per clustering step. This task asks how far a learned transformer can lower the k-means objective in a fixed number of steps.

**Experiment.** The paper compares the trained transformer with Lloyd's algorithm for the same number of clustering steps from random initial centers. Each task has 512 points in 32 dimensions, ten supplied random initial centers, and twenty clustering updates. The paper's model trains on Gaussian mixtures only and is also evaluated on mixture families it did not train on.

**Your task.** Improve `normal_log_objective` (lower is better) over the baseline transformer while satisfying four distribution-shift guards and the compute constraints below.

## 2. Data
- `data/train/generator.json` specifies online Normal-mixture training: ten equally weighted components with scale 0.1, 32 fresh tasks per batch and 10000 optimization steps. The baseline independently draws its scheduler-development tasks before training.
- `data/val/input.npz` and `data/val/labels.npz` contain 1600 public validation tasks, 320 from each of Normal, Cauchy, Gumbel, Laplace and LogNormal. The public root seed is 721903. Family streams use NumPy SeedSequence; labels contain the exact generation metadata.
- Prediction receives only the supplied observations and initial centers.
- Frozen `bench/generator.py` preserves the released mixture generator's random draws and coordinatewise min/max normalization. Inputs contain ordered opaque IDs, observed points and initial centers; family identities are only in labels. There are no supervised target cluster assignments: quality is the k-means objective of the submitted centers.
- Fresh synthetic Normal training batches are allowed. Do not alter provided data, infer hidden seeds, memorize instance IDs or use labels at prediction time. Every instance must be processed from its own observations and the fitted model.

## 3. Metric and scoring
`normal_log_objective` is the median across three independent fitted models of the Normal-family average `log(sum_i min_j ||x_i-c_j||²)` at the submitted final centers. The scorer recomputes distances in torch float32. It averages logs, rather than logging the average objective. One benchmark seed runs the complete three-model experiment; the evaluator averages the scores of three independent benchmark seeds.

| Guard | Requirement | Enforcement |
|---|---|---|
| OOD quality | Median of the three family means: Cauchy ≤ 3.37, Gumbel ≤ 5.106, Laplace ≤ 4.988, LogNormal ≤ 5.028 | scorer |
| Initial centers | Slot zero of every model's trajectory equals the supplied centers exactly | scorer |
| Output format | Exact keys, IDs in input order, float32 `centers` of shape `[3,N,21,10,32]`, all values finite | validator |
| Capacity | At most 21,672 fitted real parameters per model | scorer checks declarations; frozen rollout and replay probe count parameters; patch audit checks completeness |
| Update protocol | Exactly 20 updates from the supplied centers | frozen rollout makes the calls and records 21 center states; replay probe recomputes them; patch audit checks per-step semantics |
| Learned update | The update is computed by a learned transformer, as stated in section 5 | patch audit |
| Independent problems | No memorized predictions, cross-instance information or hidden-data access | patch audit |

Numerical guard violations return `Infinity` and fail the submission; one benchmark seed returning `Infinity` is enough. The diagnostic `paper_score` then records the specified finite worst value 6.343. Reported resource fields are not trusted: the evaluator must run the replay probe `bench/check_model.py` before accepting the score. It runs separately from the scorer (usage in `bench/io_spec.md`). Replay verifies every final per-task log objective within absolute tolerance 1e-4; this is invariant to center-label permutations and accommodates observed cross-process numerical differences.

PASS uses `max(delta, 2*reference_std/abs(reference_mean))` on the three-seed means, with delta 1%, and requires all guards. On validation the baseline's `2*std/mean` is about 0.41%, so the 1% delta is the larger term there, about 0.05 in the metric.

## 4. Baseline
Official implementation: https://github.com/rithram/kmeans-trf, upstream commit `948d976c78265a8ad50a994d77556e0b8f2406d5`. `code/kmt.py` defines the four-attention-block KModel, `losses.py` the soft k-means loss, `ctasks.py` the generator, and `train.py` the training loop.

One update embeds points with zero scratch and centers with one-hot IDs, computes `X' = X + CAX(X,C) + SAX(X,X)` and `C' = C + CAC(C,X') + SAC(C,C)`, and returns the first 32 coordinates of `C'`. Attention scores are negative squared Euclidean distances between learned projections. The baseline uses attention and residual connections, with no separate feed-forward block. Training minimizes the smoothed objective after one update. Prediction applies the same layer twenty times and re-embeds the current centers each time. The baseline scheduler uses Lloyd's from `utils.py` only as a training-time reference.

The baseline retains onehot scratch, 42-dimensional embeddings/projections, softmax attention inverse temperature 1, dropout 0.01, soft objective inverse temperature 10, Adam learning rate 0.01, plateau factor 0.5 and patience 5, validation every 50 steps, batch size 32 and 10000 updates. Predictions use the final checkpoint. The authors' functional dropout remains active even in eval mode.

The benchmark adapter adds data/model entry points, seed and CPU controls, model serialization and frozen rollout calls. Models, loss, hyperparameters, optimizer and training computations are unchanged. Up to three members run concurrently with at most sixteen numerical threads each, within the declared 128-CPU budget.

Measured on public validation over three benchmark seeds at 128 allocated CPUs:
- `normal_log_objective` mean **4.9973313014**, standard deviation **0.0102310446** (per seed 4.9908, 5.0118, 4.9894). All guards pass.
- Worst family value across seeds: Cauchy 3.3525, Gumbel 5.0680, Laplace 4.9560, LogNormal 5.0104.
- Training **24.83 min** per three-model experiment; validation prediction **1.24 min**. One full baseline run on validation therefore costs about 26 minutes.

## 5. Interface (frozen)
- `bash code/train.sh --out DIR` retrains and saves everything prediction requires.
- `bash code/predict.sh --model DIR --input PATH --output FILE` writes the exact requested output path. Tensor formats are specified in `bench/io_spec.md`.
- `code/bench_adapter.py` exports `load_model(model_dir,index)` and `step(model,samples,centers)`, performing one clustering update. The frozen runner records exactly twenty updates. The update must be computed by a learned transformer built from attention with residual connections, with optional feed-forward blocks. Transformer variants are allowed; a hand-written or analytic rule such as Lloyd's algorithm is not. An unbounded `solve` cannot replace the measured rollout.
- Generic fitted state must include `model0.parameters.npz` through `model2.parameters.npz`, with every learned scalar array; Torch modules are counted from `model.parameters()`. An optional `parameter_count(model)` must agree. Learned constants cannot be omitted.
- The frozen runner reloads the model for each batch of 32 tasks and gives each of the three models its own algorithm stream.
- Honor `BENCH_SEED` and `BENCH_CPUS`; use `BENCH_DATA_DIR`, `BENCH_BENCH_DIR` and `BENCH_CODE_DIR` for runtime locations.
- Dependencies are rebuilt from `code/env.lock` with exact PyPI / PyTorch CPU pins. Editable, git and local-path installs cannot be rebuilt.

## 6. Rules
- Free surface: `code/`, including model, loss, features, optimization and hyperparameters, subject to the guards and interfaces above.
- Frozen surface: data, `bench/`, this README, tensor format and entry-point signatures.
- Three benchmark seeds are used; each run must train three independent models and honor the supplied algorithm seed. The baseline training members derive from 153476998, 235510766 and 205084180 by adding `1000003*(BENCH_SEED-1)` modulo `2**31-1`.
- Resource allocation: 128 CPUs and 32 GB memory. Training and prediction have no network.
- Hidden-data access, seed guessing, hard-coded outputs, copied paper-result tables, falsified diagnostics or bypassing the replay interface disqualify a submission. All scored final objectives must be reproducible through the frozen runner. Do not monkey-patch its libraries, clocks or counters, inspect its call frames, or falsify model metadata; measurement integrity is checked by patch audit.
- Prediction may read only supplied observations and model artifacts.

## 7. Known pitfalls
The baseline uses evaluation-mode dropout and seeds it. A smaller CPU allocation may change float32 reduction order. The source scheduler's one-step relative validation objective differs from the twenty-step mean-log evaluation metric. Repeated model construction consumes RNG state; retain the frozen runner for reproducible replay. Use an open binary file handle with `np.savez` so NumPy does not append an unwanted extension. Inputs may arrive with a different number of instances; preserve their exact ID order and never infer a split from its size. The `ood_thresholds` array in `data/val/labels.npz` lists Cauchy as 3.352; the scorer's 3.37 is the binding value.

## Research contract
**Objective.** Minimize the Normal-family median-of-seeds mean log k-means objective after twenty updates under the original capacity and OOD controls. The median runs over the three fitted models of one benchmark seed.

**Fixed research objects.**
- Normalized 512x32 point-cloud problems, ten supplied initial centers, three independent model members and original family guards.
- Twenty actual clustering updates and at most 21,672 fitted real parameters per model.
- Original float32 objective and median-of-means aggregation; no hidden family or target access.

These stay fixed because the paper compares the objective after the same number of steps from the given initial centers, and matched capacity means a gain comes from the learned algorithm, not a larger model. The object under study is a learned transformer as a clustering algorithm.

**Editable methods.** Transformer architecture, training surrogate and optimizer within the fixed capacity and horizon. The update must be computed by a learned transformer built from attention with residual connections, with optional feed-forward blocks. Transformer variants are allowed; a hand-written or analytic rule such as Lloyd's algorithm is not. The paper itself varies the smoothed-loss temperature, the embedding size with and without the ten scratch dimensions, and attention scores, attention activation and normalization.

**Forbidden substitutions.** Within one `step` call, each fitted layer runs at most once, so a call cannot apply a layer repeatedly to iterate. Hand-written clustering computations, such as hard assignments, Lloyd means, re-seeding or farthest-point selection, may not run before, after or around the transformer inside `step`, and their results may not be given to it as input features.
- Replacing twenty measured updates with an unbounded inner solver or changing supplied initial centers. Extra iterative refinement cannot be packed into a call to bypass the horizon.
- Computing the update with a hand-written or analytic rule such as Lloyd's algorithm.
- Exceeding matched capacity, using cross-instance inference data, or weakening OOD/metric controls.
- Training or fitting reusable model state or learned preprocessing on validation features or labels, including final train+val refits or validation-based distillation.

Parameter completeness, per-step semantics and the learned-transformer requirement are checked by patch audit, not by the scorer. Timings remain truthful diagnostics.

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
