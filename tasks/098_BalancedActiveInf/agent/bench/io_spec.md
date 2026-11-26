# Balanced Active Inference: private active labels

Estimate the finite-population mean using the original frozen predictions and sampled labels. Preserve the original populations, 10,000 independent draws, 90% confidence intervals, design and realized-label budgets, and strict-endpoint coverage rule. The upstream prediction arrays are fixed input data. They do not require the candidate sampling or inference method to use XGBoost; candidate models, losses, designs and interval estimators may be replaced within the same observation/query contract.

`train.sh --out MODEL` validates the copied Python/R environment and writes configuration. `predict.sh --model MODEL --input EPISODE --output ARTIFACT` packages the candidate policy and model. The task scorer executes `policy.sh` in a fresh worker for each draw. Code/model are read-only; only that draw's permitted prediction, uncertainty and error-prediction arrays are supplied. Independent draws cannot retain candidate files, processes, IPC or labels.

## Observation and sampling API

`bench/env.py:load_episode(path)` returns `(metadata, predictions, Oracle)`. `Oracle().query(indices)` returns only requested labels. Indices must be distinct integers within the population; querying an already disclosed unit again is invalid, including through a second Oracle object. The final `selected` list must equal the actual union of disclosed units. Multiple query batches and adaptive choices within one draw are allowed.

`private_rpc.call('cube_sample', auxiliary, inclusion_probabilities)` is an optional helper that runs BalancedSampling.cube. Candidates may instead implement their own valid sampling design using authorized inputs and randomness; cube sampling is not mandatory. Multiple calls before or after label feedback are allowed. A call may sample a subset; its returned indices are relative to the supplied auxiliary rows. Conditional intermediate probabilities may include zero and one. Return a JSON object with exactly `version`, `predictions_sha256`, `designs` and `trials`. Use `version=1` and the supplied prediction hash. `designs` holds one named design with exactly `dtype` and `pi`; `dtype` is `float32` or `float64`, and `pi` contains one strictly positive inclusion probability at most one per population unit. `trials` contains one record with exactly `id`, `design`, `selected`, `lower`, `upper` and `width`. Use local `id=0`, the reported design name, finite endpoints and a nonnegative width equal to `upper - lower`. The original final design budget is checked on that reported design; intermediate sampling calls do not disclose labels or consume the label budget by themselves. The service records queries and calls, but validity of a multi-stage design, its inclusion probabilities, variance estimate and confidence interval still requires scientific method review.

## Randomness and independent draws

`BENCH_SEED` remains the algorithm seed. Metadata also provides `draw_index`, `algorithm_seed` and a deterministic `draw_seed`. Call `env.seed_draw(metadata, r)` to initialize Python, NumPy and optional rpy2 R randomness for a fresh draw. NumPy retains the released baseline's original draw-index seeding. Custom policies must honor the seed contract and independent-draw rule, rather than resetting every draw to the same global RNG state.

The default cube-active method retains the exact original seed-once R sequence: before each worker starts, the trusted scheduler advances that sequence once using only the unchanged public auxiliary arrays and original budget/tau. The first matching cube request returns this precommitted sample. Other cube requests use a separate deterministic draw-local R stream. Both streams snapshot and restore their own R RNG state; candidate calls and labels cannot change the next draw's stream. The default baseline's selected units and intervals are preserved. Custom algorithms keep their sampling distribution and independent random domains. Method and random-number compliance remain subject to review.

## Metric and resources

The metric is `mean_ci_width`, the mean interval width over all draws; lower is better. Coverage is the fraction of draws with `lower < theta < upper`, where `theta` is the finite-population mean. The guards require coverage at least 0.89, mean actual label queries at most 870, and each final design's NumPy sum in its declared dtype at most 869.0. A failed guard returns infinity, including in smoke. `bench status` reports the validation baseline and the relative-improvement delta.

The task provides 8 CPUs and 32 GB memory. Draws run concurrently: up to 8 isolated workers at a time, each with 1 CPU (`BENCH_CPUS=1`), so a method cannot use several threads within one draw. A worker starts once per draw and imports from an artifact without bytecode, so importing scipy adds about 0.9 s per draw; the baseline does not import scipy or R in its workers. The concurrent validation reference has mean width 0.20001208780788665 and across-seed standard deviation 0.0, based on one seed. For seed 0, preparation took 2.52 s, validation artifact packaging took 0.11 s, and validation replay took 480.19 s. These are measured elapsed times, not execution limits.

Sampling and inference code may change within the frozen-input, query, budget and independent-draw rules. There are no designated training observations; `data/train` contains provenance only. `data/val/evaluation` is for evaluation and method, hyperparameter or checkpoint selection only. Do not fit reusable models or learned preprocessing on validation. Within-draw estimation may use that draw's authorized frozen arrays and oracle feedback, without carrying fitted state across draws. The validation population was used to train the frozen upstream models, so it is not held-out XGBoost validation. Hidden labels and the population mean are not policy inputs. The copied runtime and dependency lock remain frozen.

The task-specific Research contract in README is binding: output validity alone does not permit substituting a different research object or objective.

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
