# Frozen I/O protocol

There is no supplied offline training dataset. Validation is only for evaluation and method, hyperparameter or checkpoint selection. Do not fit reusable models or learned preprocessing on validation. Per-instance inference may use its current observation. The baseline `train.sh --out DIR` creates `DIR/NO_TRAINING`; alternative methods save their permitted fitted state/configuration there.
The sampler uses only its current input observation and the declared prior and
likelihood family. It must not read validation labels during prediction.

Input is a directory of NPZ files. Each filename stem is an opaque instance id.
Each file has exactly `task` (Unicode scalar selecting a prior/likelihood family),
`dim` (integer scalar parameter dimension), and `observation` (float32 vector).
The family describes the inference problem, not its source observation index.
The five families are gaussian linear, gaussian linear with uniform prior, SLCP,
Gaussian mixture, and two moons. Observation dimensions are respectively
10, 10, 8, 2, 2; parameter dimensions are 10, 10, 5, 2, 2.

Output is one NPZ file at the exact `--output` path (which need not end in `.npz`).
For each input id, provide two keys:

- `id`: a finite floating array of shape `(10000, dim)`, posterior samples.
- `id__calls`: a nonnegative int64 scalar giving the total number of parameter
  vectors passed to the frozen likelihood interface for this instance.

There must be no missing, extra, or duplicate keys. Write through a file handle
to prevent NumPy from appending a filename suffix. Row order is retained when
fitting the deterministic classifier; do not reorder an existing sample array.

Labels are an NPZ file with reference samples under each id and a Unicode JSON
scalar `__meta__` containing task grouping and frozen baseline guard values.
The scorer uses exactly 10,000 reference samples and all 10,000 submitted samples.
It fits the authors' MLP with five shuffled KFold splits, seed 1, reference-only
z-scoring (torch's unbiased standard deviation), two hidden layers of width
`10 * dim`, Adam, and `max_iter=10000`. Its main JSON key is `c2st`.

Load `likelihood_env.py` by absolute path from `BENCH_BENCH_DIR`. Construct
`Likelihood(id, model_name, observation)`, call it with tensors whose last axis
is the parameter dimension, then call `finish()` for the total evaluation count.
Likelihood calculations may only occur through this interface; reproducing or
modifying its calculations or counters inside candidate code is disqualifying.
Select the supplied prior and likelihood without changing them. A common inference algorithm may adapt to dimensions, prior support, observations and counted likelihood responses; solver/hyperparameter tables keyed to family identity and source-observation lookup are forbidden. All likelihood evaluations use the counted interface. Posterior approximation families and auxiliary losses may change while targeting these same posteriors.

The budget is at most 100,000 likelihood evaluations per output sample.
The other-family mean C2ST must be at most 0.5234; each other-family mean must
also be at most its frozen baseline mean plus 0.03. Protocol failure has
conceptual worst score 1.0 and is reported as `c2st: Infinity` so the harness
rejects the submission. Honest call counts are checked by a replay probe and
patch audit; permitted input use is checked by patch audit.

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
