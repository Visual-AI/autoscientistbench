# Frozen input and output contract

Input is an NPZ archive opened with allow_pickle=False. A row is one independent trial.
The arrays are id: Unicode[n], w0: float32[n,10,40,784], v0: float32[n,10,2,40], theta0: float32[n,2].
IDs are opaque; preserve their order exactly. theta0 holds (alpha, beta), which is (0, 0).
w0 and v0 are the supplied initial state of the 10 particles, drawn with the authors' rule, using the
same JAX key for both layers: w0 = jr.normal(key, (10, 40, 784)), v0 = jr.normal(key, (10, 2, 40)).
Each trial must depend only on its own row and on data/train/train.npz.

data/train/train.npz holds X: float32[800,28,28] and y: int32[800], with 0 for digit 4 and 1 for digit 9.

Labels in truth.npz contain id, theta0, the images X: float32[m,28,28], their labels y: int32[m], trial_key: int64[n]
and the guard values baseline_lppd: float64[n], baseline_budget: int32[n,500,2]. They are for offline evaluation
only. predict receives an input archive, never labels or evaluation images.

Output is NPZ, written through an open binary file handle because the requested output
filename need not end in .npz. Exactly five arrays are required:

| Key | Shape | Meaning |
| --- | --- | --- |
| id | (n,) | Unicode IDs in input order, no duplicates |
| w | (n,10,40,784) | Actual input-layer weights of the final cloud, as float |
| v | (n,10,2,40) | Actual output-layer weights of the final cloud, as float |
| theta | (n,2) | Actual final (alpha, beta), as float |
| oracle_counts | (n,500,2) | Nonnegative integer particle-image evaluations per iteration, for latent-score and theta-gradient calls |

All numeric arrays must be finite. Predictions must include every ID exactly once.
Kernel derivatives are excluded from oracle_counts. The baseline uses 8000+8000 per
iteration: 10 particles times 800 images, once for each kind of call. Declared counts must match execution, every
model call must use all 800 training images, and the judge may independently replay
algorithm.run_trial(model,data,latent_init,theta_init,config,key,steps) with an instrumented model.
Keep that entry point consistent with predict, and return the final cloud as a dict with w and v and the
final theta as a dict with alpha and beta. For loops other than an outer `jax.lax.scan`, call
`model.bench_set_iteration(i)` when that instrumentation hook is present, before the model queries for sequential
zero-based iteration i. The hook provides no data and does not change the target model.

For each trial the scorer computes test_error and log_pointwise_predrictive_density from the submitted w and v
with bench/bnn_metrics.py, the authors' functions unchanged. The score is the mean test_error over trials.
The LPPD limit is the frozen same-images baseline mean LPPD minus two population standard deviations over
baseline trials. In each iteration each of the two oracle counts must not exceed the frozen baseline budget.
Numeric guard failures score 1.0. Malformed outputs score Infinity (and fail validation).

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
