# Private policy protocol v2

Outer input is JSON with `task_id`, `protocol=private-policy-v2`, and unique opaque `ids`. Outer prediction is a ZIP policy artifact produced by `export_policy.py`; it includes `manifest.json`, `code/policy.sh`, code, model and locked dependencies. Old trajectory CSV files are not valid outer predictions.

The frozen scorer receives a private labels directory containing `manifest.json`, original `input.csv` and `metric_labels.npz`. It never imports candidate Python in its own process. It executes one fresh policy worker per original episode through the task's sandboxed evaluator.

Worker input is JSON `{"id":"opaque episode id"}`. `ep = dueling_env.make_episode(episode_id)` opens the assigned episode once. Each round calls `get_context()`, `serve(a,b)`, then `advance()`, in that order. Delayed preference feedback is returned only by `advance()`. After all rounds call `finish()`; it returns the committed actions, actual rounds and regret.

Worker output has CSV columns `id,actions,rounds,regret`. It must exactly match the finish record. Only trusted records enter the unchanged numerical metric calculation. The score is arithmetic mean final cumulative regret, lower is better. Trusted scoring reconstructs regret from the original hidden utilities and the actions recorded by the service.

No test seed or future state is public. The environment source describes the distribution but lacks hidden episode seeds. There is no offline training split. Public validation states are for evaluation and method, hyperparameter or checkpoint selection only. Online updates within an episode may use only its causally released contexts and feedback; fitted state must not cross episodes. The service allows one episode per worker; batched helper calls cannot reset or create extra episodes.

The total original resource quota remains 8 CPUs and 16 GB RAM. By default workers each receive one CPU. An optional `code/policy_resources.json` with only `{"episode_cpus": N}` requests 1 through 8 CPUs, clamped to actual affinity; N greater than 1 runs episodes sequentially to preserve the original single-threaded rule for concurrent numerical workers. The scorer enforces causal calls and trusted counters; algorithm semantics, fresh state and reproducibility remain subject to review.


Dueling policies, internal feature transformations/predictors, auxiliary learning losses and optimizers may change, and hyperparameters may be retuned with such a change. The actual environment utility model, context generator, delay/corruption process, causal service and regret target remain fixed. Methods may compute features from legally revealed contexts; this does not grant access to private utility parameters or unreleased service data.

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
