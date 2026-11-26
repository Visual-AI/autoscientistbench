# Private policy protocol v2

Outer input is JSON with `task_id`, `protocol=private-policy-v2`, and unique opaque `ids`. Outer prediction is a ZIP policy artifact produced by `export_policy.py`; it includes `manifest.json`, `code/policy.sh`, code, model and locked dependencies. Old trajectory CSV files are not valid outer predictions.

The frozen scorer receives a private labels directory containing `manifest.json`, original `input.csv` and `metric_labels.npz`. It never imports candidate Python in its own process. It executes one fresh policy worker per original episode through the task's sandboxed evaluator.

Worker input is JSON `{"id":"opaque episode id"}`. `ep = clasp_env.make_episode(episode_id)` opens the assigned episode once. `initial()` returns the starting point. `step(x)` commits a decision in [0,1]^10 before returning that round's (H,y,A,b); `step_many(xs)` commits the supplied batch sequentially. `rounds` is a trusted counter. After 100 steps call `finish(final_x)`, returning the full trajectory and accumulated metrics.

Worker output has CSV columns `id,actions,rounds,loss,ccv1,ccv2`. It must exactly match the finish record. Only trusted records enter the unchanged numerical metric calculation. The score is mean CCV2, lower is better. Guards require mean loss at most 1.01 times the original baseline loss and CCV1 at most 1.05 times the original baseline CCV1. Domain and reported-metric checks remain unchanged. All metrics are independently reconstructed from the original labels and the service-recorded decisions.

No test seed or future state is public. The environment source describes the distribution but lacks hidden episode seeds. There is no offline training split. Validation observations and labels are for evaluation and method, hyperparameter or checkpoint selection only, not reusable-model or preprocessing fitting. Online updates may use feedback already revealed within their own episode. The service allows one episode per worker; batched helper calls cannot reset or create extra episodes.

Up to eight independent workers share the actual allocated CPUs, with one CPU per worker by default. Optional `code/policy_resources.json` with an integer `episode_cpus` from 1 through 8 requests multiple CPUs per episode; groups never overlap, and actual affinity bounds group size. Worker `BENCH_CPUS` and numerical thread defaults reflect its assigned group. The full validation replay took 39.35 s in the recorded reference run.

All code, initial decision policies, update rules, hyperparameters and training remain editable under the original causal rules. Preserve the full 100 decisions plus the final unscored update. Runtime isolation enforces the observation order and authentic records; method semantics and randomness use still need source review.


Decision policies, auxiliary surrogate losses, internal models, update rules, optimizers and hyperparameters may change. The actual per-round quadratic loss, linear constraints, decision domain, causal feedback order and loss/CCV scoring formulas are fixed; changing an internal surrogate does not change those research objectives.

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
