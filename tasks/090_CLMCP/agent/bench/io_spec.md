# The Cost of Learning Under Multiple Change Points

This task estimates the current mean of each independent real NAB CPU stream. The primary metric is the mean endpoint cumulative squared error relative to the fixed reference segment means. Lower is better. The original episodes, full horizons, reference segmentation, and ATC scientific setting are preserved.

Run `bash train.sh --out MODEL` to create a model and `bash predict.sh --model MODEL --input REQUEST --output ARTIFACT` to export a `private-policy-v2` artifact. The outer request is only a JSON descriptor. The trusted scorer starts one isolated `policy.sh` process per episode; the worker receives the public horizon and can import `private_rpc.call`.

For each step, call `observe()` through RPC once to obtain the current observation, then `submit(estimate)` before requesting the next observation. After all steps call `finish()`. Nonfinite estimates, skipped submissions, repeated observations, early finish, or incomplete execution fail the task. Only the judge-owned transcript is scored; writing a precomputed prediction CSV is not accepted. The policy writes a completion file to `--output` after its final RPC.

The baseline in `policy.py` calls the released exact ATC implementation with sigma 1 and alpha 0.05. Candidate code may change the online estimator within the scientific protocol. Current and past observations are permitted; future observations and hidden reference means are never mounted into a policy worker. Independent episode workers do not share state. BENCH_SEED is an algorithm seed, not a hidden-data generator seed.

There is no offline training split. Public validation streams and labels under `data/val/` are for evaluation and method, hyperparameter or checkpoint selection only. Do not fit reusable model state or learned preprocessing on validation data. Within each replay episode, online updates using that episode's observations are allowed; reset fitted state between episodes. `data/val/request.json` and `data/val/replay/` support the private replay interface.

RPC wire calls are `call("observe")`, `call("submit", float_estimate)`, and `call("finish")`. The worker input is `{task, protocol, horizon}`. Candidate Python dependencies are exported from the task environment lock. The policy must preserve inherited PYTHONPATH so artifact dependencies remain visible.

The optional artifact file `code/policy_resources.json` is a JSON object containing
only `episode_cpus`, an integer from 1 to 8. Its maximum encoded size is 4096 bytes. Omitting it requests one CPU per episode. Requests are clamped to the actual CPU
allocation, and concurrent workers receive disjoint groups within the unchanged
eight-CPU total. `BENCH_CPUS` and inherited numeric thread limits report the worker
allocation; the baseline script chooses one numeric thread.

Online estimators, working noise models, auxiliary fitting losses, change-detection rules and optimizers may change while estimating the same fixed reference segment mean. The target segmentation/means, cumulative squared-error objective, episode boundaries and observe-then-submit information order remain fixed; ATC is the baseline method.

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
