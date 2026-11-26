# Private policy protocol v2

Outer input is JSON with `task_id=icml-3011-AsymP-GDA`, `protocol=private-policy-v2`, and unique opaque `ids`. Outer prediction is a ZIP artifact containing manifest, code, model and locked dependencies, produced by `export_policy.py`. Policy CSV files are rejected.

The frozen scorer privately receives a labels directory containing manifest, original input.npz and original metric_labels.npz. Private game text and construction metadata never enter the worker input. Candidate code runs only in a filesystem/network-isolated worker; the scorer accepts no Python callbacks or pickled objects.

Each worker receives one JSON `{id,run_id,steps_per_run}`. It submits a declarative graph via `leduc_env.open_run`, uses counted `update` or `update_many` RPC calls, then writes the JSON returned by its latest `export`. Multiple graph handles are permitted within a worker, without resetting the shared budget. Every submitted strategy and counter must exactly match the trusted export. The combined per-instance budget is authoritative across both independent runs. The scorer combines player 1 from run 0 and player 2 from run 1, then computes the original NashConv using the private game.

The public `graph_ir` module specifies the numeric graph vocabulary. Trusted `private_graph` validates a version-1 instruction list, earlier-node references, finite scalar literals and the pinned native numeric contexts/operators. No separate 4,096-instruction/dimension or 256-color/list cap is imposed. Large instruction lists use bounded RPC chunks, and native allocations run under the task memory/time guards. Static graph initialization cannot access private utilities or reach probabilities. Only the declared finite simplex policy can be read: `snapshot` permits adaptive schedules, and `export` records the current strategy; subsequent updates invalidate that receipt and require a fresh final export. Neither exposes utility nodes or uncounted future computations. A custom solver can change formulas, graph colors, parameters other than the step size, independent graph runs and update schedules through this interface. Updates accept either a graph node or the native strategy-node list. Numeric random graph functions and `set_seed` retain the pinned native semantics. See the README for the verified compatibility scope and remaining environment/resource limitations.

There are no training instances. Validation game data is for evaluation and method, hyperparameter or checkpoint selection only. Do not fit reusable model state or learned preprocessing on validation. Each game is solved independently through the counted interface during scoring. Full evaluation supplies a `steps_per_run` hint of 500,000 sweeps. The enforced cap is 1,000,000 sweeps combined across both runs and all graph handles.


Game-solving algorithms, auxiliary regularization/optimization objectives, graph formulas, optimizers and schedules may change. The step size is fixed at eta=0.1 for every player and sweep: do not change, tune, schedule or adapt it, and do not emulate another step size by rescaling gradients, payoffs or updates. The original zero-sum game, its information sets and payoffs, the last-current-strategy output rule, combined counted traversal budget and NashConv objective are fixed. Changing an algorithmic regularizer does not authorize replacing the game being solved.

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
