# The Cost of Learning Under Multiple Change Points

This task estimates the current reference segment mean of each independent real NAB CPU stream. Reference targets are sample means within fixed reference segments. The `cumulative_regret` metric averages the sum of squared estimation errors across episodes, using every timestamp without dividing by the horizon. Lower is better. The original episodes, full horizons, reference segmentation, and ATC scientific setting are preserved.

Run `bash train.sh --out MODEL` to create a model and `bash predict.sh --model MODEL --input REQUEST --output ARTIFACT` to export a `private-policy-v2` artifact. The outer request is only a JSON descriptor. The trusted scorer starts one isolated `policy.sh` process per episode; the worker receives the public horizon and can import `private_rpc.call`.

For each step, call `observe()` through RPC once to obtain the current observation, then `submit(estimate)` before requesting the next observation. After all steps call `finish()`. Nonfinite estimates, skipped submissions, repeated observations, early finish, or incomplete execution fail the task. Only the judge-owned transcript is scored; writing a precomputed prediction CSV is not accepted. The policy writes a completion file to `--output` after its final RPC.

The baseline in `policy.py` calls the released exact ATC implementation with sigma 1 and alpha 0.05. Candidate code may change the online estimator within the scientific protocol. Current and past observations are permitted; future observations and hidden reference means are never mounted into a policy worker. Independent episode workers do not share state. BENCH_SEED is an algorithm seed, not a hidden-data generator seed.

There is no offline training split. Public validation streams and labels under `data/val/` are for evaluation and method, hyperparameter or checkpoint selection only. Do not fit reusable model state or learned preprocessing on validation data. Within each replay episode, online updates using that episode's observations are allowed; reset fitted state between episodes. `data/val/request.json` and `data/val/replay/` support the private replay interface.

Measured validation reference: mean 1992.3608585454524 and standard deviation 0.0 from one algorithm seed. For algorithm seed 1, training took 0.57 s, artifact export took 0.52 s, and validation replay took 1.406 s.

Independent
streams share at most eight allocated CPUs. By default each worker receives one
CPU. To parallelize within a stream, include `code/policy_resources.json` containing
only `{"episode_cpus": N}`, where N is an integer from 1 to 8. The scorer assigns
nonoverlapping CPU groups, clamps N to the available allocation, and reduces the
number of concurrent streams accordingly. Workers inherit `BENCH_CPUS` and numeric
thread limits matching their actual allocation; the supplied baseline `policy.sh`
still chooses one numeric thread and may be edited. The configured memory
budget remains 4 GB.

Online estimators, working noise models, auxiliary fitting losses, change-detection rules and optimizers may change while estimating the same fixed reference segment mean. The target segmentation/means, cumulative squared-error objective, episode boundaries and observe-then-submit information order remain fixed; ATC is the baseline method.

## Research contract

**Objective:** Track the fixed reference segment mean of each NAB CPU stream using the original experiment observation order, minimizing mean endpoint cumulative squared error.

**Fixed research objects:** Original stream observations, episode horizons/boundaries and fixed reference segment means are the research objects; candidate detection rules cannot relabel the evaluation target. At step t, observe X_t then submit an estimate using X_1:t before observing X_(t+1); each episode has fresh state. The original experiment metric sums squared errors at all released timestamps and averages episode endpoints.

**Editable methods:** Online mean estimators, change-point detectors, working noise/latent-state approximations, auxiliary training losses, optimizers and numerical methods may change. ATC, its sigma1/alpha.05 tuning and exact scan implementation are baseline choices; the evaluation squared loss and reference segmentation are not candidate-selectable.

**Forbidden substitutions:** Do not replace current segment-mean tracking with next-observation prediction or retrospective smoothing, redefine change points/means in labels, or optimize a substituted reference. No future observations, hidden means, precomputed episode estimates, seed reconstruction or state/data reuse from another evaluation episode. When retaining the supplied ATC pipeline, changing only sigma=1, alpha=0.05, the `gamma_paper` coefficients 6 and 2, or a global multiplier on that threshold does not count as a method improvement.

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
