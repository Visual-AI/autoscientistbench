# Tensor policy interface

The outer input is JSON with `protocol`, `task_id` and `cells`. It carries no instance-generation seed. The output is a stored ZIP artifact accepted by `private_eval.validate_artifact`, containing `manifest.json`, `code/policy.sh`, candidate code and installed pinned dependencies. The outer model is omitted: the same submitted `train.sh` is replayed inside the task scorer, and only its freshly produced model/runtime is used. Use the provided exporter. Prediction NPZ files and self-reported error certificates are not accepted.

Before hidden observations are available, the private scorer runs the submitted `train.sh --out MODEL` in an isolated worker with the original data and algorithm seed. It measures this stage itself; the submitted `train_seconds.json` is never used. The private scorer then launches:

```
bash /ws/code/policy.sh --model /ws/model --input /ws/input --output /ws/output/pred.out
```

`/ws/input` is a per-call read-only directory with `request.json` and Octave `observations.bin`. Metadata contains `kind`, the allowed cell IDs/base/transform/index, and the record budget. Small cells receive their observed Y and/or Omega, dimensions, transform, public condition number and zero-filled factor arrays; the baseline computes its spectral initialization from Y inside the worker. A regression batch receives the shared As, y, transforms and spectral initial factors. No X_star, U_seed/V_seed, generator seed, generator RNG state or private script is serialized. Regression intentionally shares its operator and paired initializations across its eight cells as in the original experiment.

Call the frozen `bench_observe(X,t,method,case_index)` once for each initial/updated tensor. `method` is `ScaledGD` (the primary trajectory label for any candidate algorithm) or `GD` (optional diagnostic); t is consecutive starting at 1 and capped at 1000. A small cell has case_index=0; regression uses `(transform_index-1)*4 + kappa_index-1`. The oracle checks shape and finite numeric values, computes relative error against private truth and returns that scalar. Intermediate updates may not be omitted. Initial t=1 has update index zero. Each worker writes JSON `{"completed":true,"kind":"small"}` or `{"completed":true,"kind":"regression"}` as its completion receipt; all primary scores come from trusted oracle records, not that receipt.

The full family uses n=n3=100,r=10 for RPCA/completion/robust completion and n=50,n3=20,r=5,m=25000 for regression. The score is the sum of first threshold indices at error<=1e-6, or 32000 if any cell is unreached. All 32 cells and the original error/iteration limits remain required. The frozen `replay.json` sets the record count and full dimensions; candidates cannot select others, and the scorer does not read its `wall_limit_s` field. The recovery solver, preconditioner, legal initialization derived from permitted inputs and optimization surrogate may change; the target tensors, forward operators, supplied inputs and complete update accounting remain fixed.

A private evaluation directory supplies `input.npz`, `labels.npz`, `replay.json` and the frozen runtime archive to the scorer only. Public validation has the same layout under `data/val/evaluation`. Generated truth is checked against the frozen labels before candidate execution. Candidate code and runtimes are executed only inside workers; the trusted scorer uses its separately frozen runtime. No candidate module, model pickle or executable is loaded into the scorer.

GD and ScaledGD observations may be interleaved as in the original regression batch. Each has its own original consecutive record count and stopping rules. Seal primary trajectories using `bench_observe([],0,'finish_primary',-1)` after all primary work; the baseline does this after both trajectories finish. Closure is irreversible for ScaledGD. Legal GD feedback before closure is permitted; all solver work is inside the prediction timer. Authentic RPC records alone do not prove that every internal update was reported, or that every input/feedback access respected the scientific problem: those are checked by patch audit.

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
