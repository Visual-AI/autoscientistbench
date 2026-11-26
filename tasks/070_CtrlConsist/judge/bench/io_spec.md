# CIR controller interface

`train.sh --out DIR` fits five independent controllers and saves all artifacts needed for prediction. Training samples are generated online from `data/train/config.yaml`. `BENCH_SEED=s` selects training seeds `5*(s-1)+i`, i=0..4; s=1 reproduces training seeds 0..4.

`predict.sh --model DIR --input PATH --output FILE` reads an NPZ input:

| Key | Shape | Meaning |
| --- | --- | --- |
| id | (n,) string | opaque trajectory ID, arbitrary row order |
| x | (n,501,1) float32 | observed CIR bridge states at the full time grid |
| t | (501,) float32 | times from 0 through 1 |
| repeat | (n,) int32 | independent trained-controller slot, 0..4 |

Each slot has 1000 Monte Carlo paths. The evaluation distribution is fixed CIR: a=b=epsilon=1, initial and final state 2, duration 1. IDs encode no seeds. No random keys or true drift are supplied as input.

Write an NPZ directly to FILE, through an open binary handle so no suffix is appended. Exactly two keys: `id` (the identical unique ID set, arbitrary order) and `drift` (n,500,1), finite floating point predictions of the **total controlled drift**, not the neural residual, control, KL, or a path sample. Predict at the first 500 times; endpoint_idx=-1 discards the final repeated drift. Each value must depend only on the fitted slot controller and its own (x,t); future states and other paths are not prediction features.

Public validation `labels.npz` has canonical IDs, exact drift, diffusion sigma, repeat, dt, generator seeds and problem parameters. Labels are for offline evaluation only. Scorer aligns IDs then computes each slot's released float32 `mean(0.5*sum(((true-pred)/sigma)**2, axes=(time,dimension))*dt)` and averages the five scalars. There is no statistical constraint or baseline-relative guard. Malformed or nonfinite predictions yield Infinity; valid exact predictions yield zero. Oracle access is a patch-audit rule, not a numerical lower bound on KL.

`python bench/validate.py PRED INPUT` validates the output. `python bench/scorer.py PRED LABELS` emits one JSON line with metric `kl_to_solution` and diagnostics. The generator in `bench/generator.py` reproduces the released sampling stream; it must not be invoked by prediction or used to manufacture ground-truth supervision for training.

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
