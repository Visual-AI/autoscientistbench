# Private ISOLET classifier interface

## Inputs and predictions
`data/train/train.npz` contains sorted int64 `id`, float32 `X` (N by 617 raw
acoustic features) and int64 `y` (classes 0..25). Validation provides `id,X` and
separate `id,y` labels. Input arrays and splits are fixed. Models are affine in these 617 raw features;
internal reparameterizations must preserve that family and the selected SVM
objective, with all private-data use included in privacy accounting.

Predict writes UTF-8 CSV with the exact columns:
`id`, `history_0`..`history_9`, `delta_0`..`delta_9`, `pred_0`..`pred_9`.
There is one row per input ID in input order. IDs are canonical nonnegative
integers, used only for bookkeeping. Predictions are class integers 0..25.
All rows repeat the same ten JSON privacy histories and deltas. Each delta is
exactly 0.00001. Standard CSV quoting is required. The score is the arithmetic
mean of the ten individual accuracies; classifiers are not combined into a vote.

## Privacy accounting
Two ledger representations are supported independently for each run:

1. The original list of `[noise_multiplier, sample_rate, steps]` segments.
   Noise multipliers are positive finite numbers, sample rates are in (0,1],
   and steps are positive integers. The trusted scorer uses the original
   Opacus RDP accountant without changing its numerical conversion.
2. A mechanism ledger with exactly `accountant:"mechanisms-v2"` and `segments`.
   Each segment is one of:

```json
{"mechanism":"gaussian","l2_sensitivity":1.0,"noise_std":10.0,"releases":1}
{"mechanism":"laplace","l1_sensitivity":1.0,"noise_scale":10.0,"releases":1}
{"mechanism":"poisson_gaussian","noise_multiplier":10.0,"sample_rate":0.1,"steps":1}
```

Sensitivities must be finite and nonnegative; noise parameters must be positive
and finite. Releases/steps are positive integers. The scorer derives the bounds
from these parameters: Gaussian RDP at order a is
`releases*a*(l2_sensitivity/noise_std)^2/2`; Laplace pure-DP cost is
`releases*l1_sensitivity/noise_scale`; Poisson-Gaussian RDP uses the trusted
subsampled-Gaussian formula. RDP costs add under composition; pure-DP segments
add their cost. For mixed mechanisms the scorer uses the conservative bound
`min_a (RDP(a)+log(1/delta)/(a-1)) + pure_DP_cost` over its fixed order grid.
For Laplace-only composition it uses the pure-DP sum directly.

Every run must have derived epsilon <=1 at delta=0.00001. Final epsilon values,
arbitrary self-reported RDP profiles and unsupported mechanism declarations are
not accepted. A failed privacy/format check produces `mean_accuracy:Infinity`
with original chance utility `rule_score:1/26` as a diagnostic.

Formula checking is distinct from training-provenance verification. Patch audit
must establish the actual neighboring-dataset sensitivity, clipping, sampling,
noise generation and all compositions, including any private preprocessing and
model selection. A plausible ledger cannot certify that these operations occurred.
The ledger must cover the actual allowed training population for each of ten
independently seeded fits. Falsified or incomplete accounting disqualifies a run.

## Required affine certificate
Train writes `model_card.json` with `format:"dp-affine/v1"`, `n_features:617`,
`n_classes:26` and exactly ten `runs`, ordered by `run=0..9`. Each run contains
its valid `history`, `delta:0.00001`, finite `weight` (26 by 617), and finite
`bias` (26). `bench/check_model.py` recomputes `argmax(Wx+b)` with the original float32 torch
linear arithmetic and smallest-index tie rule. Predictions and privacy histories
must match the frozen weights and model card. Other classifier formats, nonlinear
feature models and prediction-time fitting are not admissible.

The selected objective is the original `Regularized_loss`: mean multiclass hinge
`F.multi_margin_loss` plus `.001/1560 * sum_{k,l} ||w_k-w_l||_2^2` (including the
original sum and normalization). This objective and the raw-feature affine
family are research objects. Private mechanisms, solvers, optimizers and auxiliary
numerical approximations may change while solving this objective. `bench/check_model.py`
verifies affine reconstruction and accounting formulas; source audit must
verify the actual objective, training data, clipping/noise and mechanism execution.
An affine certificate alone does not prove any of those training properties.

A valid submission must pass the affine certificate check and the documented
patch audit. Passing the numerical scorer alone does not establish artifact
consistency or truthful privacy implementation.

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
