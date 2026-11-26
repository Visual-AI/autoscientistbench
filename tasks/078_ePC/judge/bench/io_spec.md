# Fixed MLP / cross-entropy / predictive-coding contract

Input NPZ contains `id` (sorted unique integer row IDs) and uint8 `x` with shape `(N,28,28)`. Training additionally supplies integer `y` in 0..9. The fixed predictor applies float32 division by 255, subtraction of 0.5, division by 0.5 and row-major flattening. Supplied arrays, splits and this input representation are fixed.

Prediction CSV has exactly `id,pred0,pred1,pred2,pred3,pred4`, one row per input ID in the same order and integer classes 0..9. The five columns are independently trained replicas; scoring averages individual accuracies, not an ensemble vote.

The research predictor is a 784→128→128→128→10 MLP with GELU at the three hidden layers and a linear output, 134794 parameters. The output objective is cross-entropy and the learning paradigm is predictive coding. PC inference solvers and local weight-update methods may change, along with the hyperparameters and training duration of the changed method; the baseline 14 epochs, four inner SGD steps and Adam settings are not mandatory for a changed method. Ordinary end-to-end BP and intentionally choosing an exact-BP degenerate PC construction are outside the task. Autodiff used inside genuine PC computations is allowed.

`train.sh` saves `model_card.json` with `format="epc-mlp-v1"`, `dimensions=[784,128,128,128,10]`, `activation="gelu"`, `parameters_per_model=134794`, and `training`. Training contains positive integer `bench_seed`, a truthful `rule`, and `loss="CrossEntropyLoss"` (implicit for the legacy baseline card). Standard rule descriptions are `error-based predictive coding`, `state-based predictive coding`, or `predictive coding`; a custom PC solver may instead declare `paradigm="predictive_coding"` with its truthful rule description. Declarations are auditable claims, not proof of the learning dynamics.

The card has five ordered `replicas`, each with `slot` 0..4, `rng_seed=(5*(bench_seed-1)+slot) mod 2**32`, relative NPZ `weights` path and SHA256 digest. NPZ must contain only finite float32 weight/bias tensors for prefixes `layers.0.0`, `layers.1.0`, `layers.2.0`, `layers.3`, with the fixed dimensions above. Implementations may differ internally but must export this predictor faithfully.

`check_model.py DIR --input INPUT --pred PRED` checks the declared research object and learning paradigm, validates tensor shapes/parameter count, and independently reconstructs classes without importing candidate Python. Arbitrary-model manifests are not accepted. Run this check explicitly; it does not run automatically. Source review is required to establish actual PC/local updates, CE use, lawful training and replica independence.

`validate.py` checks CSV format and input alignment; `scorer.py` preserves the original mean individual accuracy formula. Invalid submissions fail with Infinity. The supplied data-use permissions and seed mapping are unchanged.

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
