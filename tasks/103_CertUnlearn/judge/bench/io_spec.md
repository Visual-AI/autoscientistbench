# Certified unlearning I/O

`--input` is an NPZ archive with `id` (unique strings, N), `x` (uint8 RGB pixels,
N by 32 by 32 by 3), and `augment` (scalar boolean). `augment` requests the released
random crop (padding 4) and horizontal flip for validation; otherwise only normalize.
Use CIFAR channel mean (0.4914, 0.4822, 0.4465), standard deviation
(0.2023, 0.1994, 0.2010), and float32 channels-last tensors.

Labels are NPZ: `id`, integer `y` in 0..9, and scalar `batch_size=128`.
Input order is part of the evaluation protocol: the last short batch receives the same
weight as each full batch. Do not reorder inputs or change evaluation batch membership.

Prediction is UTF-8 CSV, exact header `id,epoch,pred`. Write all input IDs in original
order for epoch 1, then the same IDs for epoch 2, through epoch 50. Each `pred` is an
integer class in 0..9. The final line is `#certificate=` immediately followed by one
JSON object. No extra lines. This record has format `pabi-certificate-v1`, sanitized
`config`, `base_sha256`, `steps_per_epoch`, `final_step`, `epoch_sha256` (50 hashes),
and `noise` (the actual step audit records: step, input_key, output_key, sigma, eta,
noise_sha256, noise_shapes, noise_l2_norm).

Save `certificate.json`, `epoch_01.msgpack` through `epoch_50.msgpack` in the model
directory. Store Flax TrainState-compatible msgpack with the fixed CIFAR10TinykNet parameter tree. Arbitrary model bundles or replacement prediction APIs are not accepted. Repeated identical checkpoints are allowed when the method has converged; all 50 actual measurement epochs still must be represented.
The evidence must describe the executed algorithm and actual saved checkpoints.

The validator checks ordered ID equality, complete epoch coverage, duplicate/extra
rows, valid classes and certificate syntax. Scorer computes unweighted means of
128-example batch accuracies. For the first accuracy at least 0.50 at epoch e > 1,
score is (e-1) + (0.50 - accuracy[e-1]) / (accuracy[e] - accuracy[e-1]). A crossing
at epoch 1 gives 1; no crossing by epoch 50 gives 51. Each seed is scored separately.
Guard failures give Infinity; the JSON diagnostic includes the failure reason.

`check_model.py MODEL --input INPUT --pred PRED` independently checks the declared
privacy parameters, replays Gaussian draws, verifies checkpoint/certificate hashes, and independently
reconstructs all predictions with the fixed CIFAR10TinykNet architecture and evaluation preprocessing.
It does not prove that the training program applied those draws: audit the patch too.

The certificate must identify `model={"name":"tiny_net_cifar","n_classes":10}` and the exact frozen base digest. Replay checks Conv_0 kernels/biases `(3,3,3,32)/(32,)`, Conv_1 `(3,3,32,64)/(64,)`, and Dense_0 `(64,10)/(10,)`. Noise records cover all six tensors in this tree. A `code/model_api.py` cannot replace the research model. The existing PABI certificate requires clipped SGD without momentum during its certified transition. Retain-only recovery and surrogate losses may vary on the same model within the README's fixed optimizer, schedule and training batch rules. Other algorithms are not scientifically excluded, but their proofs need a separately trusted verifier. Artifact claims do not prove initialization, actual noise application or data provenance; those still require source review.

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
