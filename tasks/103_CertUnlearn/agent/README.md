# Certified Unlearning for Neural Networks

## 1. Goal

Research question: can a model be certifiably unlearned, at a fixed epsilon and delta and with no assumption on the loss, and still reach a target accuracy sooner than retraining from scratch?

The paper's Table 2 compares retraining from scratch, output perturbation, gradient clipping and model clipping on CIFAR-10. The privacy target is (1, 1e-5)-unlearning of a random 10% forget set from the same small network. The table counts epochs to each target accuracy. A short noisy phase on retain data earns the certificate. Fine-tuning on retain data then continues without noise or clipping, with the same hyperparameters as retraining. The authors tune lambda, C1, eta and C0 per compute budget (Appendix B).

This task takes the Gradient Clipping/PABI column at the 50% target. Recover CIFAR-10 classification utility after certified removal of the supplied forget subset from a frozen pretrained neural network. Improve `epochs_to_50pct` (lower is better): the post-unlearning epoch at which accuracy first reaches 50%. The other columns are not run here. The judge retrains your code with its own seeds and evaluates hidden inputs.

## 2. Data

- `data/train/train.npz`: 45000 raw RGB images, uint8 NHWC (32 by 32 by 3), integer
  labels 0..9, and local index arrays for 40500 retain / 4500 forget examples.
- `data/train/base/checkpoint_100`: frozen 100-epoch Flax checkpoint; use this exact
  starting model. Base training is outside the task.
- `data/val/input.npz` and `data/val/labels.npz`: 5000 public validation examples,
  drawn with the original split rule and seed 142. Use these observations only for evaluation and method, hyperparameter or checkpoint selection.
- Hidden evaluation inputs and labels are held by the judge and never available here.
- Input `augment` selects the released random crop/flip validation transform or
  normalization alone. The public validation input sets it to true. Keep supplied
  inputs and their order unchanged. Training uses the original crop, flip and CIFAR
  channel normalization. Do not resplit, relabel, fetch external data or train on
  validation labels.

## 3. Metric and scoring

`epochs_to_50pct` is computed independently for each seed from 50 epoch predictions.
Accuracy is the authors' unweighted mean of batch accuracies (batch size 128, including
any short last batch). At the first crossing, linearly interpolate between adjacent
epochs; a crossing at epoch 1 returns 1, and no crossing returns 51. The judge averages
seed scores. This deliberately differs from the paper's integer crossing of a
replicate-mean validation curve. See `bench/io_spec.md` for the exact formula.

| Guard | Enforcement class |
|---|---|
| Recomputed epsilon at most 1.004 and delta above 0 and at most 1e-5 | scorer |
| Declared settings: `pabi-certificate-v1`, `tiny_net_cifar` with 10 classes, `algorithm=iteration`, constant noise and learning-rate schedules, SGD without momentum in the certified transition, `init_model_clip_type=clip`, `init_sigma=0`, 50 unlearning epochs, `0 < eta*lambda < 1`, C0 and C1 positive, one positive sigma for all noisy steps | scorer |
| Distinct noise hashes and advancing key chain; noise records cover all six parameter tensors | scorer; judge probe replays Gaussian draws |
| Frozen base identity, valid privacy assumptions, and 50 genuine epoch checkpoints | scorer checks evidence; patch audit checks execution |
| Saved epoch hashes agree with the declared model, and predictions are the forward pass of each saved checkpoint under `bench/model_def.py` | judge probe (`bench/check_model.py`) |
| Noise is actually applied, clipping is valid, fine-tuning uses retain examples only | patch audit |
| One epoch is one complete traversal of the retain data, and extra training is not hidden by relabeling epochs | patch audit |
| Complete ordered IDs, all 50 epochs and finite class predictions | scorer / validator |

The scorer recomputes epsilon from the recorded sigma, eta, lambda, C0, C1 and noisy-step
count, with the authors' accountant as written in `bench/certificate.py`.

A guard violation gives Infinity and fails the submission. Evidence records are not
accepted as proof of execution without the applicable probe and patch review.
The scorer runs automatically in every evaluation; the probe and the patch review are
separate checks.

Measured baseline on val: **24.59727600 epochs**, population standard deviation
**2.09101819**, across three seeds (per seed: 27.55319206, 23.04545455, 23.19318140).
Relative improvement must reach `max(delta, 2 * reference_std / abs(reference_mean))`.
The task's delta is **18%**, selected from the validation measurement noise.

## 4. Baseline

PABI / Gradient Clipping from the official implementation:
https://github.com/stair-lab/certified-unlearning-neural-networks-icml-2025.git,
upstream commit `9fdf6706352a7ae26b06372e3957cae2d6c85092`. Model: `src/models/tiny_net.py`;
training, clipping, noise and accountant: `src/training/trainer.py`; settings: `config.yaml`.
The network has 20042 parameters. Starting from the supplied model, the baseline uses
lambda=50, C1=10, eta=0.01, C0=20 and 11 noisy steps, then the original SGD/onecycle
fine-tuning for the remainder of the 50-epoch budget (maximum learning rate 0.1).
Epoch 1 includes the 11 noisy updates. The baseline's training batch size is 128.

The code passes the PRNG key explicitly through the JIT step, so each noisy step draws
fresh noise. Compared with upstream, it adds isolated data entry, seeds and CPU affinity
from the environment, offline telemetry, all epoch checkpoints and deferred inference.
Model, loss and optimization remain unchanged. The original augmented validation and
batch averaging behavior are retained.

Measured on 16 CPUs: train at most **12.53 minutes** per seed, predict on val at most
**0.78 minutes** per call; memory allocation is 32 GB (not a measured peak).

## 5. Interface (frozen)

- `code/train.sh` writes all inference state to its `--out` directory.
- `code/predict.sh` emits the complete epoch CSV and certificate. Write exactly the
  `--output` file, regardless of its extension.
- Honor `BENCH_SEED`, `BENCH_CPUS`, `BENCH_DATA_DIR`, `BENCH_BENCH_DIR` and
  `BENCH_CODE_DIR`. The scripts handle absolute paths and `PYTHONSAFEPATH=1`.
- `bench eval-dev --seed 1` runs one seed; `bench validate-output FILE` checks a file.
- Python 3.11; dependencies are pinned in `code/env.lock`. Install only rebuildable
  PyPI / PyTorch CPU packages using uv; submission freezes the environment. Editable,
  git and local package references are not rebuildable.

## Research contract

**Objective.** Minimize epochs to 50% CIFAR-10 accuracy after certified removal of the fixed forget subset from the supplied pretrained TinyNet, at epsilon at most 1.004 and delta at most 1e-5.

**Fixed research objects.**

- The supplied CIFAR10TinykNet architecture (20,042 parameters) and the exact pretrained base checkpoint digest. Removing this data from this model is the question, so another model or forget set answers a different one. The parameters may change as part of genuine unlearning and retain-only recovery.
- Fixed retain/forget membership, original image transforms, 50 measured retain-data epochs, batch-128 accuracy and seed aggregation. Epochs are the compute measure and accuracy aggregation defines the crossing.
- An actual unlearning guarantee at the unchanged epsilon and delta thresholds. The scientific requirement is valid unlearning at those thresholds, not a particular optimizer. The PABI assumptions are conditions of the current verifier, not the scientific objective.

**Editable methods.**

- Unlearning algorithm, surrogate loss and implementation on the same research model, with an independently valid proof.
- Data-loader implementation for retain-only recovery.

The current automatic certificate supports one proof family: constant-step, constant-Gaussian-noise clipped SGD without momentum. Its contraction, clipping and base-identity assumptions must hold when that certificate is used. Within this family the scorer does not fix sigma, eta, lambda, C0, C1 or the noisy-step count; it recomputes epsilon from them. An alternative update rule needs a separately validated privacy proof. A self-declaration is insufficient, this interface does not yet verify other proof families, and a certificate declaring another algorithm or schedule scores Infinity.

**Forbidden substitutions.** Fine-tuning after the noisy phase keeps the baseline hyperparameters, which match retraining from scratch: SGD without momentum, weight decay 0.0005, and the linear one-cycle schedule with maximum learning rate 0.1 over the 50-epoch budget. Keep the training batch size at 128 with one optimizer update per batch, so one epoch costs the same compute as in the baseline.

- Replacing, distilling or reinitializing the supplied research model, discarding it for a fresh model, distilling into a different architecture, or changing the retain/forget request.
- Claiming an unrelated privacy proof or epsilon=0 without trusted verification.
- Changing accuracy aggregation, epoch accounting, privacy thresholds or hidden-data access.
- Training or fitting reusable model state or learned preprocessing on validation features or labels, including final train+val refits or validation-based distillation.

## 6. Rules

The editable surface is `code/`. `data/`, `bench/`, this README, script signatures and
output format are frozen.

An epoch remains a complete traversal of the retain data, with all 50 actual epoch checkpoints measured. Repeated identical checkpoints are allowed when the method has converged. Validation is for evaluation and method, hyperparameter or checkpoint selection, not gradient updates. All post-certification fitting uses the certified state and retain examples only.

Save Flax TrainState-compatible msgpack with the fixed parameter tree; arbitrary model bundles or replacement prediction APIs are not accepted. Frozen TinyNet replay validates the saved parameter structure and predictions; code review is still required to prove that the base checkpoint and certified noisy updates were actually used.

Resource allocation: 16 CPUs and 32 GB memory. No network during training or prediction.
Honor each supplied seed; the judge uses three seeds and accounts for their variance.
Never read or infer hidden labels, hard-code predictions, reuse memorized results,
fabricate evidence or bypass the certification checks.

## 7. Known pitfalls

The authors' validation split inherits random crop/flip. Deferred validation predictions
use a fresh seed-controlled augmentation stream and therefore differ from training-log
validation values. Use `bench eval-dev` for the reported metric. Keep the unweighted
batch mean; changing to sample-weighted accuracy changes the protocol.

JAX plus forked loader workers can hang. The baseline uses the reproduced single-thread
forkserver preload with four DataLoader workers, while XLA sees at most BENCH_CPUS cores.
Keep imports safe under multiprocessing. No GPU or remote logging is needed.
Store every epoch under the model output directory, not in ignored code directories.
The certificate must describe actual noise and checkpoint files. Independent replay seeds
its augmentation stream from the certificate's `global_seed` and uses four loader workers,
so keep this stream in `code/predict.sh`.

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
