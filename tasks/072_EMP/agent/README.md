# Effective Model Pruning: FC12 weight sparsity under a loss constraint

The judge runs your code in a clean environment, evaluates hidden data and compares
against the archived baseline. The requirements below are binding.

## 1. Goal

Effective Model Pruning (EMP) estimates how many weights to retain from their
importance scores. This task prunes a fixed dense FC12 classifier
on CIFAR-10. Maximize `sparsity` (higher is better): the percentage of Linear weights
removed, subject to absolute cross-entropy loss change at most 0.1 from the fixed dense
checkpoint. Every retained weight and every bias must remain identical to dense.
You may redesign how the binary pruning mask is chosen within these constraints.

## 2. Data

- `data/train/train.npz`: 40,000 training images, original integer row ids, padded
  normalized float32 images and integer class labels. Details are in `bench/io_spec.md`.
- `data/val/input.npz` and `data/val/labels.npz`: 10,000 validation images, drawn by
  stratifying a 20% holdout of the original training pool with seed 20261030.
  Labels are public. Rows are sorted by original id. Labels also contain approved
  dense tensor digests for the supported seeds.
- `data/dense/seed<S>.npz`: the approved dense FC12 checkpoint for seed S, the network
  this task prunes. It holds the 24 parameters under their state-dict names plus the
  epoch it was selected at, and its digest is the `dense_hash_S` in the labels. Load the
  file for `BENCH_SEED`.
- The evaluation split is held by the judge and is never readable in this workspace.
- The baseline loads the supplied dense checkpoint and prunes it. Candidates may fit
  their pruning strategy on train only; validation may select it;
  the approved dense checkpoint must remain identical. No outside data or downloads.
  Preserve the provided splits, labels and normalization.

## 3. Metric and scoring

`bench/scorer.py` computes `100*(1-kept/total)` over all 7,263,000 FC12 Linear weights.
Biases are excluded from that denominator. It independently executes the certified
pruned and dense models, then computes `epsilon = abs(CE_pruned-CE_dense)` with the
original sample-weighted float32 batch CE convention. Epsilon must be at most 0.1.
A prediction carries the complete model certificate, so self-reported sparsity or
loss values cannot substitute for the calculation.

| Requirement | Guard class | Enforcement |
|---|---|---|
| Approved fixed dense checkpoint for the supplied seed | scorer | Tensor digest matches frozen label metadata |
| FC12 architecture and valid finite tensors | scorer | Exact parameter keys, shapes and dtypes |
| Mask-only weights; unchanged biases | scorer | Exact elementwise comparison against dense |
| Honest kept/total counts | scorer | Counts boolean masks directly |
| Logits describe the certified model | scorer | Independent forward pass reproduces logits |
| Absolute CE difference at most 0.1 | scorer | Independent dense and pruned CE computation |
| Model directory agrees with prediction certificate | patch audit | Optional `bench/check_model.py` |
| Honor the requested seed; use only allowed data; fix mask before prediction | patch audit | Review training and prediction code |

The mathematical worst score is 0 (`raw_score`). A violated guard prints `sparsity=Infinity`
as the harness's failed-submission signal. Valid scores equal `raw_score`.
Baseline validation `sparsity`: **40.4509339575%**, population standard deviation
**0.6352422070 percentage points**, across seeds 1,2,3. Relative delta is
**4.00%**. The validation loss constraints pass for all three seeds.
PASS requires relative mean improvement at least `max(delta, 2*reference_std/reference_mean)`;
`bench status` reports the configured baseline and delta. Three supported seeds are 1, 2, 3.

## 4. Baseline

The official repository is <https://github.com/noMushroomw/Effective-model-pruning>,
upstream commit `8b21cb0d570a0370d7d86c4554b00edb419943b4`. `code/emp_cli.py` contains the original FCNet, training, optimizer and EMP routines.
FC12 widths are 1000,900,800,750,700,650,600,500,400,200,100,10 after flattening the image.
Dense training uses 200 epochs, batch size 128, SGD lr=0.01, momentum=0.9, no weight decay,
and the original constant learning-rate scheduler. Random cropping and flipping are
unchanged. EMP uses global absolute weight magnitudes and beta=1.0, including the
original inclusive threshold convention.

Deterministic pixel preprocessing is cached. Each approved dense checkpoint was
produced by training from scratch on train for that seed and selecting the epoch with
the best validation accuracy; the result is supplied under `data/dense/`, so the dense
stage is a load, not a training run. `code/emp_cli.py` keeps the original dense
training routine for reference; re-running it does not reproduce the approved tensors
(a 200-epoch CPU SGD trajectory is bit-reproducible only on the host it was frozen on)
and therefore fails the identity guard.
Measured runtime at the 32-CPU budget: the train stage (load + prune) **4.74 s**
on average (slowest 4.82 s), validation prediction **3.45 s** on average.
Memory allowance is 32 GB. Training uses up to 12 intra-op threads and four data workers;
pruning uses up to four intra-op threads to preserve the reproduced reduction setting.

## 5. Interface (frozen)

- `bash code/train.sh --out DIR`: train and write a complete model directory, including
  `certificate.npz` and `model_card.json`.
- `bash code/predict.sh --model DIR --input PATH --output FILE`: read images and write
  exactly that output path, in the NPZ certificate format specified by `bench/io_spec.md`.
- Environment: `BENCH_SEED`, `BENCH_CPUS`, `BENCH_DATA_DIR`, `BENCH_BENCH_DIR` and
  `BENCH_CODE_DIR`. The scripts accept absolute paths and honor `PYTHONSAFEPATH=1`.
- `bench status`, `bench eval-dev --seed 1`, `bench validate-output FILE`, and
  `bench submit --final` provide the normal development workflow.
- `python bench/check_model.py DIR --input data/val/input.npz --pred FILE` optionally
  checks model directory consistency. The scorer already verifies every numerical guard.
- Dependencies are pinned in `code/env.lock`. Install through `uv pip install`; submission
  freezes the environment. Only index-resolvable `name==version` pins are supported.

## 6. Rules

You may replace the pruning method, auxiliary importance model, mask-learning surrogate loss, optimizer and mask-selection strategy under `code/`. This freedom does not permit changing the dense model training target or its certified tensors. The pretrained dense network is the object being pruned, a fixed problem input. You must preserve the fixed dense tensors and FC12 computation,
keep the output interface and produce a single input-independent binary mask per run.
No fine-tuning of retained weights, bias updates or alternative dense model is allowed.
The approved dense checkpoint is supplied as data; do not retrain or substitute it.

`data/`, `bench/`, `README.md`, script signatures and prediction format are frozen.
The judge uses three seeds and a 32-CPU, 32-GB allocation. There is no network during train or predict. Reconstructing hidden data, memorizing
published predictions, reading unprovided labels or switching seeds is disqualifying.
Fit pruning strategies on train only and use validation only for evaluation/selection; the fixed dense identity still applies.

## 7. Known pitfalls

- The dense tensors must be the bytes of `data/dense/seed<S>.npz` for the seed being run.
  Load that file; re-deriving the checkpoint by training fails the digest guard.
- The sparsity denominator counts only weights, not biases; every layer matters.
  A mask that barely passes on val may violate the 0.1 loss constraint on unseen data.
- EMP's threshold includes its boundary element and magnitude ties. This behavior is
  intentional in the baseline. Preserve it when checking baseline reproducibility.
- Keep complete finite float32 model tensors and boolean masks. Predictions must follow
  input id order. NPZ output needs an open file handle to avoid automatic suffix addition.
- DataLoader uses a short relative temporary directory for worker sockets. Removing it
  can cause AF_UNIX path length errors in long workspaces.
- Frozen helpers load by absolute path from `BENCH_BENCH_DIR`; do not shadow them through
  `PYTHONPATH`. Write artifacts into the model/output directories, never data or bench.

## Research contract

**Objective:** Maximize mask-only sparsity of the specified frozen FC12 CIFAR-10 network while limiting its cross-entropy loss change to .1.

**Fixed research objects:** The approved seed-specific dense tensors, FC12 widths/ReLU computation, CIFAR-10 normalized inputs and all retained weights/biases are the fixed object being pruned. One input-independent boolean weight mask per run, 7263000-weight denominator, unchanged biases, exact dense identity and original CE difference<=.1 remain fixed.

**Editable methods:** Pruning algorithms, auxiliary importance/mask models, surrogate mask-learning losses, optimizers, selection criteria may change; fitting stays on the training split, and validation is for selection only. The dense checkpoint is a supplied read-only input and is not reconstructed; the final pruned predictor remains that checkpoint multiplied by a binary mask.

**Forbidden substitutions:** Do not retrain/fine-tune surviving weights, change architecture/activations/biases, substitute an easier dense network, alter input normalization, or change the CE guard. No input-dependent masks, evaluation-label access, forged model/sparsity/logit certificates or unsupported seed switching. When retaining the supplied EMP global magnitude pipeline, changing only beta=1.0 or the resulting global magnitude threshold does not count as a method improvement.

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
