# A Resolution-Agnostic Geometric Transformer for Chromosome Modeling Using Inertial Frame

The judge retrains your code from scratch and compares its predictions with an archived baseline.

## 1. Goal

Reconstructing chromosome 3D structure from Hi-C data usually has two steps: a numerical method turns the contact matrix into initial coordinates, then a learned model maps those coordinates to refined ones. The paper asks whether such a refinement, trained on some chromosomes, gives structures on other chromosomes whose distances agree better with contact-derived distances. Its experiment compares numerical reconstructions (LorDG, 3DMAX), graph networks (HiC-GNN, HiCEGNN) and the InertialGenome refinements (IG-LorDG, IG-3DMAX) by dSCC and dRMSE on held-out chromosomes, at several resolutions.

This task takes one cell of that comparison. Improve chromosome geometry from supplied LorDG 3D structures for the paper's GM12878 B-lymphocyte, GSDB 100KB setting. Predict refined 3D coordinates and maximize `dscc`, the mean correlation between predicted distances and contact-derived target distances. The hidden evaluation data follow the same preprocessing as validation. You may change the method subject to the rules below. Inputs, targets, splits and metric stay fixed so that a higher score means a better refinement of the same problem.

## 2. Data

- `data/train/input.npz` and `data/train/labels.npz`: 15 training structures with 17,675 real loci, supplied LorDG coordinates, local locus indices, and contact supervision.
- `data/val/input.npz` and `data/val/labels.npz`: four validation structures with 6,700 real loci and public labels.
- Inputs contain opaque `ids`, per-structure `coords_i` arrays (N,3), and `nodes_i` arrays (N,). Labels contain `contact_i` matrices (N,N), `n_i`, and matching ids. Coordinates have the source LorDG units. See `bench/io_spec.md`.
- The test split is held by the judge. Its chromosomes appear in neither train nor val. Its identities, files and labels are unavailable in this workspace.
- Use only supplied data. Do not re-split, relabel, fetch external data, or import pretrained models. Contact matrices are supervision and may not be used as prediction inputs.

## 3. Metric and scoring

`bench/scorer.py` computes the authors' `test_model` metric. Predicted float32 pairwise distances are divided by the maximum over all padded slots. Targets are contact^(-0.5), normalized by their maximum within the structure. Only off-diagonal real-locus pairs with contact above 1e-8 count. On those pairs the scorer calculates SciPy Spearman correlation (dSCC) and float32 RMSE (dRMSE), then averages equally across structures. Higher `dscc` is better. dSCC depends only on the rank order of distances; dRMSE also depends on their scale.

| Guard | Enforcement class |
|---|---|
| Mean dRMSE ≤ 0.095 on test and ≤ 0.12 on val; defined, finite metric | scorer |
| Exact ordered ids and slots, finite coordinates, correct CSV format | scorer pipeline through validator |
| Each structure's prediction depends only on its own supplied coordinates and node indices plus trained model | judge subset probe and patch audit |
| Fixed LorDG inputs; no contact labels during inference; fresh fitted state from permitted training data | patch audit; inputs exclude contact labels |

Guard violations produce `dscc: Infinity`, the benchmark rejection signal: it counts as a failure, not a high score. Diagnostic `scientific_score` preserves the scientific rule's worst value -1. The subset probe checks predictions with companion structures removed, at absolute tolerance 1e-6. An output file cannot establish that a predictor respected the input-access rule, so that rule rests on patch audit.

Validation baseline (three seeds, eight CPUs): mean dSCC **0.809245416**, σ **0.002119711**, mean dRMSE **0.109719463**. Every seed is inside the val guard, with per-seed dRMSE from 0.106 to 0.114. The test guard is the stricter one, so a candidate close to 0.12 on val may still fail on test. The judge uses three training seeds. PASS requires relative improvement of at least `max(delta, 2 × reference standard deviation / reference mean)`; delta is 1%, calibrated from seed variation. Each seed must pass the guards.

## 4. Baseline

IG-LorDG uses the inertial frame, geometric Transformer, Nyström coordinate features and RoPE. The inertial frame centers each structure and rotates it onto the principal axes of its inertia tensor, with a sign correction for chirality. RoPE encodes the canonical coordinates in attention. Nyström features are low-rank RBF similarities to uniformly sampled anchors at several bandwidths. The loss combines a bidirectional KL neighborhood term and a rank-weighted MSE on normalized distances.

Official repository: https://github.com/yize1203/InertialGenome, upstream commit `1ae3794280de0303714c50c054422d942b1096eb`. The baseline refinement model, training loss, internal preprocessing and training loop are in `code/InertialTransformer_Genome.py`; the CPU implementation is in `code/cpu_opt.py`.

The baseline uses max_len 2400, latent_dim 48, four heads, one layer, 12 sigma values, eight Gaussian kernels, 50 epochs, batch four, Adam lr 0.0005 and weight decay 0.01, hybrid alpha 0.5 and lambda 0.1. Checkpoint selection maximizes validation dSCC. Training initializes fresh weights and anchors.

On eight CPUs, maximum measured training time is **19.88 min** per seed and validation prediction time **8.03 s**. A full training run recorded a peak child RSS of **8.30 GiB**; memory allocation is 16 GB.

## 5. Interface (frozen)

- `bash code/train.sh --out DIR`: train and save all inference artifacts in DIR.
- `bash code/predict.sh --model DIR --input PATH --output FILE`: emit the exact CSV specified in `bench/io_spec.md`, at the supplied output path.
- Environment: `BENCH_SEED`, `BENCH_CPUS`, `BENCH_DATA_DIR`, `BENCH_BENCH_DIR`, `BENCH_CODE_DIR`. Honor seeds and thread budgets.
- `bench eval-dev` accepts `--seed S`, and `bench validate-output FILE` checks a prediction file.
- Dependencies are pinned in `code/env.lock`. Only exact PyPI/PyTorch CPU pins are rebuildable, with no editable, Git or local installs.

## 6. Rules

- Everything under `code/` may change, including model, loss, training and hyperparameters, while preserving the interfaces and input rules.
- `data/`, `bench/`, this README, prediction format and script arguments are frozen.
- The judge runs on eight CPUs with a 16 GB allocation.
- No network during training or prediction (sandbox). A fresh model must be trained each seed; honor `BENCH_SEED` and retain all needed artifacts under `--out`.
- Do not read or infer held-out identities, recognize validation inputs to retrieve labels during prediction, hard-code outputs, embed memorized paper results, or reuse fitted external artifacts (patch audit).

## 7. Known pitfalls

Emit all 2400 coordinate slots per structure in the required order, including padding: the authors' normalization denominator includes padding and this behavior is preserved. Constant coordinates make distances undefined and fail. Predict one structure independently of others. Inference inputs do not contain contacts. When using the baseline, save anchors with its model because they are not registered state-dict buffers. The harness passes absolute paths and enables `PYTHONSAFEPATH`; scripts must establish their import paths. The output filename may have no extension. Model selection uses validation dSCC, while the dRMSE guard is an additional acceptance condition.

## Research scope

**Objective:** Refine the supplied LorDG chromosome structures to improve contact-derived geometric agreement.

**Fixed research objects:** The supplied LorDG coordinates, node indices, splits and contact-derived target distances are fixed, because they define the problem the refinement is compared on. The equal-structure dSCC objective, float32 padded 2400-slot distance normalization and dRMSE at most 0.095 on test and 0.12 on val remain fixed, so a candidate is measured the same way as the baseline.

**Editable methods:** The refinement model, internal coordinate features, training surrogate, optimizer and hyperparameters may change; inference uses only the current structure and trained model. The paper's own ablations vary the loss weights and remove inertial alignment, RoPE or the Nyström branch, so these are method choices.

**Forbidden substitutions:** Real-locus coordinates must come from a trained refinement model; returning the supplied coordinates unchanged, or changed only by a rigid motion or a global scale, is not a valid submission. Do not substitute another initial structure source, use contacts as prediction inputs, or change normalization, padding, metric or guard. Do not train or fit reusable model state or learned preprocessing on validation features or labels, including final train+val refits or validation-based distillation.

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
