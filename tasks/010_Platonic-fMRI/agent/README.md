# Platonic Representations in the Human Brain

## 1. Goal
Research question: can a subject-specific encoder, learned only from repeated fMRI presentations of the same images, extract a stable image-level signal while keeping representational geometry stable across repetitions?

The task follows the paper's Table 2 self-supervised encoder comparison. Table 2 tests encoders within each subject: the embedding of one repetition of an image must retrieve the same image from another repetition, and the representational geometries of the two repetitions must agree. The paper reports both measures, and they can diverge: its multi-view methods improve instance-level matching but yield low RSA.

Your retrained method is compared with the paper's full encoder, "Ours (full)", run under the same protocol and scored on the same hidden trials. Improve within-subject image retrieval `recall_at_1` (higher is better), retaining stable representational geometry. The hidden evaluation split is held by the judge. You may change the method within the rules below.

## 2. Data
- `data/train/subject_S.npz` contains raw int16 ROI response matrices, trial ids and acquisition run codes. `stimulus_S.csv` gives training image groups and repetition metadata, including missing presentations.
- `data/val/input/subject_S.npz` and `data/val/labels.npz` are your public validation set. Validation is a fixed 20% holdout of the nonshared image groups, keeping repetitions together. All response values are unchanged.
- Validation uses nonshared images; candidate counts and image composition can differ between validation and evaluation.
- Inputs omit image identities and repetition correspondences. See `bench/io_spec.md` for shapes and labels.
- Both baseline and candidate models fit on train only. Validation may guide model selection, including selection before a final train-only fit. Do not use external data, reconstruct hidden rows, or alter frozen splits.

There are 153,502 training trials and 38,380 validation trials across eight subjects. Subject-specific voxel counts range from 73,520 to 104,603. Complete three-repetition validation groups, which are the retrieval candidates, range from 982 to 1,800 per subject.

## 3. Metric and scoring
`bench/scorer.py` computes cosine R@1 and Pearson RSA for the three repetition pairs within each subject, then averages subjects and pairs equally. Only groups with all three observations enter scoring. RSA correlates upper triangles of the two Pearson-distance matrices. Only `recall_at_1` is compared with the baseline; `mean_rsa`, `raw_recall_at_1`, `mean_rank`, `max_embedding_dim`, `guard` and `per_pair` are diagnostics.

| Guard | Class | Enforcement |
| --- | --- | --- |
| Mean RSA >= 0.51 on test, >= 0.48 on val | scorer | Computed from submitted embeddings |
| Embedding dimension between 1 and 128 | scorer and validator | Checked from every actual subject embedding array |
| Each trial encoded independently of evaluation batch; model immutable during prediction | patch audit | Public probe: `python3 -B bench/check_consistency.py MODEL --input INPUT --pred PRED --code code` |
| No external data, hidden-label inference, id-based embeddings or memorized answers | patch audit | Code review and isolation |

On a guard violation, the mathematical score is zero: `score_rule_value` is 0 and `recall_at_1` carries the harness failure sentinel `Infinity`. `Infinity` means failure, not a high score. Nonfinite geometry also fails.

Validation baseline over seeds [1, 2, 3]: R@1 0.736163 (sigma 0.001450); mean RSA 0.487594 (sigma 0.000334). All three seeds clear the validation guard. Validation and evaluation use different image cohorts and different RSA thresholds, so clearing 0.48 on validation does not prove clearing 0.51 on evaluation. The scorer picks the threshold from the labels path: labels under a `val` directory use 0.48, any other path uses 0.51.

The task-specific delta is 0.4%. The judge retrains three seeds and requires relative improvement >= max(delta, 2 * baseline seed standard deviation / mean).

## 4. Baseline
Official repository: https://github.com/memory-formation/platonic-representations-fmri at `e28c35a00656354b5e34ec8b85223c508b1c11ef`. A sanitized source snapshot is under `code/`; result notebooks and artifacts were removed. The adaptation retains the reproduction's CPU implementations.

The baseline fits clipping/centering, session/run residualization, voxel reliability weights, sample-weighted PCA (768), MCCA (128), and ridge projector distillation. It then trains a residual MLP (128→768→128, GELU, dropout 0.1, learnable residual scale initially 0.25) for 2,000 steps with Adam, batch 512, learning rate 1e-4, weight decay 1e-4, temperature 0.09, InfoNCE plus 0.5 pull-cosine, and the original schedule and gradient clipping. Model and loss live in `code/src/fmri_mapping/embedding/`; the original fitting functions are in `code/scripts/`.

This pipeline is the baseline method, not a fixed research object. The paper itself varies PCA dimension, embedding dimension, hidden size and MLP depth (Appendix B.3) and ablates encoder components (Appendix B.2). It uses run/session residualized responses as input for every main-text method, because raw responses show acquisition block structure (Appendix A.1).

Changes from the upstream code: train-only fitted preprocessing, materialized group split, final-step checkpoint, offline logging, environment seeds/threads, new save/predict entry points. The original loss loop, including its variable reuse, is retained. No fitted weights are supplied. Eight subject fits run independently; linear workers use up to 16 threads each and MLP workers up to 4, within the 128-CPU budget.

Measured reference runtime at 128 CPUs: training 6.16 min on average (range 5.70 to 6.65); validation prediction 22.98 s.

## 5. Interface (frozen)
- `bash code/train.sh --out DIR` writes everything required for inference into DIR.
- `bash code/predict.sh --model DIR --input PATH --output FILE` writes the NPZ prediction at the exact requested path. Input PATH is a directory.
- Honor `BENCH_SEED`, `BENCH_CPUS`, `BENCH_DATA_DIR`, and the read-only `BENCH_BENCH_DIR` when provided.
- `bench eval-dev --seed S` runs one seed; `bench validate-output FILE` checks a prediction file.
- Dependencies are pinned in `code/env.lock`; `uv pip install` changes are frozen on submission. Only PyPI/PyTorch CPU name==version pins are rebuildable.

## 6. Rules
You may change the compact fMRI encoder, derived preprocessing, self-supervised training surrogate losses and optimization under `code/`, within the research contract below. Data, bench, README and the I/O specification are frozen. No network is available during train/predict. The judge retrains with three independent seeds; honor the provided seed and CPU budget. CPU allocation is 128. Memory reservation is 384 GiB; a full baseline run measured a process-tree RSS peak of 160.20 GiB.

Encode each trial using its own response, acquisition run, subject and the saved model. Ids may only associate outputs with inputs. Do not exploit batch membership, match observations jointly, fit on evaluation inputs, or rewrite model artifacts during prediction. Run the public consistency probe after changing prediction code.

## 7. Known pitfalls
- NPZ files must be written through an open file handle: NumPy otherwise appends an unwanted extension to the output filename.
- Voxel dimensions vary by subject and matrices are large; avoid duplicating all subjects' working arrays in one process.
- All fitted preprocessing statistics belong in the model directory. Inference must reuse those statistics.
- The baseline retains absent repetitions and imputes a missing third view during training. Alternative fitting methods may handle missing views differently while preserving supplied trial/group metadata and the fixed scoring population.
- Validation retrieval candidate counts differ by subject. Absolute validation R@1 is therefore not a direct estimate of hidden retrieval R@1.
- Scripts export their own package path because the harness sets PYTHONSAFEPATH=1. Data and bench are read-only; write artifacts to the requested output directory or temporary storage.

## Research contract

**Objective.** Learn compact subject-specific representations of repeated fMRI responses, maximizing within-subject retrieval while preserving the original representational-geometry requirement.

**Fixed research objects.**
- Existing fMRI trial responses, subject/run metadata, image-group split and eight-subject evaluation population. These define the retrieval problem, so baseline and candidates face the same trials.
- Embedding dimension 1..128, independent per-trial encoding by a fixed trained model, complete repetition groups and three repetition-pair averages. These keep the score a property of a compact encoder, not of matching within a batch.
- Original cosine R@1, Pearson-distance RSA and mean RSA >= 0.51 on test and >= 0.48 on val. Retrieval without the geometry guard is not the target.

**Editable methods.** Subject encoder architecture, derived preprocessing, self-supervised training surrogate losses and optimization; training-time handling of missing repetitions without changing supplied trial/group metadata or evaluation membership. The embedding dimension is your choice within 1..128.

**Forbidden substitutions.** Use the acquisition run only to remove acquisition confounds fitted on training trials; an embedding must not carry run, session or trial-order information as a retrieval feature.
- Escaping the compact-representation problem with embeddings above 128 dimensions (validator and scorer) or changing response observations/repetition groups (frozen data and labels).
- Joint evaluation-batch matching/adaptation, identity-derived representations or model refitting during prediction (patch audit and public probe), or changing RSA/retrieval definitions (frozen scorer).
- Training or fitting reusable model state or learned preprocessing on validation features or labels, including final train+val refits or validation-based distillation (patch audit).

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
