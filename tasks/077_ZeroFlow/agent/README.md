# Zero-Flow Encoders

## 1. Goal

**Research question.** Can a flow-based, non-parametric criterion recover the conditional-dependence graph of a non-Gaussian graphical model from samples alone? The paper's answer is the zero-flow encoder. This task asks how far its graph recovery can be improved on nonparanormal data.

**Zero-flow encoder.** A rectified flow trained on independent samples of two distributions has zero velocity everywhere at t = 0.5 if and only if the distributions are equal. The paper extends this to conditionals: for a flow conditioned on Y and an encoding f(Y), the velocity at t = 0.5 vanishes if and only if p(X|Y) = p(X|f(Y)), that is, f(Y) is sufficient for predicting X. A zero-flow encoder is an f trained on the resulting least-squares loss: a rectified-flow term plus a penalty on the velocity near t = 0.5. For graph recovery, a mask m splits the variables into targets X and the rest Y, and f(y, m) = y * sigmoid(gate_net(m)). A sparsity penalty on the gates makes the selected variables an estimate of the targets' Markov blanket. The gate network takes the mask as input, so one encoder serves every target. Feeding the one-hot mask of each variable in turn gives a gate matrix, and its entries are the baseline's edge scores.

**Experiment.** The paper's Table 1 measures edge-recovery ROC AUC over 10 random trials on Gaussian, nonparanormal and truncated-Gaussian data from one graph. It compares zero-flow encoders with an MLP gate network and with an LSTM gate network, which adds a sequential inductive bias, against graphical lasso and the PC algorithm with Fisher's Z tests. All methods are scored by the same procedure against the same truth. This task keeps the nonparanormal row, and the baseline is its MLP zero-flow encoder cell.

**Task.** Each instance supplies observations of 50 variables. Fit a zero-flow encoder independently and return a full matrix of edge scores. Edge scores must come from a zero-flow encoder. Graphical lasso and PC are the paper's comparators; they and other classical estimators on the observations may not replace it here. Improve `edge_auc` (higher is better) on hidden evaluation instances. You may change the method within the rules below.

## 2. Data
- `data/train/input.npz` and `data/train/truth.npz`: four public development instances, data seeds 100–103. The baseline does not use them for shared training.
- `data/val/input.npz` and `data/val/truth.npz`: ten public validation instances, data seeds 0–9. Each contains 2048 observations of 50 variables.
- Hidden evaluation inputs and labels are unavailable during development.
- Data law: a latent Gaussian with a sparse precision matrix, then the coordinatewise transform sign(z)*abs(z)^3, then standardization. The transform makes the marginals heavy-tailed and keeps the conditional-independence graph. The precision matrix is fixed, so instances differ only in their draws. Its graph is a 3rd-order Markov chain in index order: an off-diagonal entry is nonzero exactly when abs(i-j) is 1, 2 or 3.
- `bench/generator.py` freezes the original Gaussian draw order, coordinatewise cubic transform, and standardization using each instance's observation mean and sample standard deviation. Inputs contain opaque IDs, observations, and public algorithm trial indices; row order is shuffled. Trial indices identify algorithm replicates.
- Public labels are for diagnostics and configuration selection only. Do not use them to supply graph predictions, fit a truth-table predictor, infer a data seed, or generate extra simulated data. Each prediction must depend only on its own observed instance and development-selected configuration. Do not re-split, relabel, or add external data.

## 3. Metric and scoring
`edge_auc` is the arithmetic mean of the original implementation's edge-recovery ROC AUC across supplied instances. The scorer flattens all 50×50 entries, thresholds absolute true precision at 1e-5 after zeroing its diagonal, and ranks absolute submitted scores. Both directions and diagonal negatives count. It retains the original NumPy sort behavior for ties; it does not symmetrize scores or average tied thresholds.

| Guard | Requirement | Enforcement |
|---|---|---|
| Valid predictions | Exact CSV columns and ID set, no duplicates, missing values, NaN or infinity | scorer (also validator) |
| Zero-flow edge scores | Edge scores come from a zero-flow encoder, not from a classical estimator on the observations | patch audit |
| Observations only | No fixed-graph lookup, truth-label fitting, seed inference, remembered predictions or extra simulation; each estimate uses its own observations | patch audit |
| Frozen protocol | Preserve data, evaluation code, script signatures and output format | patch audit; harness rejects forbidden patch paths |

There are no additional numerical constraints. Malformed predictions fail validation and the scorer reports `Infinity`. Public development labels describe the fixed source graph, and the scorer cannot distinguish a perfect estimate from copied truth, so preventing use of that truth requires patch audit. The zero-flow rule is also checked there, not by the scorer.

Measured validation baseline: **0.784519623602**, outer-seed standard deviation **0.004298258293**. The judge uses three outer algorithm seeds, compares their mean with the archived baseline, and requires relative improvement at least `max(0.02, 2*reference_std/abs(reference_mean))`.

## 4. Baseline
Official code: https://github.com/probabilityFLOW/zfe, upstream commit `8663ee88887f4440f6475b2ac7faa521c526887d`. `models/nntoy.py` defines the MLP encoder and vector field; `ToyLattice.py` contains the loss; `BaseExperiment.py` implements the optimization loop; `get_allgates` in `ToyLatticeROC.py` reads the gate matrix.

`bench_run.py` handles observation loading, configuration serialization, CSV output, seed handling and independent worker processes. `train.sh` writes configuration. `predict.sh` fits each instance from scratch for 5000 AdamW steps, batch 400, learning rate 1e-3, gate L1 coefficient 1e-9, and vector-field penalty weight 0.1. Each step draws two independent batches, a uniform t, and per-sample target masks: one random coordinate, plus each neighbor with probability 0.5. Model, loss, hyperparameters and optimization loop match the reproduced implementation. Checkpoint reuse, plotting, and an unrelated import-time demo are disabled. The original sampling upper bound of `2048-400` is preserved.

Measured at eight CPUs: configuration writing **0.51 s** maximum; validation prediction **1.98 min** maximum. The task uses eight workers, each with one numerical-library thread, and a configured memory budget of 16 GB (not a measured peak).

## 5. Interface (frozen)
- `bash code/train.sh --out DIR` writes everything prediction needs into DIR.
- `bash code/predict.sh --model DIR --input PATH --output FILE` fits the supplied instances and writes the CSV described in `bench/io_spec.md`.
- `BENCH_SEED` selects the algorithm stream; `BENCH_CPUS` sets the worker budget. `BENCH_DATA_DIR`, `BENCH_BENCH_DIR`, and `BENCH_CODE_DIR` identify the mounted locations. All script paths may be absolute.
- The environment is Python 3.8, with exact `name==version` dependencies rebuilt from `code/env.lock` using PyPI and the PyTorch CPU index. Use `uv pip install` in the development environment; submission freezes the installed dependencies. Local, editable and git installations are unsupported.

## 6. Rules

The research scope below determines which components may change. Supplied data and splits, trusted tests, information boundaries and the evaluation metric remain fixed.

- Frozen surface: `data/`, `bench/`, this README, script signatures and prediction format.
- CPU budget: eight. Memory budget: 16 GB. No network is available during train or predict.
- Every run must honor `BENCH_SEED`. In the baseline, outer seed 1 uses algorithm trial streams 0–9 on validation; later outer seeds select subsequent blocks. Data-generation seeds are independent of these algorithm streams. Source latent draws are discarded before initialization solely to preserve the algorithm RNG position.
- Disqualifying (patch audit): replacing the zero-flow encoder with a classical estimator on the observations, accessing or guessing hidden seeds or labels, using the fixed graph or a truth table as predictions, fitting predictions from public truth, hard-coding remembered outputs, extra simulated data, or using another instance's observations during prediction.

## 7. Known pitfalls
- Fitting occurs inside prediction. Budget development calls using prediction time; the short configuration-writing stage is not representative. The judge refits every hidden and validation instance for each of its three seeds under one overall evaluation deadline, with no separate per-call limit.
- Full-matrix AUC differs from an upper-triangle or symmetrized graph metric. Do not substitute sklearn's tie handling for the frozen scorer.
- The authors' final 400 observation rows are not sampled by the baseline loader; this behavior is deliberately preserved in the baseline. You may change it when improving the method.
- Write the requested output path exactly, even when it has no extension. CSV rows must contain all 2500 finite scores. Row order may vary, but IDs must match.
- The judge uses a clean clone of the submitted code. Keep required files committed, preserve `PYTHONPATH` setup under `PYTHONSAFEPATH=1`, and do not rely on local caches or data-generation files.

## Research scope

**Objective.** Recover the conditional-dependence graph independently from each fixed nonparanormal observation instance, maximizing the original full-matrix edge ROC-AUC.

**Fixed research objects.** Within a row of Table 1 these stay the same across methods, so AUC differences reflect the method.
- Gaussian/nonparanormal observation law, coordinatewise cubic transform, supplied standardized 50-variable instances and latent graph target.
- Observations-only independent-instance estimation and full 50x50 absolute-score AUC including diagonal negatives and original ties.

**Editable methods.**
- The zero-flow encoder's architecture, learned representations, regularization and training surrogate losses, and how edge scores are read from it.
- Numerical optimization, optimizers and internal observation representations within each instance, within the fixed nonparanormal problem and observations-only boundary.

**Forbidden substitutions.** Internal observation representations may not invert the cubic transform: cube-root, rank, quantile or other coordinatewise Gaussianizing maps of the observations are not allowed.
- Replacing the zero-flow encoder with a classical estimator on the observations, such as graphical lasso, rank correlation, partial correlation or neighborhood regression (patch audit).
- Replacing the source graphical model/distribution, using another graph metric or excluding difficult matrix entries (frozen data and scorer).
- Fitting a fixed-graph truth lookup, seed inference, extra simulations or cross-instance observation sharing (patch audit).
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
