# HSG-12M: Spatial Multigraph Classification

The judge retrains your code from scratch and compares its predictions with an archived baseline.

## 1. Goal

Research question: how well can a graph learning method identify a crystal's Hamiltonian family from its energy-spectrum graph under limited capacity and training work? The energy spectra of one-dimensional non-Hermitian crystals form arcs and loops on the complex plane. The paper "HSG-12M" (ICLR 2026) represents them as spatial multigraphs: nodes and edges have positions in the plane, and geometrically distinct edges between the same two nodes stay separate.

The paper's experiment compares eight GNNs (GCN, GAT, GATv2, GIN, GINE, MF, CGCNN, GraphSAGE) by test metrics over three seeds. For a fair comparison, the authors tune each model's convolution width so that trainable-parameter counts agree within 4%, and they cap every model at 100 epochs and 1,000 steps. GraphSAGE ranks first. The authors leave hyperparameter optimization to future work and name richer encodings of edge geometry as a promising route.

This task is the one-band × GraphSAGE cell of the paper's main results table: classify one-band energy-spectrum graphs into 24 Hamiltonian classes. Improve `accuracy` (higher is better) with the fixed graph inputs/classes and the controlled capacity/training-work budget: at most 731,203 real trainable parameters, 1,000 optimizer updates and 100 started epochs.

## 2. Data

- `data/train/input.npz` and `data/train/labels.csv`: 158,995 training graphs and their class labels.
- `data/val/input.npz` and `data/val/labels.csv`: 19,874 validation graphs with public labels, used for model selection and development scoring.
- The test split is held by the judge. The original runtime split and graph-size batch rebalancing are materialized once and shared across all training runs.
- Each class is one family of characteristic polynomials. Graphs in a class differ by the values of two free complex coefficients. The classes are balanced, so chance accuracy is 1/24.
- Inputs contain packed graph tensors and opaque ids; node features are coordinates, potential and density of states. Edge features encode weight, displacement, midpoint, average potential and average density of states. Validation graphs have 2 to 13 nodes and 2 to 30 directed edges. Details are in `bench/io_spec.md`.
- Fit only on the supplied training graphs. Use validation features and labels only for evaluation and method, hyperparameter or checkpoint selection. Do not re-split, relabel or add external data or pretrained weights.

## 3. Metric and scoring

`bench/scorer.py` computes overall Top-1 accuracy: the number of correct predicted classes divided by the number of unique supplied graphs. This is micro accuracy. Each evaluation example is counted once, including when distributed sampling needs padding. The final score is the arithmetic mean across three fits.

| Constraint | Enforcement class |
|---|---|
| Exactly one finite class prediction per supplied id | scorer pipeline through `validate.py` |
| At most 731,203 real trainable parameters, 1,000 updates and 100 started epochs | scorer checks declarations; model probe checks parameter artifacts; patch audit checks actual work |
| From-scratch fitting, training data only, validation-based selection, honest artifact records | patch audit |

Prediction-table columns declare these: `parameters` counts all trainable parameters used in the fit, `steps` counts updates to the fitted model across all ranks, with a synchronized DDP step counted once, and `epochs` counts epochs started, including a final partial epoch. A protocol violation returns `Infinity` and fails the submission. The model probe `bench/check_model.py MODEL_DIR --pred PRED_FILE` verifies the model certificate. Artifact hashes do not establish data provenance; source review is required.

Baseline validation accuracy is **0.8518499883**, population standard deviation **0.0045671896**, over three fits. Delta is **0.300%**. PASS requires relative improvement at least `max(delta, 2 × reference standard deviation / reference mean)` on the hidden split. With the validation numbers above that noise term is about 1.07%, above delta.

## 4. Baseline

The official classification implementation is https://github.com/sarinstein-yan/HSG-12M at commit `ec591130a94116a1a45ed3e896cf5891ed047171`, accompanying https://github.com/sarinstein-yan/poly2graph. The model is in `code/src/hsg/gnn_baselines.py`; loss, optimizer and validation are in `code/src/hsg/training.py`; the fit entry point is `code/bench_train.py`.

GraphSAGE uses four GNN layers of width 330 and two MLP layers of width 128, global add pooling, zero dropout, CrossEntropyLoss, AdamW with AMSGrad, learning rate 0.001, zero weight decay, and CosineAnnealingWarmRestarts with T0=100 and minimum learning rate 0.00001. This gives 703,080 trainable parameters. Its message passing reads only node features and `edge_index`: edge attributes reach the model but `SAGEConv` does not use them. The paper uses the same layer counts, batch size, optimizer and schedule for all eight models.

The baseline uses two synchronized CPU DDP ranks and 6,000 graphs per rank per batch. Training stops at 1,000 updates, in the 72nd started epoch. Checkpoint choice maximizes validation macro-F1 after each complete epoch. The authors' final partial epoch does not trigger an extra validation.

Compared with the authors' code, the baseline commit materializes data entry, adds train/predict and saved-weight certificates, separates validation selection from final prediction, removes downloads and sweeps, and honors seed/thread variables. The model, loss, optimizer, schedule, batch size and training budget are unchanged. Packed batching matches the authors' collation exactly.

Runtime measurements use a 128-CPU allocation, an upper bound; each DDP rank uses up to 20 threads. Measured training times are 12.55, 12.79, 12.64 minutes per fit; prediction on the validation graphs takes 9.63, 9.48, 9.58 seconds per fit.

## 5. Interface (frozen)

- `bash code/train.sh --out DIR`: fit from scratch and save all prediction artifacts into DIR.
- `bash code/predict.sh --model DIR --input PATH --output FILE`: predict the supplied graphs to a CSV table.
- Environment: `BENCH_SEED`, `BENCH_CPUS`, `BENCH_DATA_DIR`, `BENCH_BENCH_DIR` and `BENCH_CODE_DIR`.
- In `code/env.lock`, only exact `name==version` pins from PyPI/PyTorch CPU indexes are rebuildable.
- Export `model_card.json` and the fitted state according to `bench/io_spec.md`, then run the model probe on predictions. The baseline writes `weights.pt` and per-rank training traces.

## 6. Rules

- You may change the model, features, loss and optimization under `code/`, subject to the fixed capacity/training-work budget, unchanged inputs, labels, script arguments and output format. The research task is controlled-capacity graph classification; GraphSAGE itself is a baseline architecture, so another model or training surrogate may be used within those same controls.
- `data/`, `bench/`, this README and the interfaces are frozen. A patch changing them is invalid.
- Honor `BENCH_CPUS` across all processes and numerical libraries. Memory budget: 32 GB. Runtime has no network.
- Each harness seed maps to a training seed through the mapping in the supplied code. Preserve this mapping and let the seed govern initialization and training randomness. The supplied data split never changes with the training seed.
- Reading or reconstructing hidden targets, hard-coding predictions, memorizing paper outputs, falsifying model metadata, or fitting on validation targets is disqualifying (patch audit).

## 7. Known pitfalls

- Lightning's `MulticlassAccuracy` defaults to macro averaging; the frozen scorer uses overall accuracy. Validation checkpoint selection remains macro-F1.
- Graph edge indices are local. Batch construction must offset them by cumulative node counts.
- Do not include the labels in a prediction input or depend on an original dataset index. The ids are opaque strings.
- CSV output paths may have no extension. All ids must appear exactly once. Repeated distributed-sampler padding rows are invalid.
- `PYTHONSAFEPATH=1` is enabled. Both entry scripts export the code directory and `src/` on `PYTHONPATH`.
- Retraining uses a clean clone with your patch applied. Save fitted state under `--out`; only source files in the patch are available.
- The model certificate binds predictions to saved fitted state. Tensor models may use the `hsg-model-v1` manifest; other algorithms may save a fitted-state artifact with the generic certificate and a complete numeric learned-parameter manifest. Fitted constants cannot be relabeled as fixed buffers to evade the capacity budget. A zero count is valid only when that work or fitted parameter genuinely does not exist.

## Research contract

**Objective.** Maximize one-band spatial-multigraph Top-1 accuracy under the original controlled capacity and training-work budgets. These budgets make the paper's comparison meaningful; a gain from more capacity or work does not answer the question.

**Fixed research objects.**

- The supplied graph inputs and features, the 24 Hamiltonian classes, the supplied split, and the from-scratch fitting and data-use rules.
- At most 731,203 real trainable parameters (1.04 × 703,080, rounded down), 1,000 optimizer updates and 100 started epochs.
- Micro accuracy, three-seed aggregation and validation-based checkpoint selection.

**Editable methods.** GNN or other classification architecture, learned feature processing, surrogate loss, optimizer and training strategy, all within the same capacity and work controls. The settings in section 4 describe the baseline; the controls are the ones listed above.

**Forbidden substitutions.** The 1,000 limit counts updates to the fitted model across all ranks: a synchronized DDP step counts once, and asynchronous, local-update or separately trained workers each add their own updates to the total.

- Increasing model capacity or training-work budget to replace the controlled comparison (scorer checks declarations; patch audit checks actual work).
- Using hidden or external labels or pretrained weights, changing graph inputs or labels, or changing the scoring objective (patch audit).
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
