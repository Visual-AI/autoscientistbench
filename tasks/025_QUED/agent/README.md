# QUED: LD50 acute-toxicity regression

Predict acute oral toxicity in log10(mol/kg) from the designated lowest-energy
DFTB3+MBD BOB+DQM observations. Minimize mean absolute error over every required
molecule. RMSE is a separate diagnostic.

The source paper asks whether electronic structure features from semi-empirical
DFTB3+MBD, the DQM descriptor, help predict biological endpoints once concatenated with
a cheap geometric descriptor. For acute toxicity it compares KRR and XGBoost over BOB,
SLATM, DQM and their combinations, and reports XGBoost on BOB+DQM as its best model.
This task fixes that descriptor pair, molecule population and endpoint, and
leaves the regression method open, because the paper varies the regressor itself.

## Data and scientific population

The benchmark keeps its existing 4000 training and 1000 validation rows, drawn from
the paper's 5000-row training pool with seed 20261101 and original row order retained.
The authors selected that training pool by farthest point sampling. Development and
formal replay both fit only the same 4000 training rows. The 1000 validation rows
remain held out from every fit, including final retraining. The different KRR
partition is not used. The benchmark's training population therefore differs from
the paper's full 5000-row fit.

Public training features and labels are `data/train/input.h5` and `labels.csv`;
validation features and labels are `data/val/features.h5` and `labels.csv`.
`data/val/replay/` supplies the same public development experiment to the evaluator.
Features are immutable float64 arrays: 39340 original BOB columns followed by 213
unscaled DQM columns of DFTB energies, dipole norm, orbital energies and zero-padded
Mulliken charges. `data/descriptor.json` specifies their order. Feature arrays are
compressed and read in small batches; do not allocate all molecules at once.

## Baseline and measured reference

The reference method is a fresh XGBoost regressor with the stored hyperparameters:
4000 trees, maximum depth 11, learning rate 0.008, minimum child weight 2, subsample
0.7 and column subsample 0.5, plus the regularization values in `code/config.json`.
It uses CPU histogram training and the benchmark seed. The authors tuned those values
with Optuna against five-fold cross-validated RMSE on their own 5000-row fit, not
against MAE on these 4000 rows. No author estimator, fitted scaler or archived
predictions are included, so the baseline fits from scratch, fitting both the DQM
scaler and the full-descriptor scaler on active training rows only and applying them at
evaluation. Constant training columns are omitted internally because they cannot split
a tree, which can change stochastic feature subsampling relative to the source
estimator. All training rows remain present.

On validation the baseline scores MAE 0.4526, standard deviation 0.0013 over the three
judge seeds, and RMSE about 0.618, at the declared 16 CPUs. Nearly all of the wall
clock is the isolated fit: about 35 minutes per development replay, with prediction in
a few seconds. The outer `train.sh` and `predict.sh` take about two seconds together.

## Interface and environment

`bash code/train.sh --out DIR` writes the training recipe, and
`bash code/predict.sh --model DIR --input PATH --output FILE` exports the candidate
source and recipe. The trusted evaluator invokes `code/policy.sh` in a fresh isolated
fit worker that fits on the supplied training rows. Its inert fitted-state archive then
goes to a separate prediction worker with the evaluation features, which returns
`id,pred` CSV. The outer output is an inert ZIP. See `bench/io_spec.md`.

`BENCH_SEED`, `BENCH_CPUS`, `BENCH_DATA_DIR`, `BENCH_BENCH_DIR` and `BENCH_CODE_DIR`
define runtime resources and paths. The declared budget is 16 CPUs and 32 GB memory.
The copied Python 3.11.2 CPU runtime is mounted read-only at
`/opt/arb/runtime/python`; `code/env.lock` records its pinned distributions and is
immutable for this runtime snapshot. Its native libraries require Linux x86_64 and
system Python 3.11. Both isolated workers run inside that same runtime with no network,
and only `code/`, the recipe and the supplied input reach them, so a method must be
built from the locked distributions; packages installed elsewhere in the workspace are
not importable there. QML is used only by trusted data preparation, not during fitting.

## Research contract

Objective: predict acute toxicity in log10(mol/kg) from the designated lowest-energy
DFTB3+MBD descriptor observations, minimizing all-molecule MAE with RMSE as a separate
diagnostic.

Fixed research objects: all designated lowest-energy conformers and the fixed raw
39340-column BOB plus 213-column DQM descriptor definition; the designated evaluation
population and one prediction for every evaluation observation; the fixed 4000
training and 1000 validation division within the original 5000-row training pool,
with final fits on those 4000 training observations only; the acute-toxicity endpoint
in log10(mol/kg), complete-population MAE and the separate RMSE diagnostic; the script
interfaces; and isolation of evaluation labels during replay. Geometry, conformer
choice, descriptor definition, target units, data, README, trusted benchmark helpers
and the environment lock are frozen.

Editable methods: you may change the regression estimator, training surrogate loss,
optimizer, hyperparameters and internal feature processing, retaining the same frozen
BOB+DQM input and LD50 objective.

Forbidden substitutions: the incompatible KRR data partition, a different conformer or
descriptor source, another biological endpoint, external data, external pretrained
estimators, hidden labels, raw-data retrieval, memorized evaluation predictions,
omitted observations, a changed evaluation population, and fitting reusable model state
or learned preprocessing on validation features or labels.

Guards: the validator checks that the outer artifact is an inert archive for this task
and protocol. The scorer rejects a replay pack whose fit population is not the 4000
authorized rows, whose descriptor width is not 39553, or whose ids are duplicated or
shared between fit and evaluation. It then requires one finite prediction per evaluation
id in input order, and computes MAE with a separate RMSE. The sandbox keeps evaluation
features and labels out of the fit worker. Patch review remains necessary for arbitrary
edited methods to use every authorized training row, retain training-only preprocessing
and avoid deferring fitting to the prediction phase.

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
