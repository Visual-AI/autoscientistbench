# DISSOLVR: aqueous solubility prediction

The judge retrains your code from scratch and compares predictions on a hidden
split against an archived baseline.

## 1. Goal

Predict aqueous solubility as log10(mol/L), called logS, from molecular SMILES.
This task uses DISSOLVR regime I (one fixed solvent, water) on curated AqSolDB.
Improve `rmse` (lower is better), defined as the mean RMSE of five independently
trained models. Validation uses scaffold holdout. You may change the method
subject to the rules below.

**Research question.** Does state-of-the-art solubility prediction require
non-transparent deep architectures, or can explicit descriptors reach it? The
paper's answer, DISSOLVR, maps each molecule to a fixed vector of simple, named,
physically interpretable descriptors and fits gradient-boosted trees on it.

**Experiment.** The paper trains all methods on identical splits and
reports mean RMSE over five seeds. On AqSolDB it compares DISSOLVR with empirical
equations (GSE, ESOL), published models with their own featurizers (Chemprop,
fastprop, AqSolPred and others), graph networks, and generic regressors (random
forest, XGBoost, LightGBM, a shallow neural network) given the same 176
descriptors. DISSOLVR has the lowest RMSE there. The paper estimates an irreducible
noise floor of about 0.6 to 0.8 logS units for data like AqSolDB, from
inter-laboratory disagreement.

**What follows for this task.** Molecules, split, units and the five-model metric
stay fixed so that a candidate is comparable with the baseline. The representation
stays explicit because the paper's claim is about explicit descriptors; a learned
or hashed representation would answer a different question. Regressors may change
because the paper itself compares regressors on one descriptor set. The specific
176 descriptors and CatBoost are the baseline, not fixed objects.

## 2. Data

- `data/train/train.csv`: 5984 molecules, columns `id,SMILES,y`; y is logS.
- `data/val/input.csv` and `data/val/labels.csv`: 1496 validation molecules,
  columns `id,SMILES` and `id,y`. Validation holds out whole Bemis-Murcko
  scaffold groups from the released training set, using fixed seed 20261030.
  A shuffled greedy group assignment targets 20% of training rows. Empty
  scaffolds form one group; unparseable SMILES use an exact-string group.
- The empty-scaffold group (1861 acyclic molecules) stayed in training, so
  validation has no acyclic molecules. Mean logS is -3.02 in training and -4.06
  in validation.
- The test split is held by the judge and is never readable from this workspace.
- The curated table has no multi-component SMILES (no `.`), and its logS values
  lie between -13.5 and 0.5.
- IDs are original curated-table row numbers; all tables are sorted by numeric id.
  Preserve the released SMILES and labels. Do not re-split, relabel, download,
  or add outside data. Validation features and labels are for evaluation and method,
  hyperparameter or checkpoint selection only.
- **Only train.csv labels may be used to fit models. Final training on train+val
  is not allowed.** This follows the benchmark-wide train-only fitting rule for both baseline and candidates.

## 3. Metric and scoring

`bench/scorer.py` computes the mean of five values
`sqrt(mean((pred_j - y)**2))` on the original logS scale. It does not average
predictions before computing RMSE. Invalid output or any guard violation receives
Infinity, the worst score.

| Guard | Enforcement class |
|---|---|
| Five finite prediction columns, correct IDs, valid schema | scorer and validator |
| Train-only label provenance, train_only=1 | scorer and patch audit |
| Fit only training-file labels; truthful metadata; no fitting during predict | patch audit |
| Explicit, named descriptors; no learned or hashed molecular representation | patch audit |
| Five independent models on the supplied seeds; no outside data | patch audit |

A training-time declaration cannot prove the elapsed time or label provenance, so
external timing and patch review apply. Frozen `bench/run_train.py` measures
training including imports, features, all five models and saving; keep train.sh
routed through it.

The judge runs three repetitions, each containing five models. Delta is 0.5%; `bench status` shows the archived validation score. PASS requires relative improvement
at least `max(delta, 2 * reference_std / abs(reference_mean))`.

## 4. Baseline

The official implementation is https://github.com/VanshRamani/Dissolvr at commit
`bc6c3f42b3fcfc23760289fa60076a1842033f34`. `code/` contains the baseline adapted to the frozen interface.

`code/regime-i/featurizer.py` is byte-identical to upstream: RDKit descriptors,
MOSE graph counts, Joback melting-point proxy and Abraham proxies, giving 176
features with tautomer canonicalization. The paper groups them as compositional
(109: size, atom and ring counts, `fr_*` functional-group counts), topological
(19: graph indices, MoSE cycle, path and star counts), energetic (6: Joback and
Abraham proxies) and physicochemical (42: EState and partial charges, surface
areas, MolLogP, QED). In its AqSolDB ablation, removing the physicochemical group
hurts most and removing any other group hurts much less.

`code/regime-i/train.py` fits five CatBoost regressors with 10000 iterations,
learning rate 0.02, depth 8 and l2_leaf_reg 5, without early stopping. Invalid
molecules retain the authors' zero-filled feature behavior.

Compared with upstream, the baseline reads materialized data, saves all five
models, loads no evaluation labels, selects no best model, predicts supplied
inputs, and reads seeds and the CPU budget from the environment. Four independent
fits can run concurrently with four threads each. Models, losses, features and
hyperparameters are unchanged. Archived validation RMSE is **0.92617157 ± 0.00147894** (population standard deviation across three five-model repetitions). On 16 CPUs, full training takes about 5.1 minutes and predicting the validation set takes about 22 seconds.
Memory allowance is 16 GB; peak memory has not been measured here.

## 5. Interface (frozen)

- `bash code/train.sh --out DIR`: train and save everything needed for inference.
- `bash code/predict.sh --model DIR --input PATH --output FILE`: predict the
  supplied molecules. Exact CSV schema is in `bench/io_spec.md`.
- Environment: `BENCH_SEED`, `BENCH_CPUS`, `BENCH_DATA_DIR`, `BENCH_BENCH_DIR`
  and `BENCH_CODE_DIR`. Paths supplied by the judge are absolute.
- `bench/run_train.py` runs `code/regime-i/train.py --out DIR`, reads
  `DIR/label_usage.json` and writes `DIR/train_meta.json`. It sets `train_only` to 1
  only if `label_sources` in that file is exactly `["train/train.csv"]`. The
  baseline predictor copies `train_s` and `train_only` from `train_meta.json`.
- Only PyPI `name==version` pins are supported for this task.

## 6. Rules

- Editable surface: explicit, named molecular descriptors computed from the SMILES, regressors
  on those descriptors, training surrogate losses and optimization under `code/`, within the research scope below.
  Preserve the script arguments, five-model experiment, metadata and output format.
- Frozen surface: `data/`, `bench/`, this README, the prediction schema and
  interface signatures. Patches touching them are invalid.
- Resource allowance: 16 CPUs and 16 GB memory.
  Runtime has no network; all required data and code must be present locally.
  The judge replays all repetitions under one overall budget, separate from your
  working budget; a run that exhausts it does not pass.
- The five base model seeds are 42, 101, 123, 456, 789. For a repetition add
  `BENCH_SEED - 1` modulo 2**32 to each; honor the CPU budget as well.
- Fit from scratch using only train.csv labels. Do not fit during prediction,
  encode memorized results, guess hidden rows, read evaluation labels, bypass
  timing, or fabricate metadata. Validation is for evaluation and method, hyperparameter or checkpoint selection only.

## 7. Known pitfalls

The five-model metric differs from ensemble RMSE. Keep five separate columns.
Prediction output paths can have no extension; write exactly the requested path.
NaN, infinity, duplicate/missing IDs and unsorted rows are rejected. The prediction
file must include constant training metadata even for small subsets.

RDKit versions affect descriptor definitions and molecule parsing; the baseline
pins its reproduction version. Some released SMILES do not parse; preserve the
baseline behavior when reproducing it. No weights, feature cache or full raw
corpus is supplied. Set PYTHONPATH explicitly, because PYTHONSAFEPATH is enabled.
The standalone predictor must work without evaluation labels or writable data.

## Research scope

**Objective.** Predict aqueous solubility in log10(mol/L) from the supplied molecular SMILES, minimizing the mean of five independent model RMSEs.

**Fixed research objects.** Released SMILES and solubility observations, fixed scaffold validation split and fixed evaluation partition; only train.csv labels for fitting, validation labels for selection, no fitting during prediction and five independent supplied-seed models; original logS units and mean of five separate all-row RMSEs.

**Editable methods.** Explicit, named molecular descriptors computed from supplied SMILES, regressors on those descriptors, training surrogate loss and optimizer; feature processing, model-selection procedure and hyperparameters using only permitted fitting labels.

**Forbidden substitutions.** Each prediction column must come from one regressor fit with its one supplied seed; averaging several independently seeded fits inside a column counts as substituting ensemble RMSE. Checked by patch audit, because the scorer sees only predictions. Do not replace explicit descriptors with a learned or hashed molecular representation, such as a graph neural network, SMILES model, pretrained embedding or hashed fingerprint. Do not change the molecules, the solubility target or units, or the split. Do not fit on validation labels or prediction batches. Do not substitute ensemble RMSE, duplicate one fitted model across seed slots, use external data or look up answers by row ID. Do not train or fit reusable model state or learned preprocessing on validation features or labels, including final train+val refits or validation-based distillation.

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
