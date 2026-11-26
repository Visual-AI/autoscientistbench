# TT-Sparse: Learning Sparse Rule Models with Differentiable Truth Tables

## 1. Goal

Predict diabetes risk from eight numeric Pima features using sparse extracted rule models. Improve `auc` (higher is better), the mean ROC-AUC of five independently trained rule sets, subject to mean rule complexity at most 14. The judge retrains your code from scratch and evaluates a hidden split with the same feature meanings and preprocessing as the public validation data. You may improve the method within the rules below.

## 2. Data

- `data/train/train.csv`: 491 training rows, with `id,preg,plas,pres,skin,insu,mass,pedi,age,target`. Features are raw pregnancy count, glucose, blood pressure, skin thickness, insulin, BMI, diabetes pedigree function and age. Targets are `tested_negative` or `tested_positive`.
- `data/val/input.csv` and `data/val/labels.csv`: 123 public validation rows. Labels contain `id,y` and the raw features; y=1 is the positive class. The duplicate features support frozen rule reconstruction during scoring.
- The internal validation split was materialized once by stratified sampling from the development data, using seed 0. Training and pruning use the training rows for gradient updates and this fixed validation split for selection only.
- `data/train/development_order.json` contains only public training/validation ids, preserving their preprocessing order. Ids are opaque and do not encode source row numbers.
- Fit only on the supplied training data; use validation only for evaluation and selection. Do not re-split, relabel, download the original dataset, load it from an installed package or add external training data. Validation labels may select checkpoints and pruning decisions. The baseline fits its numeric encoder and extracts rules using training rows only. Fitted transforms are applied to validation without updating their state.

## 3. Metric and scoring

`bench/scorer.py` computes five ROC-AUC values from the five probability columns, then averages the AUCs. Averaging probabilities first would define a different metric and is not allowed. The mean complexity of those five rule sets must be at most 14. Complexity follows the released implementation: 1 per intercept or skip term; each LTT rule costs 1 plus its number of `&`, `|` and `^` operators. A single replicate may exceed 14.

| Guard | Enforcement class |
|---|---|
| Mean extracted-rule complexity ≤14 | scorer, recomputed from certificate expressions |
| Probabilities equal the authors' `predict_rules` on the certificate | scorer, frozen recomputation with absolute tolerance 1e-12 |
| Certificate equals the rules saved during training and stays fixed across inputs | judge probe, `bench/check_model.py` |
| Rules are learned from permitted data by any method; independent seeds and fixed prediction-time certificate | patch audit |
| Exact ids/header, five finite probability columns and valid JSON certificate | scorer pipeline through `validate.py` |

A violation returns `auc: Infinity`, which the harness treats as failure; `paper_score: 0.5` records the original protocol's worst value. The saved `rules.json` and prediction certificate must agree and remain fixed across inputs.

The judge evaluates three outer seeds, each containing five independent replicates. The validation baseline has mean AUC **0.845445736434** and standard deviation **0.001804095988** across the three outer seeds. The configured delta is **0.50%**. PASS requires relative improvement at least `max(delta, 2 × reference standard deviation / reference mean)`; `bench status` reports the current delta and validation baseline.

## 4. Baseline

Official code: https://github.com/hansfarrell/tt-sparse at commit `5d60e785c65a002ff05f411761e4012cb8d925de`. `code/src/tt_sparse/model.py` implements the model, training and pruning; `encoder.py` provides thermometer encoding; `rules.py` and `qmc.py` extract and evaluate the rules.

The baseline uses 9 thermometer bits, fan-in 4, 30 nodes, tau 0.05, dropout 0.1, AdamW learning rate 0.005, batch size 1024, at most 200 epochs and validation patience 25. Pruning uses max_drop_pct=2, 30 finetuning epochs, at most 80 iterations and max_fanin=16. It preserves the released saliency pruning, continuous skip terms and XOR rule simplification.

The baseline reads the supplied files, uses the fixed validation split for training and pruning selection, and saves model and rule artifacts. Encoder fitting and rule extraction use training rows only. The declared budget is 8 CPUs and 8 GB RAM; each sequential baseline instance uses at most two threads. Measured training times for outer seeds 1, 2 and 3 are 12.03, 11.68 and 11.98 seconds; validation prediction times are 5.52, 5.58 and 5.52 seconds. Training includes rule extraction. Memory is a budget, not a measured peak.

## 5. Interface (frozen)

- `bash code/train.sh --out DIR`: train five independent replicates and save `rules.json` plus supporting artifacts.
- `bash code/predict.sh --model DIR --input PATH --output FILE`: write CSV predictions as specified in `bench/io_spec.md`; the output filename need not have an extension.
- Environment: `BENCH_SEED`, `BENCH_CPUS`, `BENCH_DATA_DIR`, `BENCH_BENCH_DIR`, `BENCH_CODE_DIR`. All paths may be absolute.
- Seeds: replicate s=0..4 uses `(5*(BENCH_SEED-1)+s) mod 2**32`. This maps the default outer seed 1 to five algorithm seeds 0..4. Honor this mapping for independent replicates.
- `bench status`, `bench eval-dev --seed S`, `bench validate-output FILE`, `bench submit --final -m MESSAGE` provide the standard development interface.
- Dependencies: Python 3.11; `uv pip install` in the provided environment. Submission freezes packages into `code/env.lock`. Only `name==version` pins from PyPI/PyTorch CPU indexes can be rebuilt; no editable, local or Git installs.

## 6. Rules

The research scope below determines which components may change. Supplied data and splits, trusted tests, information boundaries and the evaluation metric remain fixed.


- You may change sparse-rule learning, teacher/distillation models, training surrogate losses and optimization under `code/`, while preserving the five independent replicates and faithful constrained rule outputs.
- `data/`, `bench/`, this README and the prediction format are frozen. Patches touching them are invalid.
- The allocation is 8 CPUs and 8 GB memory. Training and prediction have no network access.
- Learn the submitted rules during training. You may learn them directly or train any model and distill/extract its rule representation; the original TT-Sparse model and `explain` routine are not required. `rules.json` must remain unchanged during prediction, and the CSV certificate must equal it. Prediction can evaluate those rules on the supplied rows, but cannot fit or select a new certificate using prediction inputs.
- Reading or reconstructing hidden targets, memorizing paper results, hard-coding row predictions, downloading external data or bypassing frozen rule inference is disqualifying. Rules must be learned with the supplied independent seeds and permitted data; input-specific lookups are disallowed.
- The judge checks three outer seeds; variation between five-replicate groups affects the PASS threshold.

## 7. Known pitfalls

- The metric is the mean of five AUCs, not an ensemble AUC. Keep all five probability columns.
- Rule complexity includes the intercept and continuous skip terms. The scorer recomputes it; a reported complexity counter is not accepted.
- Exact encoder state matters: skip features are float32 and thermometer thresholds retain full precision. Parsing only the rounded text of extracted comparisons can change predictions. Use the JSON encoder representation and frozen helpers.
- CSV has one nonempty certificate cell, correctly quoted. `certificate.write_prediction` handles quoting. The final column is numeric; missing rows, duplicate ids, NaN and infinities fail validation.
- Both entry scripts export their own code path because `PYTHONSAFEPATH=1` is enabled. Frozen scorer helpers import by their own directory.
- Runtime data must come from the supplied directories and artifacts saved under `--out`. Only committed files are copied into the judge's clean code checkout.

## Research scope

**Objective.** Learn five independent sparse interpretable Boolean-rule predictors for the supplied diabetes classification problem, maximizing mean ROC-AUC with mean rule complexity<=14.

**Fixed research objects.** supplied eight-feature diabetes observations, target and materialized partitions; Boolean-rule/encoder certificate semantics, frozen rule inference, mean complexity accounting and five independent AUCs.

**Editable methods.** direct sparse-rule learning, training surrogate losses, optimization, feature-threshold selection and pruning; training a different teacher/estimator and distilling to a faithful admissible sparse-rule certificate.

**Forbidden substitutions.** replacing the constrained interpretable output with an unrestricted black-box predictor or unfaithful explanatory rules; changing complexity accounting, fitting a new certificate at prediction time or changing certificate between input batches. When retaining the supplied TT-Sparse pipeline, changing only max_drop_pct=2, finetune_epochs=30, max_iterations=80, num_bits=9, n_bits=4, num_nodes=30, tau=0.05, dropout=0.1, lr=0.005, epochs=200 or patience=25 does not count as a method improvement.

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
