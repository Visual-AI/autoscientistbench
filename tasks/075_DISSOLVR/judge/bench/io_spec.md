# I/O specification

Input is CSV `id,SMILES`, sorted by numeric id. IDs are original curated-table row
numbers. SMILES are the released strings; inputs never contain labels.
Training CSV is `data/train/train.csv`, columns `id,SMILES,y`; y is log10 aqueous
solubility in mol/L. Validation labels use `id,y`.

Prediction output is CSV with exactly this header:
`id,pred_0,pred_1,pred_2,pred_3,pred_4,train_s,train_only`.
One row per input, same IDs in the same order, no duplicates, all numbers finite.
Each prediction column is one independently trained model's predicted logS.
The metric is the mean of five separately computed RMSEs, not the RMSE of an
averaged prediction. Training metadata must be constant across rows:
`train_s` is nonnegative elapsed seconds for the entire training process,
including feature calculation and all five fits; `train_only` is 1 exactly when
only training-file labels were used for fitting. Scores are Infinity if
`train_only != 1`. Invalid tables also score Infinity.
Metadata is an auditable claim, not proof; external timing and patch review apply.

Save all five models and required metadata in `--out`. The baseline uses
`model_0.joblib` through `model_4.joblib`, `label_usage.json`, `panel.json`, and
`train_meta.json`. Call frozen `bench/run_train.py` from train.sh to measure the
training subprocess. The outer process measurement is authoritative.

Five base model seeds are 42, 101, 123, 456, 789; add `BENCH_SEED - 1` to each
(modulo 2**32). Thus a run is a complete five-model experiment. Prediction must
load those models without fitting on prediction input or labels.

Research scope: The fixed research object is aqueous logS prediction from the supplied SMILES. Explicit, named molecular descriptors, regressors on them and training surrogate losses may change, while train.csv-only fitting labels, the five independent seed models, no prediction-time refitting and mean-of-five RMSE remain fixed. Each prediction column must come from one regressor fit with its one supplied seed; averaging several independently seeded fits inside a column counts as substituting ensemble RMSE.

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
