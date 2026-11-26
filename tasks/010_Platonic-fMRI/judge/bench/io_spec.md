# I/O specification

`--input PATH` is a directory containing `subject_S.npz` files, one per subject S. Each file has exactly these arrays:

- `id`: sorted unique int64 trial identifiers, the original stimulus-table row numbers.
- `betas`: int16 matrix of raw trial responses, shape N by V; V depends on the subject. Divide by 300 before the baseline's preprocessing.
- `run`: int64 vector of acquisition session/run codes, shape N. The baseline fits a categorical confound residualizer.

Inputs contain neither image identities nor repetition matching labels. Every embedding must be a function of that trial's response, acquisition run, subject and the saved trained model. Trial ids serve only to identify output rows. Joint matching, adaptation to the evaluation batch, and inference of image identity from ids are prohibited. Use the acquisition run only to remove acquisition confounds fitted on training trials; an embedding must not carry run, session or trial-order information as a retrieval feature.

`--output FILE` is an NPZ archive written to the exact supplied filename, regardless of extension. For each input subject S it contains `id_S` (identical to input ids, same order) and `embedding_S` (finite floating N by D matrix with 1 <= D <= 128). No additional keys. The method chooses its dimension within this fixed compact-representation limit.

Training data comprise `data/train/subject_S.npz` and `stimulus_S.csv`. CSV columns are `id,subject,nsd_id,exists,shared,subject_index,repetition`; `subject_index` addresses the compact training response matrix. Missing presentations use -1 and exists=False. Image groups and their missing repetitions are kept intact. Training and evaluation observations are disjoint.

Validation labels `data/val/labels.npz` contain `id_S,group_S,repetition_S`. The scorer aligns predictions by trial id, selects complete three-repetition groups, sorts by group, and averages equally across subjects and repetition pairs (0,1), (0,2), (1,2). Retrieval uses cosine nearest neighbor from the second repetition to the first, including self-image matches. RSA is Pearson correlation of upper-triangular Pearson-distance matrices across repetitions. Incomplete groups remain available for analysis but do not enter these metrics.

Scorer output key `recall_at_1` is higher-is-better. Every actual embedding array must have 1 <= D <= 128; invalid dimensions fail validation and scoring. If mean RSA is below the split's threshold (0.51 on test, 0.48 on val) or nonfinite, the mathematical score is 0 (`score_rule_value`); the harness failure sentinel is `Infinity` in `recall_at_1`. Otherwise both fields equal mean R@1. No self-reported metric is accepted.

The model directory must contain everything needed to transform new raw trials, including all fitted preprocessing and encoder parameters used by the method. The baseline saves clipping/centering statistics and confound, PCA/MCCA and encoder parameters. It also writes a descriptive `model_card.json`; embedding dimension is reported from the actual arrays. `check_consistency.py` tests that subset prediction agrees with full-batch prediction and that prediction does not rewrite the trained model.

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
