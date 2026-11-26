# Host-record overlap interface

Input JSON has mode and allowed_assemblies. Full runs read immutable released observations, taxonomy and existing RNA caches from BENCH_DATA_DIR/train. Smoke additionally supplies eight unchanged existing records, leaving full inputs intact.

Output exactly {"assemblies":["GCF_...",...]}: one unique assembly ID per selected host record. Multiple assemblies sharing a species remain separate denominator records. IDs must occur in the public input universe. The frozen input taxonomy determines identities; candidate-supplied species labels or scores are not accepted.

The original global list-size guard is45 through182 inclusive. Scoring then selects the evaluation population, performing the original maximum one-to-one species-taxid matching and normalized-name fallback for unresolved ranks. Resolved taxid disagreements never match by name. F1=2TP/(predicted records+curated records),worst0 if global size fails.

Identity components conservatively join species aliases and normalized names. Validation contains 20% of groups and is used only for evaluation and method or hyperparameter selection. Do not fit models or learned preprocessing on its features or labels. No curated list is an input to the method.

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
