# ALPINE read classification

Input is a directory: manifest.json, four FASTQ.GZ observation pools, and unchanged existing per-site reference/config files. Each manifest entry gives pool,file,ids,site,data_type. Read IDs are opaque; no true outcome labels or original simulation names are inputs. No new sequence generation is part of this task.

Prediction JSON has exactly pools: a map containing hifi_trac,hifi_trbc,ont_trac,ont_trbc. Each pool maps every input read ID to one class string. Duplicate keys, missing/extra pools or reads, malformed IDs/classes are rejected. Filtered/unclassified reads still need the Unclassified label and count in the denominator.

Frozen scoring uses the original evaluate_pool function byte-for-byte: per-pool integration and Total F1 rounded with Python round(...,2), then mean of the two ONT Total F1 values. All four pools require integration F1>=baseline-.2 percentage points and Total F1>=baseline-.5; otherwise f1=0. Fixed (integration,Total) guard reference values: hifi_trac(98.45,96.08),hifi_trbc(99.44,95.64),ont_trac(99.62,87.16),ont_trbc(99.63,86.91). Integration is exact class-level TP/FP/FN aggregated across the five integration groups, not binary integration/non-integration detection.

The submitted method is applied to full and smoke inputs; ALPINE/CWL defaults describe the baseline, not a required model. Only the explicit smoke fixture contains fewer existing reads. A smoke score0 is valid when unchanged performance guards fail; it is not a runtime failure.

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
