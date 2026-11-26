# TARPON boundary decisions

Input directory has manifest.json and BAM observations. Header and IDs are sanitized; sequence,quality,observed XS strand are preserved. Candidate output JSON has samples map, sample name to every input read ID, with exactly status (retained/no_start/filtered) and start (integer0<=start<read length for retained; null otherwise). No dropped/extra samples/reads or submitted composition statistics are accepted.

Frozen scorer recalculates WT and1N composition from original observed sequence tails using the authors exact nonoverlapping regex rule. Denominator is all boundary-stage input reads and must equal the fixed original count for each sample. Mean purity must be >= baseline-0.5 percentage points per sample. Per sample, the median tail must be >= 0.9 times `median_tail_baseline`, and retained reads with tails shorter than 1000 bases count only up to `short_tail_baseline`. Score is unweighted mean identification rate across the current split; any guard failure gives0. It computes composition independently rather than trusting candidate percentages. Boundaries themselves are candidate outputs; the original guards do not establish biological boundary accuracy beyond purity and fixed population.

Upstream original subtelomere-pass filtering is frozen as input; the boundary method is editable and may be replaced; the submitted method executes on every input. Full datasets are untruncated. Both WB60 libraries are validation data. There is no training split. Validation is only for evaluation and selection; do not fit reusable models or learned preprocessing on it. Inference at prediction time may use the reads in its own input.

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
