# Existing-read surveillance interface

Input directory manifest.json lists samples with opaque id,r1,r2. FASTQs contain existing paired observations only; headers and order are scrubbed. The fixed existing surveillance tree and reference are in data/train/reference. No mixture size, seed, expected lineage label, original sequence-origin identifier or generating haplotype is provided to candidate input.

Prediction JSON is {"samples":[{"id":"sample-id","abundances":[["lineage",0.1],...]}]}. Every input sample appears exactly once; lineage names are unique nonempty strings and abundances finite nonnegative numbers. Missing estimated lineages mean zero. Empty lists are allowed and receive the unchanged missed-lineage penalty. Duplicate JSON keys/lineages/samples,NaN,infinity,negative values,extra or missing samples are invalid.

Score: mean over samples of sqrt(mean squared abundance error over union of expected names and estimated names with abundance at least 0.005)). In every sample, at most one positive expected lineage may have estimated abundance below0.005, and total estimated abundance on non-simulated lineages must be at most0.02. Any failed guard sets rmse to +inf. No extra normalization or estimated-total-mass gate is added. Complete balanced replicate blocks make this sample mean identical to the original mean of replicate-block means for the selected mixture scales.

Only lineage abundance analysis is performed. Optional sequence-bearing diagnostic BAM export and all simulated-sequence generation are absent. The baseline uses its original placement/deconvolution functions; candidates may replace the abundance-estimation method while respecting the same inputs and score guards. Tiny smoke selects existing reads explicitly and does not change default input populations or scoring guards.

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
