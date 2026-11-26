# 061_DP-FRT interface
Input CSV columns: id,n1,n0,epsilon,t11,t01. Only the sample margins, privacy budget and clipped privatized counts are observed. No original counts, case labels, true p-value or observation-generation seed is included in an input row. Public validation labels map opaque IDs to the effect case and true p_FRT rounded to 12 decimals.

Output CSV columns: id,lo,hi,credible_set. Include exactly one row per input ID. credible_set is a semicolon-separated, strictly increasing finite sequence in [0,1], whose first/last values equal lo/hi. The score tests discrete set membership after rounding to 12 decimals. Width is hi-lo.

Validation has all 36 cells with 1000 releases per cell. An independent smoke fixture has one release per cell; the baseline uses 10000 posterior draws and the same coverage guards apply. Candidate inference methods may differ while preserving the uniform count prior, geometric-mechanism posterior, nonprivate Fisher right-tail p-value and central 95% discrete posterior-set definition. Endpoints are the 2.5%/97.5% posterior quantiles, with supported p-values between them included; sampling or enumeration may compute the same object. There is no training split. Validation is for evaluation and selection only; per-instance posterior inference uses only that instance's observations.

The task-specific Research contract in README is binding: output validity alone does not permit substituting a different research object or objective.

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
