# I/O specification

## Input (`--input PATH`)
CSV with header `id,<feature columns...>`; one row per instance. Full history / full context is given — the method
decides how much of it to use (no method hyperparameters are baked into this format).

## Output (`--output FILE`)
CSV with header `id,pred`; exactly one row per input `id` (same set of ids, any order); `pred` numeric, no NaN/inf.

## Scoring
`bench/scorer.py <pred> <labels>` joins on `id` and prints a JSON object `{"<metric>": value}`.

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
