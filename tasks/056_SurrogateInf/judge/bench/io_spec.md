# PII input and prediction contract

`--input PATH` is an NPZ file loaded with `allow_pickle=False`. With R instances it contains:

- `id`: R opaque instance identifiers.
- `X`: float64, shape (R,100,1), observed covariate.
- `Y`: binary observations, shape (R,100,1000).
- `C`: integer shape (R,500), indices of known surrogate controls, in the supplied order.
- `hypothesis_id`: strings, shape (R,500), identifies outcomes in ascending complement-of-C order.

No generating seeds, latent factors, or tested-outcome true coefficients are input features.
Each instance must be processed independently using only its own observations.

Output is CSV with exact header `id,beta,var,p_value`, one row for every hypothesis_id
(any order, no duplicates). beta is the coefficient estimate, var its strictly positive
sampling variance, and p_value is the candidate method's two-sided p value for the
specified zero-effect hypothesis. All numbers must be finite; p_value is in [0,1].
The baseline uses 2*(1 - t_cdf(abs(beta/sqrt(var)),df=99)); other valid
test statistics and p-value calibration procedures are allowed.

Validation labels in `data/val/truth.npz` contain instance IDs, hypothesis IDs, tested
true coefficients B, latent U, n, and public instance seeds. Baseline per-instance
metrics may be stored for diagnostics; the score uses the fixed thresholds below.

Power = mean across replications of the fraction of non-null outcomes with p_value < .05.
Type-I error = mean across replications of the null rejection fraction. BH FDR = mean
replication FDP after Benjamini–Hochberg adjusted p_value < .05, with zero FDP if no
hypothesis is rejected. Score is power if type-I error <= .069 and BH FDR <= .085;
otherwise it is 0. Invalid output yields Infinity.

The scorer uses only predictions and labels, independently of mutable method code.

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
