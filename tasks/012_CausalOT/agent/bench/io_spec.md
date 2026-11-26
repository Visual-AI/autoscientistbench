# Input and prediction interface

Input is a NumPy NPZ file (`allow_pickle=False`). `id` is a one-dimensional Unicode array of unique opaque instance IDs. For each position i, `Y0_i`, `Z0_i`, `Y1_i`, `Z1_i` contain the two groups' observed outcomes and covariates as finite float64 matrices. Outcomes and covariates lie in [0,1]. Each pair has equal row count. Every instance has (dY,dZ,n=m)=(2,1,3000) and its own distribution. Neither instance ordering nor IDs carry information about the observations.

Output is UTF-8 CSV with the exact header `id,point,lower,upper`. Supply one finite row per input ID, with the same ID set and no duplicate or additional rows. Row order is arbitrary. `lower <= upper`; the point estimate and interval may be produced by any observations-only method, including asymmetric intervals. Write exactly the supplied output path, which may have no extension.

The baseline implementation calls its normal approximation interval “basic”: point ± normal.ppf(.975) * std(100 bootstrap estimates, ddof=1). That describes the baseline construction; alternatives may use different interval methods. The benchmark evaluates length and empirical coverage of the submitted intervals; their scientific construction and observations-only restriction require patch audit.

Labels are NPZ with `id` and `truth`, the instance's COT value for its distribution before clipping. Public validation labels are for evaluation and configuration selection only, not for fitting; labels for hidden evaluation are held by the judge.

`python bench/validate.py PRED INPUT` exits nonzero on an invalid table. `python bench/scorer.py PRED LABELS` prints one JSON line. Metric `ci_length_gmean` is the mean interval length if coverage is at least 0.80 on test or 0.75 on val, otherwise Infinity. Coverage includes both endpoints. A zero mean length scores zero when coverage passes. The target `truth` is the instance's COT value for its distribution before clipping, even though observed outcomes are clipped.

Research scope: Density estimators, OT solvers and valid interval constructions may change for this same COT target. Bootstrap resampling and internal integration from distributions fitted to the current observations are allowed; simulating from an assumed or reconstructed population model, truth-table fitting and cross-instance pooling are prohibited. The estimator must stay nonparametric: it may assume smooth densities, but it may not assume a parametric form for the conditional laws, such as a Gaussian or location-scale model, and it may not evaluate the closed-form value of such a model at fitted parameters. Intervals must target nominal 95% coverage: do not lower the nominal level, scale widths by a factor tuned against validation coverage, or return degenerate intervals on a subset of instances.

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
