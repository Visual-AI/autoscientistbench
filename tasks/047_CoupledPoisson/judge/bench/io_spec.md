# 047_CoupledPoisson interface
Input CSV has exactly `case_id,initial_1,initial_2,initial_3,initial_4`: each row gives four independent observed initial positions from N(0,4²), used in order for the two pairs of coupled chains. It contains no generation seed or estimand. The known AR(1) kernel is supplied by the frozen runner.

Output CSV has exactly `case_id,estimate,reported_cost,single,coupled`, with one row per input. `estimate` is finite; ledger counters are nonnegative integers. The frozen runner increments `single` once for each single-kernel call and `coupled` once for each coupled-kernel call. `reported_cost` from the candidate method must equal `single + 2*coupled`; scoring uses the latter. These columns are a transparent execution ledger, not a cryptographically authenticated receipt. Counter bypass, direct simulation outside callbacks, ledger forgery, and constant answers require patch review and are forbidden.

Full scoring requires 1000 independent estimates and uses the unbiased sample variance (denominator M−1), times mean counted cost. If the empirical mean differs from 10000 by more than three standard errors, score is +infinity.

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
