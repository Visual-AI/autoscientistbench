# 042_FrontDoor-NP I/O

`train.sh --out DIR` writes a model marker. This is per-dataset estimation, so there
is no shared model-training phase. `predict.sh --model DIR --input FILE --output FILE`
estimates the front-door ATE and a nominal 95% interval for each input dataset.

Input is a CSV with `case_id,X,A,M,Y,AX,MX`; every case has 500 rows. Only observed
quantities are present; the unobserved confounder U, population truth and generation
seeds are absent. The derived columns AX and MX equal A*X and M*X.

Output CSV columns are exactly `case_id,estimate,lower,upper`, one row for each input
case. IDs must match exactly, estimates and bounds must be finite, and lower <= upper.
The frozen scorer computes MSE against labels and returns +Infinity if strict-bound
coverage is below 0.887. Full validation contains 1000 datasets.

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
